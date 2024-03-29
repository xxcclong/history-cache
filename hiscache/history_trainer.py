from tqdm import tqdm
import cxgnncomp
import cxgnndl

import hiscache
from .history_table import HistoryTable
from .history_cache import HistoryCache
import torch
from .history_model import get_model
from .util import log
import time
import numpy as np


class HistoryTrainer(cxgnncomp.Trainer):

    def __init__(self, config):
        # self.table = HistoryTable(config)
        self.loader = cxgnndl.get_loader(config.dl)
        self.feat_mode = config.dl.loading.feat_mode
        config.dl.loading.feat_mode = self.loader.feat_mode.replace(
            "history_", "")
        if self.feat_mode == "memory":
            self.uvm = None
        else:
            self.uvm = cxgnndl.UVM(config.dl)
        self.table = HistoryCache(uvm=self.uvm,
                                  config=config,
                                  mode=self.feat_mode)
        self.device = torch.device(config.dl.device)
        self.model = get_model(config, self.table)
        self.model = self.model.to(self.device)
        self.optimizer = cxgnncomp.get_optimizer(config.train, self.model)
        self.scheduler = cxgnncomp.get_scheduler(config.train, self.optimizer)
        self.loss_fn = cxgnncomp.get_loss_fn(config.train)
        self.type = config.train.type.lower()
        self.load_type = config.dl.type.lower()
        self.config = config
        self.glb_iter = 0
        self.prof = config.dl.performance.prof
        self.evaluator = cxgnncomp.get_evaluator(config.train)
        self.val_metrics = []
        self.test_metrics = []
        self.best_metric = 0
        self.cached_emd_id = None

        # self.stream = torch.cuda.Stream(device=self.device)  # Create a new stream.

    def cxg_dgl_train_epoch(self):
        self.model.train()
        for batch in tqdm(
                self.loader.train_loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            self.optimizer.zero_grad()
            blocks = self.to_dgl_block(batch)
            self.table.lookup_and_load(batch, len(self.model.convs))
            batch_inputs = batch.x
            batch_labels = batch.y
            batch_pred = self.model([blocks, batch_inputs])
            loss = self.loss_fn(batch_pred, batch_labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            # self.table.evict_history(batch, self.glb_iter)
            # self.table.record_history(batch.sub_to_full, batch, self.glb_iter,
            #                           self.used_masks)
            self.table.update_history(batch, self.glb_iter)
            self.glb_iter += 1
        torch.cuda.synchronize()

    def record(self, batch):
        batch.y.record_stream(self.stream)
        batch.ptr.record_stream(self.stream)
        batch.idx.record_stream(self.stream)
        batch.sub_to_full.record_stream(self.stream)

    def cxg_train_epoch_prof(self):
        self.model.train()
        tepoch = time.time()
        t5 = time.time()
        times = {
            "load": 0,
            "forward": 0,
            "backward": 0,
            "loss": 0,
            "update": 0,
            "sample": 0,
            "step": 0,
        }

        for batch in tqdm(
                self.loader.train_loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            self.optimizer.zero_grad()
            torch.cuda.current_stream().synchronize()
            tbegin = time.time()
            times["sample"] += tbegin - t5
            self.table.lookup_and_load(batch, len(self.model.convs))
            torch.cuda.current_stream().synchronize()
            t0 = time.time()
            out = self.model(batch)
            torch.cuda.current_stream().synchronize()
            t1 = time.time()
            loss = self.loss_fn(out, batch.y)
            torch.cuda.current_stream().synchronize()
            t2 = time.time()
            loss.backward()
            torch.cuda.current_stream().synchronize()
            t3 = time.time()
            self.optimizer.step()
            self.scheduler.step()
            torch.cuda.current_stream().synchronize()
            t4 = time.time()
            self.table.update_history(batch, self.glb_iter)
            torch.cuda.current_stream().synchronize()
            t5 = time.time()
            # log.info(f"load-time: {t0-tbegin}, forward-time: {t1-t0}, loss-time: {t2-t1}, backward-time: {t3-t2}, step-time: {t4-t3}, update-time: {t5-t4}",
            #          extra={"iter": self.glb_iter})
            times["load"] += t0 - tbegin
            times["forward"] += t1 - t0
            times["loss"] += t2 - t1
            times["backward"] += t3 - t2
            times["step"] += t4 - t3
            times["update"] += t5 - t4

            self.glb_iter += 1
        torch.cuda.synchronize()
        log.info(f"epoch time: {time.time()-tepoch}")
        total = sum(times.values())
        log.info(
            f"load-time: {times['load']}, forward-time: {times['forward']}, loss-time: {times['loss']}, backward-time: {times['backward']}, step-time: {times['step']}, update-time: {times['update']}, sample-time: {times['sample']} total-time: {total}",
        )

    def batch_to_file(self, batch, filename):
        output_dict = {}
        output_dict["ptr"] = batch.ptr.cpu()
        output_dict["idx"] = batch.idx.cpu()
        output_dict["num_node_in_layer"] = batch.num_node_in_layer.cpu()
        # output_dict["num_edge_in_layer"] = batch.num_edge_in_layer.cpu()
        torch.save(output_dict, filename)
        exit()

    # test the hit rate of historical embeddings
    def cxg_just_load(self):
        rate = 0.9
        staleness = 200
        is_cached = torch.zeros(self.table.total_num_node, dtype=torch.int)
        cached_embs = []
        num_visit = torch.zeros(self.table.total_num_node, dtype=torch.int)

        for epoch in range(3):
            for batch in tqdm(
                    self.loader.train_loader,
                    bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"
            ):
                nodes = batch.sub_to_full[:batch.num_node_in_layer[1].item(
                )].cpu()
                num_visit[nodes] += 1
                uncached_nodes = nodes[is_cached[nodes] == 0]
                print(
                    f"history hitrate {torch.sum(is_cached[nodes] > 0) / nodes.shape[0]} {torch.sum(is_cached > 0)}"
                )
                rand = torch.randn(nodes.shape[0], )
                # rand = num_visit[nodes].float() * -1
                percentile = torch.quantile(rand, rate)
                nodes_to_check_in = nodes[torch.logical_and(
                    rand < percentile, is_cached[nodes] == 0)]
                cached_embs.append(nodes_to_check_in)
                if len(cached_embs) > staleness:
                    popped = cached_embs.pop(0)
                    is_cached[popped] -= 1
                is_cached[nodes_to_check_in] += 1
                self.glb_iter += 1
        exit()

    def cxg_train_epoch(self):
        self.model.train()
        tepoch = time.time()
        for batch in tqdm(
                self.loader.train_loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            # self.batch_to_file(batch, f"papers100M-{batch.num_node_in_layer[0]}.pt")
            self.optimizer.zero_grad()
            self.table.lookup_and_load(batch, len(self.model.convs))
            out = self.model(batch)
            loss = self.loss_fn(out, batch.y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.table.update_history(batch, self.glb_iter)
            self.glb_iter += 1
        torch.cuda.synchronize()
        log.info(f"epoch time: {time.time()-tepoch}")

    def cxg_eval_epoch(self, split="val"):
        self.model.eval()
        y_preds, y_trues = [], []
        losses = []
        if split == "val":
            loader = self.loader.val_loader
        else:
            loader = self.loader.test_loader
        tepoch = time.time()
        for batch in tqdm(
                loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            self.optimizer.zero_grad()
            self.table.lookup_and_load(batch,
                                       len(self.model.convs),
                                       feature_cache_only=True)
            out = self.model(batch)
            loss = self.loss_fn(out, batch.y)
            losses.append(loss.detach())
            y_preds.append(out.detach())
            y_trues.append(batch.y.detach())
        torch.cuda.synchronize()
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)
        self.metric = self.evaluator(y_preds, y_trues).item()
        log.info(f"epoch time: {time.time()-tepoch}")
        log.info(f"{split}-metric: {self.metric}")
        if split == "val":
            self.best_metric = max(self.best_metric, self.metric)
            log.info(f"best-metric: {self.best_metric}")
            self.val_metrics.append(self.metric)
        else:
            self.test_metrics.append(self.metric)

    def train(self):
        for epoch in range(self.config.train.train.num_epochs):
            log.info(f"Epoch {epoch}/{self.config.train.train.num_epochs}")
            if self.type == "dgl" and self.load_type == "cxg":
                self.cxg_dgl_train_epoch()
            elif self.type == "cxg" and self.load_type == "cxg":
                # with torch.cuda.stream(self.stream):
                if self.prof:
                    self.cxg_train_epoch_prof()
                else:
                    # self.cxg_just_load()
                    self.cxg_train_epoch()
            else:
                assert False
            if epoch >= self.config.train.train.eval_begin:
                self.cxg_eval_epoch(split="val")
                if not "mag" in self.config.dl.dataset.name.lower():
                    self.cxg_eval_epoch(split="test")
        if len(self.val_metrics) > 0:
            self.val_metrics = np.array(self.val_metrics)
            self.test_metrics = np.array(self.test_metrics)
            log.info(
                f"best-val-metrics: {self.val_metrics.max()} at epoch {self.val_metrics.argmax()}"
            )
            log.info(
                f"best-test-metrics: {self.test_metrics.max()} valided test-metric {self.test_metrics[self.val_metrics.argmax()]}"
            )
