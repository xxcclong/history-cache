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


class HistoryTrainer(cxgnncomp.Trainer):

    def __init__(self, config):
        # self.table = HistoryTable(config)
        self.loader = cxgnndl.get_loader(config.dl)
        assert self.loader.feat_mode in ["history_uvm", "history_mmap"]
        config.dl.loading.feat_mode = self.loader.feat_mode.replace(
            "history_", "")
        self.uvm = cxgnndl.UVM(config.dl)
        self.table = HistoryCache(config=config, uvm=self.uvm)
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

    def prepare_history_x(self, batch):
        if self.model.training:
            # self.table.history_out = []
            self.table.lookup_and_load(batch, len(self.model.convs))
            self.used_masks = hiscache.count_history_reconstruct(
                batch.ptr,
                batch.idx,
                self.table.sub_to_history_layered,
                batch.sub_to_full.shape[0],
                batch.y.shape[0],
                len(self.model.convs),
            )
            # self.used_masks[0] = torch.zeros_like(self.used_masks[0])
            # self.used_masks[0] = (torch.randn(
            #     self.used_masks[0].shape, device=self.used_masks[0].device)>0)

            num_uses = []
            for it, mask in enumerate(self.used_masks):
                num_uses.append(mask.sum().item())
            num_uses.reverse()
            arr = []
            for it, item in enumerate(num_uses):
                arr.append(item / batch.num_node_in_layer[it])
            log.info(f"num_uses: {arr}")
            log.info(batch.num_node_in_layer)
            degree = batch.ptr[1:] - batch.ptr[:-1]
            dst = torch.arange(
                batch.ptr.shape[0] - 1).cuda().repeat_interleave(degree)
            remain = self.used_masks[1][dst].sum().item()
            log.info(f"remain: {remain / batch.idx.shape[0]}")
            # log.info(f"used masks: {self.used_masks[0].sum()} {self.used_masks[0].shape}")
            self.table.show()
            if self.uvm.has_cache:
                batch.x = self.uvm.cached_masked_get(batch.sub_to_full,
                                                     self.used_masks[0])
            else:
                batch.x = self.uvm.masked_get(batch.sub_to_full,
                                              self.used_masks[0])
            # log.info(f"{torch.sum(self.used_masks[0])} {self.used_masks[0].shape}")
            # log.info(time.time())
        else:
            batch.x = self.uvm.get(batch.sub_to_full)

    def cxg_dgl_train_epoch(self):
        self.model.train()
        for batch in tqdm(
                self.loader.train_loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            self.optimizer.zero_grad()
            blocks = self.to_dgl_block(batch)
            self.prepare_history_x(batch)
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

    def cxg_train_epoch(self):
        self.model.train()
        t0 = time.time()
        for batch in tqdm(
                self.loader.train_loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            self.optimizer.zero_grad()
            self.table.lookup_and_load(batch, len(self.model.convs))
            # self.prepare_history_x(batch)

            out = self.model(batch)
            loss = self.loss_fn(out, batch.y)
            torch.cuda.synchronize()
            loss.backward()
            torch.cuda.synchronize()
            self.optimizer.step()
            self.scheduler.step()
            # self.table.evict_history(batch, self.glb_iter)
            # self.table.record_history(batch.sub_to_full, batch, self.glb_iter,
            #                           self.used_masks)
            self.table.update_history(batch, self.glb_iter)
            self.glb_iter += 1
        torch.cuda.synchronize()
        log.info(f"epoch time: {time.time()-t0}")

    def train(self):
        for epoch in range(self.config.train.train.num_epochs):
            if self.type == "dgl" and self.load_type == "cxg":
                self.cxg_dgl_train_epoch()
            elif self.type == "cxg" and self.load_type == "cxg":
                self.cxg_train_epoch()
            else:
                assert False
