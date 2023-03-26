from tqdm import tqdm
import cxgnncomp
import cxgnndl

from .history_cache import HistoryCache
import torch
from .history_model import get_model
from .util import log
import time
from .comm import prepare_transfer, all2all
import numpy as np


class MultiGpuHistoryTrainer(cxgnncomp.Trainer):

    def __init__(self, config):
        self.num_device = int(config.dl.num_device)
        assert self.num_device > 1
        self.loader = cxgnndl.get_loader(config.dl)
        assert self.loader.feat_mode in ["history_uvm", "history_mmap"]
        config.dl.loading.feat_mode = self.loader.feat_mode.replace(
            "history_", "")
        self.uvm = cxgnndl.UVM(config.dl)
        self.init_models(config)
        self.num_layer = len(self.models[0].convs)
        self.loss_fn = cxgnncomp.get_loss_fn(config.train)
        self.type = config.train.type.lower()
        self.load_type = config.dl.type.lower()
        self.config = config
        self.glb_iter = 0
        self.val_metrics = []
        self.test_metrics = []
        self.best_metric = 0
        self.evaluator = cxgnncomp.get_evaluator(config.train)

    def init_models(self, config):
        self.tables = []
        self.models = []
        self.optimizers = []
        self.schedulers = []
        for i in range(self.num_device):
            table = HistoryCache(config=config,
                                 uvm=self.uvm,
                                 mode=self.loader.feat_mode,
                                 device_id=i)
            model = get_model(config, table)
            model = model.to(i)
            optimizer = cxgnncomp.get_optimizer(config.train, model)
            scheduler = cxgnncomp.get_scheduler(config.train, optimizer)
            self.tables.append(table)
            self.models.append(model)
            self.optimizers.append(optimizer)
            self.schedulers.append(scheduler)
        self.num_para = 0
        self.parameter_name_to_id = {}
        for a, b in self.models[0].named_parameters():
            self.parameter_name_to_id[a] = self.num_para
            self.num_para += 1

    def load(self, batches):
        for i in range(self.num_device):
            self.optimizers[i].zero_grad()

        for i in range(self.num_device):
            batches[i].y = batches[i].ys[i]
            batches[i].ptr = batches[i].ptrs[i]
            batches[i].idx = batches[i].idxs[i]
            batches[i].sub_to_full = batches[i].sub_to_fulls[i]
        outputs = []
        for i in range(self.num_device):
            torch.cuda.set_device(i)
            outputs.append(self.tables[i].lookup_and_load(
                batches[i], self.num_layer))
        if outputs[0] is not None:  # distributed store
            tensor_list = prepare_transfer(batches, outputs, [
                self.tables[i].distributed_buffer
                for i in range(self.num_device)
            ])
            all2all(tensor_list)

    def run_forward(self, batches):
        times = []
        t0 = time.time()
        outs = []
        for i in range(self.num_device):
            torch.cuda.set_device(i)
            out = self.models[i](batches[i])
            outs.append(out)
            times.append(time.time() - t0)
            # torch.cuda.synchronize(i)
        # log.info(f"forward time: {times}")
        return outs

    def run_backward(self, batches, outs):
        times = []
        t0 = time.time()
        grad = [[] for i in range(self.num_para)]
        for i in range(self.num_device):
            loss = self.loss_fn(outs[i], batches[i].y)
            loss.backward()
            for a, b in self.models[i].named_parameters():
                grad[self.parameter_name_to_id[a]].append(b.grad)
            times.append(time.time() - t0)
            # torch.cuda.synchronize(i)
        # log.info(f"backward time: {times}")
        return grad

    def update_grad(self, grad):
        times = []
        t0 = time.time()
        for i in range(self.num_para):
            torch.cuda.nccl.all_reduce(grad[i])
            times.append(time.time() - t0)
        # log.info(f"all_reduce time: {times}")
        times = []
        t0 = time.time()
        for dev_it in range(self.num_device):
            torch.cuda.set_device(dev_it)
            for para in self.models[dev_it].parameters():
                para.grad /= self.num_device
            self.optimizers[dev_it].step()
            self.schedulers[dev_it].step()
            times.append(time.time() - t0)
        # log.info(f"update time: {times}")

    def update_history(self, batches):
        times = []
        t0 = time.time()
        for i in range(self.num_device):
            self.tables[i].update_history(batches[i], self.glb_iter)
            times.append(time.time() - t0)
        # log.info(f"update_history time: {times}")
        self.glb_iter += 1

    def cxg_train_epoch(self):
        for i in range(self.num_device):
            self.models[i].train()
        t0 = time.time()
        batches = []
        for batch in tqdm(
                self.loader.train_loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            batches.append(batch)
            if len(batches) == self.num_device:
                self.load(batches)
                # self.run_nonsense(batches)
                outs = self.run_forward(batches)
                grad = self.run_backward(batches, outs)
                # batches = []
                # continue
                self.update_grad(grad)
                self.update_history(batches)
                batches = []
        for i in range(self.num_device):
            torch.cuda.synchronize(i)
        log.info(f"epoch time: {time.time()-t0}")

    def cxg_eval_epoch(self, split="val"):
        for i in range(self.num_device):
            self.models[i].eval()
        y_preds, y_trues = [], []
        losses = []
        if split == "val":
            loader = self.loader.val_loader
        else:
            loader = self.loader.test_loader
        tepoch = time.time()
        batches = []
        for batch in tqdm(
                loader,
                bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            batches.append(batch)
            if len(batches) == self.num_device:
                self.load(batches)
                outs = self.run_forward(batches)
                for i in range(self.num_device):
                    loss = self.loss_fn(outs[i], batches[i].y)
                    losses.append(loss.detach())
                    y_preds.append(outs[i].detach().to("cpu"))
                    y_trues.append(batches[i].y.detach().to("cpu"))
                batches = []
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
            if self.type == "cxg" and self.load_type == "cxg":
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
