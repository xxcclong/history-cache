from tqdm import tqdm
import cxgnncomp
import cxgnndl

from .history_cache import HistoryCache
import torch
from .history_model import get_model
from .util import log
import time


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

    def init_models(self, config):
        self.tables = []
        self.models = []
        self.optimizers = []
        self.schedulers = []
        for i in range(self.num_device):
            table = HistoryCache(config=config, uvm=self.uvm, device_id=i)
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
            batches[i].y = batches[i].ys[i]
            batches[i].ptr = batches[i].ptrs[i]
            batches[i].idx = batches[i].idxs[i]
            batches[i].sub_to_full = batches[i].sub_to_fulls[i]

        for i in range(self.num_device):
            torch.cuda.set_device(i)
            self.tables[i].lookup_and_load(batches[i], self.num_layer)

    def run_forward(self, batches):
        outs = []
        for i in range(self.num_device):
            self.optimizers[i].zero_grad()
            out = self.models[i](batches[i])
            outs.append(out)
        return outs

    def run_backward(self, batches, outs):
        grad = [[] for i in range(self.num_para)]
        for i in range(self.num_device):
            loss = self.loss_fn(outs[i], batches[i].y)
            loss.backward()
            for a, b in self.models[i].named_parameters():
                grad[self.parameter_name_to_id[a]].append(b.grad)
        return grad

    def update_grad(self, grad):
        for i in range(self.num_para):
            torch.cuda.nccl.all_reduce(grad[i])
        for dev_it in range(self.num_device):
            torch.cuda.set_device(dev_it)
            for para in self.models[dev_it].parameters():
                para.grad /= self.num_device
            self.optimizers[dev_it].step()
            self.schedulers[dev_it].step()

    def update_history(self, batches):
        for i in range(self.num_device):
            self.tables[i].update_history(batches[i], self.glb_iter)
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
                outs = self.run_forward(batches)
                grad = self.run_backward(batches, outs)
                self.update_grad(grad)
                self.update_history(batches)
                batches = []
        torch.cuda.synchronize()
        log.info(f"epoch time: {time.time()-t0}")

    def train(self):
        for epoch in range(self.config.train.train.num_epochs):
            if self.type == "cxg" and self.load_type == "cxg":
                self.cxg_train_epoch()
            else:
                assert False
