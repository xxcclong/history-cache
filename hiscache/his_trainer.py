from torchmetrics import UniversalImageQualityIndex
from tqdm import tqdm
import cxgnncomp
import cxgnndl

import hiscache
from .history_table import HistoryTable
import torch


class HistoryTrainer(cxgnncomp.Trainer):

    def __init__(self, config):
        super().__init__(config)
        self.table = HistoryTable(config)
        assert self.loader.feat_mode in ["history_uvm", "history_mmap"]
        config.dl.loading.feat_mode = self.loader.feat_mode.replace(
            "history_", "")
        self.uvm = cxgnndl.UVM(config.dl)
        self.glb_iter = 0

    def prepare_history_x(self, batch):
        if self.model.training:
            self.table.history_out = []
            self.table.lookup(batch.sub_to_full, batch.num_node_in_layer)
            self.used_masks = hiscache.count_history_reconstruct(
                batch.ptr,
                batch.idx,
                self.table.sub_to_history_layered,
                batch.sub_to_full.shape[0],
                batch.y.shape[0],
                len(self.model.convs),
            )
            batch.x = self.uvm.masked_get(batch.sub_to_full,
                                          self.used_masks[0])
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
            self.table.evict_history(batch, self.glb_iter)
            self.table.record_history(batch.sub_to_full, batch, self.glb_iter,
                                      self.model.used_masks)
            self.glb_iter += 1
        torch.cuda.synchronize()

    def train(self):
        if self.type == "dgl" and self.load_type == "cxg":
            self.cxg_dgl_train_epoch()
