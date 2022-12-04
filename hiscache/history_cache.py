import torch
from os import path
import hiscache
from .util import log
import numpy as np


class HistoryCache:
    def __init__(self, uvm, config):
        self.in_channels = config.dl.dataset.feature_dim
        self.out_channels = config.dl.dataset.num_classes
        self.hidden_channels = config.train.model.hidden_dim
        self.num_layers_all = config.train.model.num_layers
        self.total_num_node = int(
            open(
                path.join(str(config["dl"]["dataset"]["path"]), "processed",
                          "num_nodes.txt")).readline())
        self.device = torch.device(config["dl"]["device"])
        # history embedding settings
        self.method = config.history['method']
        self.rate = float(config.history['rate'])
        assert self.method in ["random", "grad", "staleness"]
        self.staleness_thres = int(config.history.staleness_thres)

        # self.embed2full = torch.empty([self.total_num_node,])
        self.full2embed = torch.ones(
            [self.total_num_node * 2, ], dtype=torch.int64, device=self.device) * -1
        self.header = torch.zeros([1], dtype=torch.int64, device=self.device)
        self.produced_embedding = None
        self.allocate()
        self.uvm = uvm
        self.fill(config)

    def allocate(self):
        max_byte = torch.cuda.mem_get_info()[0]
        max_byte = max_byte - 1024 * 1024 * 1024
        max_buffer_byte = max_byte // (self.hidden_channels *
                                       4 + 8) * (self.hidden_channels * 4)
        max_buffer_byte = max_buffer_byte // self.hidden_channels * self.hidden_channels
        max_buffer_byte = max_buffer_byte // self.in_channels * self.in_channels
        self.buffer = torch.empty(
            [max_buffer_byte // 4], device=self.device)  # 4 bytes per float
        self.embed2full = torch.ones(
            self.buffer.shape[0] // self.hidden_channels, dtype=torch.int64, device=self.device) * -1
        self.overall_feat_num = self.buffer.shape[0] // self.in_channels
        self.overall_embed_num = self.buffer.shape[0] // self.hidden_channels
        log.info(f"max buffer size: {self.buffer.shape}")

    def fill(self, config):
        # fill with raw features
        ptr_datapath = path.join(str(config["dataset"]["path"]), "processed",
                                 "csr_ptr_undirected.dat")
        ptr = torch.from_numpy(np.fromfile(ptr_datapath,
                                           dtype=np.int64))
        deg = ptr[1:] - ptr[:-1]  # in degree
        del ptr
        sorted, indice = torch.sort(deg)
        del sorted
        overall_num_node = deg.shape[0]
        del deg
        if overall_num_node < self.overall_feat_num:
            self.overall_feat_num = overall_num_node
            log.warn("overall_num_node < self.overall_feat_num")
        cache_indice = indice[-self.overall_feat_num:]
        new_id = torch.arange(0, len(cache_indice))
        self.full2embed[self.total_num_node +
                        cache_indice] = new_id

    def update_history(self, batch, glb_iter, used_masks):
        num_node = batch.num_node_in_layer[1].item()
        if glb_iter % self.staleness_thres == 0:
            self.header.zero_()
        grad = self.produced_embedding.grad
        used_mask = used_masks[2][:num_node]
        assert grad.shape[0] == num_node, "grad shape: {}, num_node: {}".format(
            grad.shape, num_node)
        grad = grad.norm(dim=1)
        thres = torch.quantile(grad, self.rate)
        record_mask = grad < thres and used_mask
        # record
        self.produced_embedding = self.produced_embedding[record_mask]
        self.buffer.view(-1, self.hidden_channels)[
            self.header: self.header+self.produced_embedding.shape[0]] = self.produced_embedding
        self.header += self.produced_embedding.shape[0]
        self.embed2full[self.header: self.header +
                        self.produced_embedding.shape[0]] = batch.sub_to_full[:num_node][record_mask]
        self.produced_embedding = None
        # evict
        evict_mask = grad > thres and ~used_mask
        self.embed2full[self.sub2embed[evict_mask]] = -1

    # infer both node features and history embeddings
    def lookup_and_load(self, batch, uvm, num_layer):
        nodes = batch.sub_to_full[:batch.num_node_in_layer[1].item()]
        input_nodes = batch.sub_to_full
        self.sub2embed = self.full2embed[nodes]
        self.sub2feat = self.full2embed[self.total_num_node + input_nodes]
        self.used_masks = hiscache.count_history_reconstruct(
            batch.ptr,
            batch.idx,
            self.sub2embed,
            batch.sub_to_full.shape[0],  # num_node
            batch.y.shape[0],  # num_label
            num_layer,  # num_layer
        )
        used_mask = self.used_masks[0]
        # not pruned and not loaded
        load_mask = used_mask and (self.sub2feat == -1)
        uvm.cached_masked_get(batch.sub_to_full, load_mask)
