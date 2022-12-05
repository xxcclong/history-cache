import torch
from os import path
import hiscache
from .util import log
import numpy as np
import math
import time


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
        assert self.staleness_thres > 0

        # self.embed2full = torch.empty([self.total_num_node,])
        self.full2embed = torch.ones(
            [self.total_num_node * 2, ], dtype=torch.int64, device=self.device) * -1
        # self.header = torch.zeros([1], dtype=torch.int64, device=self.device)
        self.header = 0
        self.produced_embedding = None  # embedding that produced from training iterations
        self.allocate()
        self.fill(config)
        self.uvm = uvm

    def allocate(self):
        t0 = time.time()
        max_byte = torch.cuda.mem_get_info()[0]
        # reverve 4GB memory for other usage
        max_byte = max_byte - 4 * 1024 * 1024 * 1024
        lcm = self.hidden_channels * self.in_channels // math.gcd(
            self.hidden_channels, self.in_channels)
        max_buffer_byte = max_byte // (self.hidden_channels *
                                       4 + 8) * (self.hidden_channels * 4)
        # if max_buffer_byte > self.in_channels * self.total_num_node * 4:
        #     max_buffer_byte = self.in_channels * self.total_num_node * 4
        #     log.warn("caching all the features")
        max_buffer_byte = (max_buffer_byte + 4*lcm - 1) // (4*lcm) * (4 * lcm)
        self.buffer = torch.empty([max_buffer_byte // 4])  # 4 bytes per float
        self.embed2full = torch.ones(
            self.buffer.shape[0] // self.hidden_channels, dtype=torch.int64, device=self.device) * -1
        self.overall_feat_num = self.buffer.shape[0] // self.in_channels
        self.overall_embed_num = self.buffer.shape[0] // self.hidden_channels
        log.info(f"max buffer size: {self.buffer.shape}")
        self.max_buffer_byte = max_buffer_byte
        log.info(f"allocating buffer: {time.time() - t0}")

    def fill(self, config):
        t0 = time.time()
        # fill with raw features
        ptr_datapath = path.join(str(config["dl"]["dataset"]["path"]), "processed",
                                 "csr_ptr_undirected.dat")
        ptr = torch.from_numpy(np.fromfile(ptr_datapath,
                                           dtype=np.int64))
        deg = ptr[1:] - ptr[:-1]  # in degree
        del ptr
        sorted, indice = torch.sort(deg)
        del sorted
        overall_num_node = deg.shape[0]
        del deg
        # assert overall_num_node >= self.overall_feat_num
        cache_indice = indice[-self.overall_feat_num:]
        new_id = torch.arange(0, len(cache_indice)).to(self.device)
        self.full2embed[self.total_num_node +
                        cache_indice.to(self.device)] = new_id
        #
        buffer_path = path.join(str(config["dl"]["dataset"]["path"]), "processed",
                                "node_features.dat")
        if "mag240" in buffer_path:
            self.dtype = np.float16
            self.torch_dtype = torch.float16
        else:
            self.dtype = np.float32
            self.torch_dtype = torch.float32
        if "twitter" in buffer_path or "friendster" in buffer_path:
            tmp_buffer = torch.randn(
                [self.total_num_node, self.in_channels], dtype=self.torch_dtype)
        else:
            tmp_buffer = torch.from_numpy(
                np.fromfile(buffer_path, dtype=self.dtype)).view(self.total_num_node, self.in_channels)
        self.buffer.view(-1, self.in_channels)[-cache_indice.shape[0]:] = tmp_buffer[cache_indice].view(-1, self.in_channels)
        del tmp_buffer
        if self.buffer.shape[0] % self.hidden_channels != 0:
            self.buffer = torch.cat(
                [self.buffer, torch.zeros(self.hidden_channels - (self.buffer.shape[0] % self.hidden_channels), dtype=self.torch_dtype)])
        self.buffer = self.buffer.to(self.device)
        log.info(f"filling buffer with raw features: {time.time() - t0}")

    @torch.no_grad()
    def update_history(self, batch, glb_iter):
        num_node = batch.num_node_in_layer[1].item()
        # print(self.staleness_thres)
        if (glb_iter % self.staleness_thres) == 0:
            self.header = 0
            # self.header.zero_()
        grad = self.produced_embedding.grad
        used_mask = self.used_masks[2][:num_node]
        assert grad.shape[0] == num_node, "grad shape: {}, num_node: {}".format(
            grad.shape, num_node)
        grad = grad.norm(dim=1)
        thres = torch.quantile(grad, self.rate)
        record_mask = torch.logical_and(grad < thres, used_mask)
        # record
        # print(self.produced_embedding.shape)
        self.produced_embedding = self.produced_embedding[record_mask]
        # print(self.produced_embedding.shape[0],
        #   torch.sum(record_mask), num_node)
        # print(self.header, self.buffer.view(-1, self.hidden_channels).shape)
        self.buffer.view(-1, self.hidden_channels)[
            self.header: self.header+self.produced_embedding.shape[0]] = self.produced_embedding
        # self.full2embed[self.embed2full[self.header: self.header +
        #                 self.produced_embedding.shape[0]]] = -1
        self.full2embed[torch.logical_and(
            self.full2embed >= self.header, self.full2embed < self.header + self.produced_embedding.shape[0])] = -1
        self.full2embed[batch.sub_to_full[:num_node][record_mask]] = torch.arange(
            self.header, self.header+self.produced_embedding.shape[0], device=self.device)
        self.embed2full[self.header: self.header +
                        self.produced_embedding.shape[0]] = batch.sub_to_full[:num_node][record_mask]
        self.header += self.produced_embedding.shape[0]
        self.produced_embedding = None
        # evict
        evict_mask = torch.logical_and(grad > thres, ~used_mask)
        self.embed2full[self.sub2embed[evict_mask]] = -1
        self.full2embed[batch.sub_to_full[:num_node][evict_mask]] = -1

    # infer both node features and history embeddings

    def lookup_and_load(self, batch, num_layer):
        nodes = batch.sub_to_full[:batch.num_node_in_layer[1].item()]
        input_nodes = batch.sub_to_full
        self.sub2embed = self.full2embed[nodes]
        # 1. move cached embedding
        self.cached_embedding = self.buffer.view(
            -1, self.hidden_channels)[self.sub2embed]
        self.sub2feat = self.full2embed[input_nodes + self.total_num_node]
        # torch.cuda.synchronize()
        self.used_masks = hiscache.count_history_reconstruct(
            batch.ptr,
            batch.idx,
            self.sub2embed,
            batch.sub_to_full.shape[0],  # num_node
            batch.y.shape[0],  # num_label
            num_layer,  # num_layer
        )
        # torch.cuda.synchronize()
        used_mask = self.used_masks[0]
        hit_feat_mask = self.sub2feat != -1
        # not pruned and not loaded
        load_mask = torch.logical_and(used_mask, (~hit_feat_mask))
        self.ana = 1
        if self.ana:
            print(
                f"prune-by-his: {torch.sum(~self.used_masks[0])} prune-by-feat: {torch.sum(hit_feat_mask)} prune-by-both: {torch.sum(~load_mask)} overall: {batch.sub_to_full.shape[0]}")
        # 2. load raw features
        x = self.uvm.masked_get(batch.sub_to_full, load_mask)
        # 3. load hit raw feature cache
        # print(x.device, self.buffer.device,
        #       self.sub2feat.device, hit_feat_mask.device)
        # print(torch.max(self.sub2feat))
        # print(self.buffer.view(-1, self.in_channels).shape)
        # print(x.shape)
        # print(hit_feat_mask.shape)
        # torch.cuda.synchronize()
        x[hit_feat_mask] = self.buffer.view(-1, self.in_channels)[
            self.sub2feat[hit_feat_mask]]
        batch.x = x
