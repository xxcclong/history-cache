import torch
from os import path
import hiscache
from .util import log
import numpy as np
import math
import time


class HistoryCache:

    def __init__(self, uvm, config, mode, device_id=-1):
        self.in_channels = config.dl.dataset.feature_dim
        self.out_channels = config.dl.dataset.num_classes
        self.hidden_channels = config.train.model.hidden_dim
        self.num_layers_all = config.train.model.num_layers
        self.total_num_node = int(
            open(
                path.join(str(config["dl"]["dataset"]["path"]), "processed",
                          "num_nodes.txt")).readline())
        self.num_device = 1
        self.device_id = device_id
        if config["dl"]["num_device"] > 1:
            self.num_device = int(config["dl"]["num_device"])
            assert device_id >= 0
            self.device = torch.device(device_id)
        else:
            self.device = torch.device(config["dl"]["device"])
        # history embedding settings
        self.method = config.history['method']
        self.rate = float(config.history['rate'])
        assert self.method in ["random", "grad", "staleness"]
        self.staleness_thres = int(config.history.staleness_thres)
        self.limit = float(config.history.limit)
        log.warn("limiting cache size to {}GB".format(self.limit))

        self.full2embed = torch.ones([
            self.total_num_node * 2 + 1,
        ],
                                     dtype=torch.int64,
                                     device=self.device) * -1
        # self.header = torch.zeros([1], dtype=torch.int64, device=self.device)
        self.header = 0
        self.produced_embedding = None  # embedding that produced from training iterations
        self.feat_mode = mode
        self.uvm = uvm
        self.glb_iter = 0
        self.used_masks = None

        self.distributed_store = False
        if "history" in self.feat_mode:
            if self.num_device > 1:
                if config.history.distributed_store:
                    self.distributed_store = True
                    self.allocate_distributed_store()
            self.allocate()
            self.fill(config)

    def allocate_distributed_store(self):
        t0 = time.time()
        needed_byte = torch.numel(self.uvm.buffer) // self.num_device * (
            4 if self.uvm.buffer.dtype == torch.float32 else 2)
        max_byte = torch.cuda.mem_get_info()[0]
        assert needed_byte < max_byte

        # partition at feature dimension
        allocated_num = self.uvm.buffer.shape[1] // self.num_device
        self.distributed_buffer = self.uvm.buffer[:, self.device_id *
                                                  allocated_num:
                                                  (self.device_id) *
                                                  allocated_num].to(
                                                      self.device)
        # partition at node dimension
        # allocated_num = self.uvm.buffer.shape[0] // self.num_device
        # if self.device_id != self.num_device - 1:
        #     self.distributed_buffer = self.uvm.buffer[allocated_num * self.
        #                                              device_id:allocated_num *
        #                                              (self.device_id + 1)].to(
        #                                                  self.device)
        # else:
        #     self.distributed_buffer = self.uvm.buffer[allocated_num *
        #                                              self.device_id:].to(
        #                                                  self.device)
        t1 = time.time()
        log.info(
            f"Initializing distributed store for device {self.device_id} takes {t1 - t0} seconds"
        )

    def allocate(self):
        t0 = time.time()
        max_byte = torch.cuda.mem_get_info()[0]
        # reverve 4GB memory for other usage
        max_byte = max_byte - 4 * 1024 * 1024 * 1024
        lcm = self.hidden_channels * self.in_channels // math.gcd(
            self.hidden_channels, self.in_channels)
        # hidden stored in float32
        # max_buffer_byte = max_byte // (self.hidden_channels *
        #                                4 + 8) * (self.hidden_channels * 4)
        max_buffer_byte = max_byte
        if self.limit > 0 and max_buffer_byte > self.limit * 1024 * 1024 * 1024:
            max_buffer_byte = int(self.limit * 1024 * 1024 * 1024)
        max_buffer_byte = (max_buffer_byte + 4 * lcm - 1) // (4 * lcm) * (4 *
                                                                          lcm)
        self.buffer = torch.empty([max_buffer_byte // 4])  # 4 bytes per float
        self.embed2full = torch.ones(
            self.buffer.shape[0] // self.hidden_channels,
            dtype=torch.int64,
            device=self.device) * -1
        self.overall_feat_num = self.buffer.shape[0] // self.in_channels
        self.overall_embed_num = self.buffer.shape[0] // self.hidden_channels
        log.info(f"max buffer size: {self.buffer.shape}")
        self.max_buffer_byte = max_buffer_byte
        log.info(f"allocating buffer: {time.time() - t0}")

    def fill(self, config):
        t0 = time.time()
        # fill with raw features
        sorted_indices_datapath = path.join(
            str(config["dl"]["dataset"]["path"]), "processed",
            "sorted_indices.dat")
        indice = torch.from_numpy(
            np.fromfile(sorted_indices_datapath, dtype=np.int64))
        log.info("load indice")
        # assert overall_num_node >= self.overall_feat_num
        if self.overall_feat_num > self.total_num_node:
            cache_indice = indice  # cache everything
        else:
            cache_indice = indice[-self.overall_feat_num:]
        new_id = torch.arange(0, len(cache_indice)).to(self.device)
        if self.overall_feat_num > self.total_num_node:
            new_id += self.overall_feat_num - self.total_num_node
        self.feat2full = cache_indice.to(self.device) + self.total_num_node
        self.full2embed[self.feat2full] = new_id
        # self.embed2full
        buffer_path = path.join(str(config["dl"]["dataset"]["path"]),
                                "processed", "node_features.dat")
        dset_name = config.dl.dataset.name.lower()
        # if "mag240" in dset_name and (not "384" in dset_name and not "768" in dset_name):
        self.dtype = np.float32
        self.torch_dtype = torch.float32
        self.load_dtype = np.float32
        if dset_name.lower() in ["mag240m", "rmag240m"]:
            self.load_dtype = np.float16
        if "mmap" in self.feat_mode or dset_name in [
                "twitter", "friendster", "mag240m_384", "mag240m_768",
                "rmag240m_384", "rmag240m_768"
        ]:
            # tmp_buffer = torch.randn(
            #     [self.total_num_node, self.in_channels], dtype=self.torch_dtype)
            self.buffer.view(
                -1, self.in_channels)[-cache_indice.shape[0]:] = torch.randn(
                    [cache_indice.shape[0], self.in_channels])
        else:
            log.info("begin loading feature file")
            tmp_buffer = self.uvm.buffer[cache_indice].to(torch.float32)
            log.info("end loading feature file")
            assert tmp_buffer.shape[1] == self.in_channels
            self.buffer.view(
                -1,
                self.in_channels)[-cache_indice.shape[0]:] = tmp_buffer.view(
                    -1, self.in_channels)
            del tmp_buffer
        if self.buffer.shape[0] % self.hidden_channels != 0:
            self.buffer = torch.cat([
                self.buffer,
                torch.zeros(self.hidden_channels -
                            (self.buffer.shape[0] % self.hidden_channels),
                            dtype=self.torch_dtype)
            ])
        self.buffer = self.buffer.to(self.device)
        log.info(f"buffer size {self.buffer.shape}")
        log.info(f"filling buffer with raw features: {time.time() - t0}")

    @torch.no_grad()  # important for removing grad from computation graph
    def update_history(self, batch, glb_iter):
        if not "history" in self.feat_mode or self.staleness_thres == 0:
            return
        num_node = batch.num_node_in_layer[1].item()
        if (glb_iter % self.staleness_thres) == 0:
            self.header = 0
        if self.produced_embedding is None:
            self.produced_embedding = torch.randn(
                [num_node, self.hidden_channels], device=self.device)
        # used_mask = self.used_masks[2][:num_node]
        if self.produced_embedding.grad is not None:
            grad = self.produced_embedding.grad
            assert grad.shape[
                0] == num_node, "grad shape: {}, num_node: {}".format(
                    grad.shape, num_node)
            grad = grad.norm(dim=1)
        else:
            grad = torch.randn([num_node], device=self.device)
        thres = torch.quantile(grad, self.rate)
        record_mask = torch.logical_and(grad < thres, self.sub2embed == -1)
        self.produced_embedding.grad = None
        # record
        # print(self.produced_embedding.shape)
        embed_to_record = self.produced_embedding[record_mask]
        num_to_record = embed_to_record.shape[0]
        self.buffer.view(-1,
                         self.hidden_channels)[self.header:self.header +
                                               num_to_record] = embed_to_record

        begin = self.header * self.hidden_channels // self.in_channels
        end = math.ceil((self.header + num_to_record) * self.hidden_channels /
                        self.in_channels)
        self.full2embed[self.feat2full[begin:end]] = -1

        # invalidate the cache that are about to be overwritten
        tmp = self.embed2full[self.header:self.header + num_to_record]
        change_area = self.full2embed[tmp]
        change_area[torch.logical_and(
            change_area >= self.header,
            change_area < self.header + num_to_record)] = -1
        self.full2embed[tmp] = change_area
        # hkz comment:
        #   below is not right: there are some nodes that are updated by other iteration of embeddings
        #   so embed2full is not needed
        # self.full2embed[self.embed2full[self.header: self.header +
        #                 self.produced_embedding.shape[0]]] = -1

        # update the mapping from full id to embed id
        change_id = batch.sub_to_full[:num_node][record_mask]
        # log.info(f"used_mask: {torch.sum(used_mask)} record_mask: {torch.sum(record_mask)} change_id: {change_id.shape[0]} num_node: {num_node}")
        # assert(torch.sum(self.full2embed[change_id] != -1) == 0)
        new_id = torch.arange(self.header,
                              self.header + num_to_record,
                              device=self.device)
        self.full2embed[change_id] = new_id
        self.embed2full[self.header:self.header +
                        num_to_record] = batch.sub_to_full[:num_node][
                            record_mask]
        self.header += num_to_record
        self.produced_embedding = None
        # # evict
        # evict_mask = torch.logical_and(grad > thres, self.sub2embed != -1)
        # # self.embed2full[self.sub2embed[evict_mask]] = -1
        # self.full2embed[batch.sub_to_full[:num_node]
        #                 [evict_mask]] = -1  # TODO: may be wrong

    # infer both node features and history embeddings

    def lookup_and_load(self, batch, num_layer, load_all=False):
        if load_all:
            self.sub2feat = self.full2embed[batch.sub_to_full +
                                            self.total_num_node]
            load_mask = self.sub2feat == -1
            if self.distributed_store:  # load from other GPUs
                assert False
            else:  # load from UVM
                batch.x = self.uvm.masked_get(batch.sub_to_full,
                                              load_mask)  # load from UVM
            cached_feat = self.buffer.view(
                -1, self.in_channels)[self.sub2feat]  # load from feat cache
            cached_feat[load_mask] = 0
            batch.x += cached_feat  # combine feat from UVM and feat cache
            return None

        if not "history" in self.feat_mode:
            self.sub2embed = torch.ones([batch.num_node_in_layer[1].item()],
                                        dtype=torch.int64,
                                        device=self.device) * -1
            self.cached_embedding = torch.tensor([])
            if self.distributed_store:
                return torch.ones(batch.sub_to_full.shape[0],
                                  device=self.device,
                                  dtype=torch.bool)
            else:
                if self.feat_mode in ["uvm", "mmap"]:
                    batch.x = self.uvm.get(batch.sub_to_full)
                else:
                    assert False
                return None

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
        self.ana = 0
        if self.ana:
            log.info(
                f"glb-iter: {self.glb_iter} prune-by-his: {torch.sum(~self.used_masks[0])} prune-by-feat: {torch.sum(hit_feat_mask)} prune-by-both: {torch.sum(~load_mask)} overall: {batch.sub_to_full.shape[0]} hit-rate {torch.sum(~load_mask) / batch.sub_to_full.shape[0]}"
            )
            embed_num = torch.sum(self.full2embed[:self.total_num_node] != -1)
            feat_num = torch.sum(self.full2embed[self.total_num_node:] != -1)
            overall_size = feat_num * self.in_channels + embed_num * self.hidden_channels
            log.info(
                f"embed-num: {embed_num} feat-num: {feat_num} overall: {overall_size} buffersize: {self.buffer.shape[0]}"
            )
        self.glb_iter += 1
        if not self.distributed_store:
            # 2. load raw features
            x = self.uvm.masked_get(batch.sub_to_full, load_mask)
            # 3. load hit raw feature cache
            # x[hit_feat_mask] = self.buffer.view(-1, self.in_channels)[
            #     self.sub2feat[hit_feat_mask]]

            cached_feat = self.buffer.view(-1, self.in_channels)[self.sub2feat]
            cached_feat[~hit_feat_mask] = 0
            x += cached_feat
            batch.x = x
            return None
        else:
            cached_feat = self.buffer.view(-1, self.in_channels)[self.sub2feat]
            cached_feat[~hit_feat_mask] = 0
            batch.x = cached_feat
            return load_mask  # return the mask of feature that needs communication
