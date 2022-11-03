import torch
from os import path
import hiscache_backend


class HistoryTable:

    def __init__(self, config):
        self.in_channels = config.dl.dataset.feature_dim
        self.out_channels = config.dl.dataset.num_classes
        self.hidden_channels = config.train.model.hidden_dim
        self.num_layers_all = config.train.model.num_layers
        self.total_num_node = int(
            open(
                path.join(str(config["dl"]["dataset"]["path"]), "processed",
                          "num_nodes.txt")).readline())
        self.device = torch.device(config["dl"]["device"])
        self.store_last_layer = config.history.get('store_last_layer', False)
        self.num_layers = self.num_layers_all - 1 + int(
            bool(self.store_last_layer))  # NOTE: layers of table
        self.history_size_layered = self.get_hist_size(config)
        print("history_size_layered", self.history_size_layered)
        if isinstance(config.history.buffer_size, int):
            self.history_num = [config.history.buffer_size] * self.num_layers
        else:
            assert len(config.history.buffer_size) == self.num_layers
            self.history_num = config.history.buffer_size
        self.history_buffer_layered = [
            torch.empty([self.history_num[i], self.history_size_layered[i]],
                        device=self.device) for i in range(self.num_layers)
        ]
        # mapping
        self.full_to_history_layered = [
            torch.ones(self.total_num_node, device=self.device).long().neg()
            for _ in range(self.num_layers)
        ]
        self.history_to_full_layered = [
            torch.ones(self.history_num[i], device=self.device).neg().long()
            for i in range(self.num_layers)
        ]
        self.checkin_iter = [
            torch.ones(self.history_num[i], device=self.device).neg().long()
            for i in range(self.num_layers)
        ]
        self.cuda_header_layered = [
            torch.zeros([1], device=self.device).long()
            for _ in range(self.num_layers)
        ]
        self.method = config.history['method']
        self.rate = float(config.history['rate'])
        assert self.method in ["random", "grad", "staleness"]
        if self.method != "staleness":
            assert self.rate > 0
        else:
            self.rate = 1
        self.staleness_thres = config.history.staleness_thres
        self.num_used = 0
        self.new_histories = []
        self.history_out = []
        self.graph_struct_norm = int(config.history.graph_struct_norm)

    def get_hist_size(self, config):
        if "SAGE" in config.train.model.type or "GCN" == config.train.model.type:  # record history after aggregation for SAGE now
            return [self.in_channels
                    ] + [self.hidden_channels] * (self.num_layers - 1)
        elif "gat" in config.train.model.type.lower(
        ) or "rgcn" == config.train.model.type.lower(
        ):  # record history after lin for GAT
            if not self.store_last_layer:
                return [self.hidden_channels] * self.num_layers
            else:
                return [self.hidden_channels] * (self.num_layers - 1) + [
                    self.out_channels
                ]

    def lookup(self, sub_to_full, num_node_in_layer):
        self.tmp_history_buffer_layered = []
        self.sub_to_history_layered = []
        self.sub_to_history = []
        num_seed = num_node_in_layer[0]
        num_all_node = num_node_in_layer[-1]
        for layer_id in range(self.num_layers):
            # [-1, x, -1, y, -1, z] history_id
            num_node = num_node_in_layer[self.num_layers_all - 1 - layer_id]
            sub_to_history = self.full_to_history_layered[layer_id][
                sub_to_full[:num_node]]
            # all seed nodes do original training
            sub_to_history[:num_seed] = -1
            sub_to_history.requires_grad = False
            mask = sub_to_history != -1
            self.sub_to_history.append(sub_to_history)
            selected_sub_to_history = torch.masked_select(
                sub_to_history, mask)  # [x,y,z]
            tmp_history_buffer = self.history_buffer_layered[layer_id][
                selected_sub_to_history]  # .view(-1)
            new_id = torch.arange(0,
                                  len(selected_sub_to_history),
                                  device=sub_to_history.device)  # [0,1,2]
            index = mask.nonzero().view(-1)
            output_sub_to_history = torch.ones(
                num_node, device=sub_to_history.device).long().neg()
            sub_to_history_ = output_sub_to_history.index_put_(
                [index], new_id)  # [-1, 0, -1, 1, -1, 2]
            if self.method == "grad":
                tmp_history_buffer.requires_grad = True
            self.tmp_history_buffer_layered.append(tmp_history_buffer)
            self.sub_to_history_layered.append(sub_to_history_)

    @torch.no_grad()
    def record_history(self, sub_to_full, batch, glb_iter, used_masks):
        for layer_id in range(self.num_layers):
            if self.history_buffer_layered[layer_id].shape[0] == 0:
                continue
            num_node = batch.num_node_in_layer[self.num_layers_all - 1 -
                                               layer_id].item()
            his = self.history_out[layer_id][:num_node]
            grad = (self.history_out[layer_id].grad)[:num_node]
            if self.method == "grad":
                # record_sum = torch.einsum("ij,ij->i", grad, grad)
                record_sum = torch.sum(torch.abs(grad), 1)
                if self.graph_struct_norm:
                    graph_structure_score = hiscache_backend.get_graph_structure_score(
                        batch.ptr, batch.idx, batch.x.shape[0],
                        batch.y.shape[0], self.num_layers)
                    record_sum = torch.mul(
                        record_sum,
                        graph_structure_score[:record_sum.shape[0]])
            elif self.method == "random":
                record_sum = torch.randn([grad.shape[0]]).to(grad.device)
            else:
                assert False, "Error record type"
            thres = torch.quantile(record_sum, self.rate, dim=0, keepdim=False)
            hiscache_backend.record_history(
                his,
                self.history_buffer_layered[layer_id],
                # used mask: raw_feature, embedding_layer0, embeding_layer1 ... seed_node_embedding
                used_masks[layer_id + 1],
                sub_to_full,
                self.sub_to_history[layer_id],
                self.history_to_full_layered[layer_id],
                self.full_to_history_layered[layer_id],
                self.cuda_header_layered[layer_id],
                record_sum,
                thres,
                self.checkin_iter[layer_id],
                glb_iter)

    def evict_by_staleness(self, glb_iter, staleness_thres):
        if isinstance(staleness_thres, int):
            staleness_thres = [staleness_thres] * self.num_layers
        for layer_id in range(self.num_layers):
            stale_iter = glb_iter - staleness_thres[layer_id]
            checkin = self.checkin_iter[layer_id]
            hist_id = torch.logical_and(-1 < checkin,
                                        checkin <= stale_iter).nonzero()
            full_id = self.history_to_full_layered[layer_id][hist_id]
            self.checkin_iter[layer_id][hist_id] = -1
            self.history_to_full_layered[layer_id][hist_id] = -1
            self.full_to_history_layered[layer_id][full_id] = -1

    def evict_by_both(self, batch, glb_iter, staleness_thres):
        if self.method == "staleness":
            self.evict_by_staleness(glb_iter, staleness_thres)
            return

        if isinstance(staleness_thres, int):
            staleness_thres = [staleness_thres] * self.num_layers

        for layer_id in range(self.num_layers):
            if self.history_buffer_layered[layer_id].shape[0] == 0:
                continue
            # by gradient first
            num_node = batch.num_node_in_layer[self.num_layers_all - 1 -
                                               layer_id].item()
            if self.method == "grad":
                grad = (self.history_out[layer_id].grad)[:num_node]
                # evict_sum = torch.einsum("ij,ij->i", grad, grad)
                evict_sum = torch.sum(torch.abs(grad), 1)
                if self.graph_struct_norm:
                    graph_structure_score = hiscache_backend.get_graph_structure_score(
                        batch.ptr, batch.idx, batch.x.shape[0],
                        batch.y.shape[0], self.num_layers)
                    evict_sum = torch.mul(
                        evict_sum, graph_structure_score[:evict_sum.shape[0]])
            elif self.method == "random":
                evict_sum = torch.randn([num_node], device=self.device)
            else:
                assert False, "Error record type"
            thres = torch.quantile(evict_sum,
                                   1 - self.rate,
                                   dim=0,
                                   keepdim=False)
            to_evict_id = torch.masked_select(self.sub_to_history[layer_id],
                                              evict_sum >= thres)
            to_evict_id = to_evict_id[to_evict_id != -1]
            # invalidate the history by gradient
            self.full_to_history_layered[layer_id][
                self.history_to_full_layered[layer_id][to_evict_id]] = -1
            self.history_to_full_layered[layer_id][to_evict_id] = -1
            self.checkin_iter[layer_id][to_evict_id] = -1

            # then by staleness
            stale_iter = glb_iter - staleness_thres[layer_id]
            checkin = self.checkin_iter[layer_id]
            hist_id = torch.logical_and(-1 < checkin,
                                        checkin <= stale_iter).nonzero()
            full_id = self.history_to_full_layered[layer_id][hist_id]
            self.checkin_iter[layer_id][hist_id] = -1
            self.history_to_full_layered[layer_id][hist_id] = -1
            self.full_to_history_layered[layer_id][full_id] = -1

    def evict_history(self, batch, glb_iter):
        self.evict_by_both(batch, glb_iter, self.staleness_thres)