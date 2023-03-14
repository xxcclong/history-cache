import cxgnncomp
import cxgnncomp_backend
import hiscache_backend
import torch
import torch.nn.functional as F
import dgl.nn.pytorch.conv as dglnn
from .util import log
from torch_geometric.nn.inits import glorot        
import time


class DGLRGCNHistory(torch.nn.Module):

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_rel,
        need_grad,
    ):
        super(DGLRGCNHistory, self).__init__()
        self.conv = dglnn.RelGraphConv(
            in_channels,
            hidden_channels,
            num_rel,
        )
        self.need_grad = need_grad

    def forward(self,
                graph,
                x,
                etypes,
                history_map,
                history_buffer,
                history_size=1):
        output = self.conv(graph, x, etypes)
        his = output
        if self.need_grad:
            his.retain_grad()
        if history_size == 0:
            return output, None
        assert output.shape[0] == history_map.shape[0]
        valid_pos = history_map != -1
        valid_buffer_pos = history_map[valid_pos]
        if valid_buffer_pos.shape[0] == 0:
            return output, his
        output[valid_pos] = history_buffer[valid_buffer_pos]
        return output, his

    def reset_parameters(self):
        self.conv.reset_parameters()


class MySageConvHistory(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 need_grad,
                 root_weight: bool = True,
                 bias: bool = True):
        super(MySageConvHistory, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.root_weight = root_weight
        self.lin_l = torch.nn.Linear(in_channels, hidden_channels, bias=bias)
        if self.root_weight:
            self.lin_r = torch.nn.Linear(in_channels,
                                         hidden_channels,
                                         bias=False)
        self.reset_parameters()
        self.need_grad = need_grad

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(self, x, ptr, idx, history_map, history_buffer, history_size,
                num_node):
        if history_size > 0:
            # log.info(
            #     f"{torch.max(history_map)} {torch.min(history_map)} {history_map.shape} {history_buffer.shape} {history_size} {num_node} {x.shape}")
            # torch.cuda.synchronize()
            out = hiscache_backend.aggr_forward_history(
                x, ptr, idx, history_map, history_buffer, history_size,
                num_node)
            # torch.cuda.synchronize()
        else:
            out = cxgnncomp.sage_mean_forward(x, ptr, idx, num_node)
        his = out
        if self.need_grad and self.training and history_size > 0:
            his.retain_grad()
        out = self.lin_l(out)
        if self.root_weight:
            if num_node != 0:
                out += self.lin_r(x[:num_node])
            else:
                out += self.lin_r(x)
        return out, his


class HistorySAGE(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, graph_type, config, table, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.graph_type = graph_type
        self.bns = torch.nn.ModuleList()
        self.num_layers = num_layers
        for _ in range(self.num_layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.dropout = dropout
        self.table = table
        self.init_convs(**kwargs)

    def forward(self, batch):
        x = batch.x
        for i, conv in enumerate(self.convs[:-1]):
            if self.training and i == self.num_layers - 2:
                x, his = conv(x, batch.ptr, batch.idx,
                              self.table.sub2embed,
                              self.table.cached_embedding,
                              self.hidden_channels,
                              batch.num_node_in_layer[self.num_layers - 1 - i])
                # x, his = conv(x, batch.ptr, batch.idx,
                #               self.table.sub_to_history_layered[i],
                #               self.table.tmp_history_buffer_layered[i],
                #               self.table.history_size_layered[i],
                #               batch.num_node_in_layer[self.num_layers - 1 - i])
                self.table.produced_embedding = his
            else:
                x, his = conv(x, batch.ptr, batch.idx, torch.tensor([]),
                              torch.tensor([]), 0,
                              batch.num_node_in_layer[self.num_layers - 1 - i])
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.training and self.last_layer_history:
            assert False
            x, his = self.convs[-1](x, batch.ptr, batch.idx,
                                    self.table.sub_to_history_layered[-1],
                                    self.table.tmp_history_buffer_layered[-1],
                                    batch.num_node_in_layer[0])
            self.table.history_out.append(his)
        else:
            x, his = self.convs[-1](x, batch.ptr, batch.idx, torch.tensor([]),
                                    torch.tensor([]), 0,
                                    batch.num_node_in_layer[0])
        return x.log_softmax(dim=-1)

    def init_convs(self, **kwargs):
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            self.init_history_conv(self.in_channels,
                                   self.hidden_channels,
                                   need_grad=(self.table.method == "grad"),
                                   **kwargs))
        for _ in range(self.num_layers - 2):
            self.convs.append(
                self.init_history_conv(self.hidden_channels,
                                       self.hidden_channels,
                                       need_grad=(self.table.method == "grad"),
                                       **kwargs))
        if self.num_layers > 1:
            self.last_layer_history = False  # self.table.num_layers == self.num_layers
            self.convs.append(
                self.init_history_conv(self.hidden_channels,
                                       self.out_channels,
                                       need_grad=(self.table.method == "grad"),
                                       **kwargs))

    def init_history_conv(self, in_channels, out_channels, **kwargs):
        assert "need_grad" in kwargs
        return MySageConvHistory(in_channels, out_channels, **kwargs)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

class MyRGCNConvHistory(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_rel,
                 need_grad,
                 ):
        super(MyRGCNConvHistory, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_rel = num_rel
        self.linear = torch.nn.Parameter(
            torch.randn(num_rel, in_channels, hidden_channels))
        log.info("linear shape: {}".format(self.linear.shape))
        self.register_parameter("rel_weight", self.linear)
        self.reset_parameters()
        self.need_grad = need_grad

    def reset_parameters(self):
        glorot(self.linear)

    def forward(self, x, ptr, idx, edge_types, count, history_map, history_buffer, used_mask, history_size,
                num_node):
        out = cxgnncomp.RGCNOP.apply(x, self.linear, ptr, idx, edge_types, num_node)
        deg = ptr[1:] - ptr[:-1]
        out = out / deg.unsqueeze(-1)[:out.shape[0]]
        if history_size > 0:
            out[history_map != -1] = history_buffer[history_map != -1]
        his = out
        if self.need_grad and self.training and history_size > 0:
            his.retain_grad()
        return out, his

class MyRGCNConvHistory2(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_rel,
                 need_grad,
                 ):
        super(MyRGCNConvHistory2, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_rel = num_rel
        self.linear = torch.nn.Parameter(
            torch.randn(num_rel, in_channels, hidden_channels))
        log.info("linear shape: {}".format(self.linear.shape))
        self.register_parameter("rel_weight", self.linear)
        self.reset_parameters()
        self.need_grad = need_grad

    def reset_parameters(self):
        glorot(self.linear)

    def forward(self, x, ptr, idx, edge_types, count, history_map, history_buffer, used_mask, history_size,
                num_node):
        num_edge = ptr[num_node]
        deg = ptr[1:num_node + 1] - ptr[:num_node]
        dst = torch.arange(num_node, device=x.device).repeat_interleave(deg)
        src = idx[:num_edge]
        rel = edge_types[:num_edge]
        if used_mask is not None:
            rel[~used_mask[src]] = self.num_rel + 0 # exclude the computation of unused ones
        sorted, perm = torch.sort(rel)
        src = src[perm]
        dst = dst[perm]
        # count = []
        # for i in range(self.num_rel + 1):
        #     count.append((sorted == i).sum().item())
        # print("count num", torch.max(rel), torch.min(rel))
        torch.cuda.synchronize()
        t0 = time.time()
        count = hiscache_backend.count_num(rel, self.num_rel).cpu()
        t1 = time.time()
        print("count num time", t1 - t0, num_edge, num_node, count)
        # print(count)
        # count = sorted.bincount(minlength=self.num_rel).cpu()
        # torch.cuda.synchronize()
        out = cxgnncomp.RGCNOP_sorted(x, self.linear, src, dst, count, num_node)

        # out = cxgnncomp.aggr_rel_direct(x, ptr, idx, self.linear, edge_types.to(torch.int32), num_node,
        #                            self.num_rel)


        out = out / deg.unsqueeze(-1)
        if history_size > 0:
            out[history_map != -1] = history_buffer[history_map != -1]
        his = out
        if self.need_grad and self.training and history_size > 0:
            his.retain_grad()
        return out, his

# class MyRGCNConvHistory(torch.nn.Module):

#     def __init__(self,
#                  in_channels,
#                  hidden_channels,
#                  num_rel,
#                  need_grad,
#                  root_weight: bool = True,
#                  bias: bool = True):
#         super(MyRGCNConvHistory, self).__init__()
#         self.in_channels = in_channels
#         self.hidden_channels = hidden_channels
#         self.root_weight = root_weight
#         self.lin_l = torch.nn.Linear(in_channels, hidden_channels, bias=bias)
#         if self.root_weight:
#             self.lin_r = torch.nn.Linear(in_channels,
#                                          hidden_channels,
#                                          bias=False)
#         self.reset_parameters()
#         self.need_grad = need_grad

#     def reset_parameters(self):
#         self.lin_l.reset_parameters()
#         if self.root_weight:
#             self.lin_r.reset_parameters()

#     def forward(self, x, ptr, idx, etype, history_map, history_buffer, history_size,
#                 num_node):
#         if history_size > 0:
#             # log.info(
#             #     f"{torch.max(history_map)} {torch.min(history_map)} {history_map.shape} {history_buffer.shape} {history_size} {num_node} {x.shape}")
#             # torch.cuda.synchronize()
#             out = hiscache_backend.aggr_forward_history(
#                 x, ptr, idx, history_map, history_buffer, history_size,
#                 num_node)
#             # torch.cuda.synchronize()
#         else:
#             out = cxgnncomp.sage_mean_forward(x, ptr, idx, num_node)
#         his = out
#         if self.need_grad and self.training and history_size > 0:
#             his.retain_grad()
#         out = self.lin_l(out)
#         if self.root_weight:
#             if num_node != 0:
#                 out += self.lin_r(x[:num_node])
#             else:
#                 out += self.lin_r(x)
#         return out, his

class HistoryRGCN(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, graph_type, config, table, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.graph_type = graph_type
        self.bns = torch.nn.ModuleList()
        self.num_layers = num_layers
        for _ in range(self.num_layers - 1):
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.dropout = dropout
        self.table = table
        self.num_rel = kwargs["num_rel"]
        self.dataset_name = kwargs["dataset_name"]
        self.gen_rel = self.dataset_name == "rmag240m"
        if self.gen_rel:
            self.num_rel = 5
        self.init_convs(num_rel=self.num_rel)

    def forward(self, batch):
        x = batch.x
        if self.gen_rel:
            etypes = batch.edge_type
            # assert etypes is not None
            etypes = cxgnncomp_backend.gen_edge_type_mag240m(
                batch.ptr, batch.idx, batch.sub_to_full)
            # print("ptr", batch.ptr, batch.ptr.shape, torch.max(batch.ptr), torch.min(batch.ptr))
            # print("idx", batch.idx, batch.idx.shape, torch.max(batch.idx), torch.min(batch.idx))
            # print("sub_to_full", batch.sub_to_full, batch.sub_to_full.shape, torch.max(batch.sub_to_full), torch.min(batch.sub_to_full))
            # print("generated etypes,", etypes.shape, etypes, torch.max(etypes), torch.min(etypes))
        else:
            etypes = torch.randint(
                0,
                self.num_rel, (batch.num_edge_in_layer[self.num_layers - 1], ),
                device=x.device)
        for i, conv in enumerate(self.convs[:-1]):
            if self.training and i == self.num_layers - 2:
                x, his = conv(x, batch.ptr, batch.idx, etypes[:batch.num_edge_in_layer[self.num_layers - 1 - i]],
                              batch.typed_num_node_in_layer[i * self.num_rel : (i + 1) * self.num_rel] if batch.typed_num_node_in_layer is not None else None,
                              self.table.sub2embed,
                              self.table.cached_embedding,
                              self.table.used_masks[i] if self.table.used_masks is not None else None,
                              self.hidden_channels,
                              batch.num_node_in_layer[self.num_layers - 1 - i])
                self.table.produced_embedding = his
                self.table.produced_embedding.retain_grad()
            else:
                x, his = conv(x, batch.ptr, batch.idx, etypes[:batch.num_edge_in_layer[self.num_layers - 1 - i]], 
                              batch.typed_num_node_in_layer[i * self.num_rel : (i + 1) * self.num_rel] if batch.typed_num_node_in_layer is not None else None,
                              torch.tensor([]),
                              torch.tensor([]), 
                              self.table.used_masks[i] if self.table.used_masks is not None else None,
                              0,
                              batch.num_node_in_layer[self.num_layers - 1 - i])
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.training and self.last_layer_history:
            assert False
        else:
            x, his = self.convs[-1](x, batch.ptr, batch.idx, etypes[:batch.num_edge_in_layer[0]], 
                                    batch.typed_num_node_in_layer[i * self.num_rel : (i + 1) * self.num_rel] if batch.typed_num_node_in_layer is not None else None,
                                    torch.tensor([]),
                                    torch.tensor([]), 
                                    self.table.used_masks[-2] if self.table.used_masks is not None else None,
                                    0,
                                    batch.num_node_in_layer[0])
        return x.log_softmax(dim=-1)

    def init_convs(self, **kwargs):
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            self.init_history_conv(self.in_channels,
                                   self.hidden_channels,
                                   need_grad=(self.table.method == "grad"),
                                   **kwargs))
        for _ in range(self.num_layers - 2):
            self.convs.append(
                self.init_history_conv(self.hidden_channels,
                                       self.hidden_channels,
                                       need_grad=(self.table.method == "grad"),
                                       **kwargs))
        if self.num_layers > 1:
            self.last_layer_history = False
            self.convs.append(
                self.init_history_conv(self.hidden_channels,
                                       self.out_channels,
                                       need_grad=(self.table.method == "grad"),
                                       **kwargs))

    def init_history_conv(self, in_channels, out_channels, **kwargs):
        assert "need_grad" in kwargs
        assert "num_rel" in kwargs
        return MyRGCNConvHistory2(in_channels, out_channels, **kwargs)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


model_dict = {
    "sage": HistorySAGE,
    "rgcn": HistoryRGCN,
    # "rgcn": cxgnncomp.RGCN
}

graph_type_dict = {
    "cxg": "CSR_Layer",
    "dgl": "DGL",
}


def get_model(config, table):
    in_channel = config.dl.dataset.feature_dim
    out_channel = config.dl.dataset.num_classes
    hidden_channel = config.train.model.hidden_dim
    num_layers = config.train.model.num_layers
    dropout = config.train.model.dropout
    graph_type = graph_type_dict[config.train.type.lower()]
    if config.train.model.type.lower() == "rgcn":
        model = model_dict[config.train.model.type.lower()](
            in_channel,
            hidden_channel,
            out_channel,
            num_layers,
            dropout,
            graph_type,
            config,
            table,
            num_rel=config.train.model.num_rel,
            dataset_name=config.dl.dataset.name.lower(),
            )
    else:
        model = model_dict[config.train.model.type.lower()](
            in_channel, hidden_channel, out_channel, num_layers, dropout,
            graph_type, config, table)
    return model
