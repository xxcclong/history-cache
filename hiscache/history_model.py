import cxgnncomp
import hiscache_backend
import torch
import torch.nn.functional as F
import dgl.nn.pytorch.conv as dglnn
from .util import log


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
        self.init_convs(**kwargs)
        self.num_rel = kwargs["num_rel"]

    def forward(self, batch):
        blocks = batch[0]
        x = batch[1]
        for i, conv in enumerate(self.convs[:-1]):
            etypes = torch.randint(
                0,
                self.num_rel,
                (blocks[i].number_of_edges(), ),
                device=x.device,
            )
            if self.training:
                x, his = conv(
                    blocks[i],
                    x,
                    etypes,
                    self.table.sub_to_history_layered[i],
                    self.table.tmp_history_buffer_layered[i],
                )
                self.table.history_out.append(his)
            else:
                x, his = conv(
                    blocks[i],
                    x,
                    etypes,
                    torch.tensor([]),
                    torch.tensor([]),
                    0,
                )
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        etypes = torch.randint(
            0,
            self.num_rel,
            (blocks[-1].number_of_edges(), ),
            device=x.device,
        )
        if self.training and self.last_layer_history:
            x, his = self.convs[-1](
                blocks[-1],
                x,
                etypes,
                self.table.sub_to_history_layered[-1],
                self.table.tmp_history_buffer_layered[-1],
            )
            self.table.history_out.append(his)
        else:
            x, his = self.convs[-1](
                blocks[-1],
                x,
                etypes,
                torch.tensor([]),
                torch.tensor([]),
                0,
            )
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
            self.last_layer_history = self.table.num_layers == self.num_layers
            self.convs.append(
                self.init_history_conv(self.hidden_channels,
                                       self.out_channels,
                                       need_grad=(self.table.method == "grad"),
                                       **kwargs))

    def init_history_conv(self, in_channels, out_channels, **kwargs):
        assert "need_grad" in kwargs
        assert "num_rel" in kwargs
        return DGLRGCNHistory(in_channels, out_channels, **kwargs)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


model_dict = {
    "sage": HistorySAGE,
    "rgcn": HistoryRGCN,
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
            num_rel=config.train.model.num_rel)
    else:
        model = model_dict[config.train.model.type.lower()](
            in_channel, hidden_channel, out_channel, num_layers, dropout,
            graph_type, config, table)
    return model
