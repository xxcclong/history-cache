import cxgnncomp
import hiscache_backend
import torch
import torch.nn.functional as F


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
            out = hiscache_backend.aggr_forward_history(
                x, ptr, idx, history_map, history_buffer, history_size,
                num_node)
        else:
            out = cxgnncomp.sage_mean_forward(x, ptr, idx, num_node)
        his = out
        if self.need_grad and self.training:
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
            if self.training:
                x, his = conv(x, batch.ptr, batch.idx,
                              self.table.sub_to_history_layered[i],
                              self.table.tmp_history_buffer_layered[i],
                              self.table.history_size_layered[i],
                              batch.num_node_in_layer[self.num_layers - 1 - i])
                self.table.history_out.append(his)
            else:
                x, his = conv(x, batch.ptr, batch.idx, torch.tensor([]),
                              torch.tensor([]), 0,
                              batch.num_node_in_layer[self.num_layers - 1 - i])
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.training and self.last_layer_history:
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
            self.last_layer_history = self.table.num_layers == self.num_layers
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


model_dict = {
    "sage": HistorySAGE,
}

graph_type_dict = {
    "cxg": "CSR_Layer",
}


def get_model(config, table):
    in_channel = config.dl.dataset.feature_dim
    out_channel = config.dl.dataset.num_classes
    hidden_channel = config.train.model.hidden_dim
    num_layers = config.train.model.num_layers
    dropout = config.train.model.dropout
    graph_type = graph_type_dict[config.train.type.lower()]
    model = model_dict[config.train.model.type.lower()](
        in_channel, hidden_channel, out_channel, num_layers, dropout,
        graph_type, config, table)
    return model