import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from typing import Any, Optional
from lib_models.HNN.utils import zeros,glorot


class TaskLayerAdapter(nn.Module):
    """Task-conditioned bottleneck adapter applied after each HGNN layer."""

    TASK_TO_ID = {
        "node_cls": 0,
        "edge_pred": 1,
        "hg_cls": 2,
    }

    def __init__(self, channels: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.channels = int(channels)
        hidden_dim = int(hidden_dim)
        self.bottleneck_dim = hidden_dim if hidden_dim > 0 else max(8, self.channels // 8)
        self.condition_dim = max(16, self.channels // 4)
        self.task_embeddings = nn.Embedding(len(self.TASK_TO_ID), self.condition_dim)
        self.down_hypernet = nn.Linear(self.condition_dim, self.channels * self.bottleneck_dim)
        self.up_hypernet = nn.Linear(self.condition_dim, self.bottleneck_dim * self.channels)
        self.norms = nn.ModuleList([nn.LayerNorm(self.channels) for _ in self.TASK_TO_ID])
        self.residual_scales = nn.Parameter(torch.empty(len(self.TASK_TO_ID)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.task_embeddings.weight, std=0.02)
        nn.init.normal_(self.down_hypernet.weight, std=0.02)
        nn.init.zeros_(self.down_hypernet.bias)
        nn.init.zeros_(self.up_hypernet.weight)
        nn.init.zeros_(self.up_hypernet.bias)
        nn.init.constant_(self.residual_scales, 0.1)
        for norm in self.norms:
            norm.reset_parameters()

    @classmethod
    def _task_id(cls, task_type: Any) -> int:
        value = getattr(task_type, "value", task_type)
        value = "node_cls" if value is None else str(value)
        if value.startswith("TaskType."):
            value = value.split(".", 1)[1].lower()
        return cls.TASK_TO_ID.get(value, cls.TASK_TO_ID["node_cls"])

    def forward(self, x: Tensor, task_type: Any) -> Tensor:
        task_id = self._task_id(task_type)
        task_tensor = torch.tensor(task_id, dtype=torch.long, device=x.device)
        condition = self.task_embeddings(task_tensor)
        down = self.down_hypernet(condition).view(self.channels, self.bottleneck_dim)
        up = self.up_hypernet(condition).view(self.bottleneck_dim, self.channels)
        flat = x.reshape(-1, self.channels)
        delta = F.gelu(flat.matmul(down)).matmul(up)
        delta = self.norms[task_id](delta).view_as(x)
        scale = self.residual_scales[task_id].to(device=x.device, dtype=x.dtype)
        return x + scale * delta

class HypergraphConv(MessagePassing):

    def __init__(self, in_channels, out_channels, symdegnorm=False, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HypergraphConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.symdegnorm = symdegnorm

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = nn.Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.layer_id = 0

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)


    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                incidence_weight: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x (Tensor): Node feature matrix :math:`\mathbf{X}`
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Sparse hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges,device=x.device)
        if incidence_weight is None:
            incidence_weight = x.new_ones(hyperedge_index.size(1), device=x.device)
        else:
            incidence_weight = incidence_weight.to(device=x.device, dtype=x.dtype).view(-1)

        x = torch.matmul(x, self.weight)
        alpha = None
        if self.use_attention:
            assert num_edges <= num_edges
            x = x.view(-1, self.heads, self.out_channels)
            x_i, x_j = x[hyperedge_index[0]], x[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if not self.symdegnorm:
            D = scatter_add(hyperedge_weight[hyperedge_index[1]] * incidence_weight,
                            hyperedge_index[0], dim=0, dim_size=num_nodes)
            D = 1.0 / D
            D[D == float("inf")] = 0

            B = scatter_add(incidence_weight,
                            hyperedge_index[1], dim=0, dim_size=num_edges)  
            B = 1.0 / B
            B[B == float("inf")] = 0
            
            self.flow = 'source_to_target'
            edge_embed = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                                 incidence_weight=incidence_weight,
                                 size=(num_nodes, num_edges))
            self.flow = 'target_to_source'
            out = self.propagate(hyperedge_index, x=edge_embed, norm=D, alpha=alpha,
                                 incidence_weight=incidence_weight,
                                 size=(num_nodes, num_edges))
            
        else:  # this correspond to HGNN
            D = scatter_add(hyperedge_weight[hyperedge_index[1]] * incidence_weight,
                            hyperedge_index[0], dim=0, dim_size=num_nodes)
            D = 1.0 / D**(0.5)
            D[D == float("inf")] = 0

            B = scatter_add(incidence_weight,
                            hyperedge_index[1], dim=0, dim_size=num_edges)
            B = 1.0 / B
            B[B == float("inf")] = 0

            x = D.unsqueeze(-1)*x
            self.flow = 'source_to_target'
            edge_embed = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                                 incidence_weight=incidence_weight,
                                 size=(num_nodes, num_edges))

            self.flow = 'target_to_source'
            out = self.propagate(hyperedge_index,x=edge_embed, norm=D, alpha=alpha,
                                 incidence_weight=incidence_weight,
                                 size=(num_nodes, num_edges))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
            edge_embed = edge_embed.view(-1,self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            edge_embed = edge_embed.mean(dim=-1)

        if self.bias is not None:
            out = out + self.bias

        return out,edge_embed 

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor, incidence_weight: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels
        
        out = norm_i.view(-1, 1, 1) * incidence_weight.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
        return out

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class HGNN(nn.Module):

    def __init__(self, num_features, num_targets, args):
        super(HGNN, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout  # Note that default is 0.6
        self.symdegnorm = args.HGNN_symdegnorm
        self.hidden_dim = args.MLP_hidden

#       Note that add dropout to attention is default in the original paper
        self.convs = nn.ModuleList()
        if self.num_layers == 1:
            self.convs.append(HypergraphConv(num_features,
                               num_targets, self.symdegnorm))
        else:
            self.convs.append(HypergraphConv(num_features,
                                            self.hidden_dim, self.symdegnorm))
            for _ in range(self.num_layers-2):
                self.convs.append(HypergraphConv(
                    self.hidden_dim, self.hidden_dim, self.symdegnorm))
            # Output heads is set to 1 as default
            self.convs.append(HypergraphConv(
                self.hidden_dim, num_targets, self.symdegnorm))

        self.task_type = "node_cls"
        self.adapters = nn.ModuleList()
        self.use_adapter = False

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for adapter in self.adapters:
            adapter.reset_parameters()

    def enable_adapter(self, hidden_dim: int = 16):
        self.use_adapter = True
        self.adapters = nn.ModuleList(
            [TaskLayerAdapter(int(conv.heads * conv.out_channels), hidden_dim) for conv in self.convs]
        )

    def set_task_type(self, task_type):
        value = getattr(task_type, "value", task_type)
        self.task_type = "node_cls" if value is None else str(value)

    def _apply_adapter(self, layer_id: int, x: Tensor) -> Tensor:
        if not self.use_adapter or layer_id >= len(self.adapters):
            return x
        return self.adapters[layer_id](x, self.task_type)

    def forward(self, data):

        # regular node classification

        x = data.x
        edge_index = data.hyperedge_index
        incidence_weight = getattr(data, "incidence_weight", None)

        for i, conv in enumerate(self.convs[:-1]):
            x , e = conv(
                x,
                edge_index,
                incidence_weight=incidence_weight,
            )
            x = self._apply_adapter(i, x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x,e = self.convs[-1](
            x,
            edge_index,
            incidence_weight=incidence_weight,
        )
        x = self._apply_adapter(len(self.convs) - 1, x)

        return x,e

    @torch.no_grad()
    def predict(self,data):
        self.eval()
        return self.forward(data)
