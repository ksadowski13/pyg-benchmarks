from typing import Callable

import torch
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.nn import GATConv

# REFERENCE PyG GAT EXAMPLE
# class Net(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()

#         self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
#         # On the Pubmed dataset, use heads=8 in conv2.
#         self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=False,
#                              dropout=0.6)

#     def forward(self, x, edge_index):
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=-1)

class PyGGAT(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        num_heads: int,
        num_layers: int,
        batch_norm: bool = False,
        input_dropout: float = 0,
        dropout: float = 0,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__()
        self._hidden_feats = hidden_feats
        self._out_feats = out_feats
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._input_dropout = nn.Dropout(input_dropout)
        self._dropout = nn.Dropout(dropout)
        self._activation = activation

        self._layers = nn.ModuleList()

        self._layers.append(GATConv(in_feats, hidden_feats, num_heads))

        for i in range(1, num_layers - 1):
            self._layers.append(GATConv(num_heads * hidden_feats, hidden_feats, num_heads))

        self._layers.append(GATConv(num_heads * hidden_feats, out_feats, num_heads))

        if batch_norm:
            self._batch_norms = nn.ModuleList()

            for i in range(num_layers - 1):
                self._batch_norms.append(nn.BatchNorm1d(self._hidden_feats[i]))

        else:
            self._batch_norms = None

    def _apply_layers(
        self,
        layer_idx: int,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = inputs

        if self._batch_norms is not None:
            x = self._batch_norms[layer_idx](x)

        if self._activation is not None:
            x = self._activation(x)

        x = self._dropout(x)

        return x

    def forward(
        self,
        g,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = self._input_dropout(inputs)

        if isinstance(g, list):
            for i, (layer, block) in enumerate(zip(self._layers, g)):
                edge_index, _, size = block

                x_target = x[:size[-1]]

                x = layer((x, x_target), edge_index)

                if i < self._num_layers - 1:
                    x = self._apply_layers(i, x)
        else:
            for i, layer in enumerate(self._layers):
                x = layer(x, g)

                if i < self._num_layers - 1:
                    x = self._apply_layers(i, x)

        x = x.squeeze(-1)

        return x

    def inference(
        self,
        g,
        inputs: torch.Tensor,
        batch_size: int,
        num_workers: int,
        device: torch.device,
    ) -> torch.Tensor:
        x = inputs

        for i, layer in enumerate(self._layers):
            hidden_dim = self._hidden_feats if i < self._num_layers - 1 else self._out_feats

            y = torch.zeros((g.num_nodes, hidden_dim))

            dataloader = pyg.loader.NeighborSampler(
                g.edge_index,
                [-1],
                node_idx=None,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers,
            )

            for batch_size_, nids, block in dataloader:
                edge_index, _, size = block.to(device)

                x_ = x[nids].to(device)
                x_target = x_[:size[-1]]

                h = layer((x_, x_target), edge_index)

                if i < self._num_layers - 1:
                    h = self._apply_layers(i, h)

                y[nids[:batch_size_]] = h.cpu()

            x = y

        x = x.squeeze(-1)

        return x
