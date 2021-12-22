from typing import Callable, Union

import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import SAGEConv


class DGLGraphSAGE(nn.Module):
    def __init__(
        self,
        in_feats: int,
        hidden_feats: int,
        out_feats: int,
        num_layers: int,
        batch_norm: bool = False,
        input_dropout: float = 0,
        dropout: float = 0,
        activation: Callable[[torch.Tensor], torch.Tensor] = None,
    ):
        super().__init__()
        self._hidden_feats = hidden_feats
        self._out_feats = out_feats
        self._num_layers = num_layers
        self._input_dropout = nn.Dropout(input_dropout)
        self._dropout = nn.Dropout(dropout)
        self._activation = activation

        self._layers = nn.ModuleList()

        self._layers.append(SAGEConv(in_feats, hidden_feats, 'mean'))

        for i in range(1, num_layers - 1):
            self._layers.append(SAGEConv(hidden_feats, hidden_feats, 'mean'))

        self._layers.append(SAGEConv(hidden_feats, out_feats, 'mean'))

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
        g: Union[dgl.DGLGraph, list[dgl.DGLGraph]],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = self._input_dropout(inputs)

        if isinstance(g, list):
            for i, (layer, block) in enumerate(zip(self._layers, g)):
                x = layer(block, x)

                if i < self._num_layers - 1:
                    x = self._apply_layers(i, x)
        else:
            for i, layer in enumerate(self._layers):
                x = layer(g, x)

                if i < self._num_layers - 1:
                    x = self._apply_layers(i, x)

        x = x.squeeze(-1)

        return x

    def inference(
        self,
        g: dgl.DGLGraph,
        inputs: torch.Tensor,
        batch_size: int,
        num_workers: int,
        device: torch.device,
    ) -> torch.Tensor:
        x = inputs

        for i, layer in enumerate(self._layers):
            hidden_dim = self._hidden_feats if i < self._num_layers - 1 else self._out_feats

            y = torch.zeros((g.num_nodes(), hidden_dim))

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                g.nodes(),
                sampler,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,
                drop_last=False,
            )

            for in_nodes, out_nodes, blocks in dataloader:
                block = blocks[0].int().to(device)

                h = layer(block, x[in_nodes].to(device))

                if i < self._num_layers - 1:
                    h = self._apply_layers(i, h)

                y[out_nodes] = h.cpu()

            x = y

        x = x.squeeze(-1)

        return x
