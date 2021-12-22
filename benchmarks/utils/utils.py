from collections import namedtuple

import torch
import torch_geometric as pyg
from ogb.nodeproppred import (DglNodePropPredDataset, Evaluator,
                              PygNodePropPredDataset)

ProcessedDataset = namedtuple(
    'ProcessedDataset',
    [
        'g',
        'train_idx',
        'valid_idx',
        'test_idx',
        'in_feats',
        'out_feats',
        'evaluator',
    ],
)


class LibraryError(Exception):
    def __init__(self, library: str) -> None:
        super().__init__()
        self._message = f'Wrong library provided ({library}), choices are (dgl, pyg).'

    def __str__(self) -> str:
        return self._message


def process_dataset(
    library: str,
    name: str,
    root: str,
    reverse_edges: bool = False,
    self_loop: bool = False,
) -> ProcessedDataset:
    if library == 'dgl':
        dataset = DglNodePropPredDataset(name, root=root)
    else:
        dataset = PygNodePropPredDataset(name, root=root)

    split_idx = dataset.get_idx_split()

    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    if library == 'dgl':
        g, labels = dataset[0]

        g.ndata['label'] = labels.squeeze(-1)
    else:
        g = dataset[0]

        g.y = g.y.squeeze(-1)

    if reverse_edges:
        if library == 'dgl':
            src, dst = g.all_edges()

            g.add_edges(dst, src)
        else:
            g.edge_index = pyg.utils.to_undirected(g.edge_index)

    if self_loop:
        if library == 'dgl':
            g = g.remove_self_loop().add_self_loop()
        else:
            edge_index, _ = pyg.utils.remove_self_loops(g.edge_index)
            edge_index, _ = pyg.utils.add_self_loops(edge_index)

            g.edge_index = edge_index

    if library == 'dgl':
        in_feats = g.ndata['feat'].shape[-1]
    else:
        in_feats = g.x.shape[-1]

    out_feats = dataset.num_classes

    evaluator = Evaluator(name)

    processed_dataset = ProcessedDataset(
        g,
        train_idx,
        valid_idx,
        test_idx,
        in_feats,
        out_feats,
        evaluator,
    )

    return processed_dataset


def get_evaluation_score(
    evaluator: Evaluator,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    if labels.dim() > 1:
        y_pred = logits
        y_true = labels
    else:
        y_pred = logits.argmax(dim=-1, keepdim=True)
        y_true = labels.unsqueeze(dim=-1)

    _, score = evaluator.eval({
        'y_pred': y_pred,
        'y_true': y_true,
    }).popitem()

    return score


if __name__ == '__main__':
    process_dataset('ogbn-arxiv', '/home/ksadowski/datasets',
                    '/home/ksadowski/pyg_datasets', self_loop=True, reverse_edges=True)
