from collections import namedtuple

import torch
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

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


def process_dataset(
    name: str,
    root: str,
    reverse_edges: bool = False,
    self_loop: bool = False,
) -> ProcessedDataset:
    dataset = DglNodePropPredDataset(name, root=root)

    split_idx = dataset.get_idx_split()

    train_idx = split_idx['train']
    valid_idx = split_idx['valid']
    test_idx = split_idx['test']

    g, labels = dataset[0]

    g.ndata['label'] = labels.squeeze(-1)

    if reverse_edges:
        src, dst = g.all_edges()

        g.add_edges(dst, src)


    if self_loop:
        g = g.remove_self_loop().add_self_loop()


    in_feats = g.ndata['feat'].shape[-1]
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
