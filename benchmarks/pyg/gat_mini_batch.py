import argparse
from timeit import default_timer
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import utils
from ogb.nodeproppred import Evaluator

from gat import GAT


def train(
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    evaluator: Evaluator,
    dataloader: pyg.loader.NeighborSampler,
    g=None,
) -> tuple[float]:
    model.train()

    total_loss = 0
    total_score = 0

    start = default_timer()

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch_size, nids, blocks = batch

        blocks = [block.to(device) for block in blocks]

        inputs = g.x[nids].to(device)
        labels = g.y[nids[:batch_size]].to(device)

        logits = model(blocks, inputs)

        loss = loss_function(logits, labels)
        score = utils.get_evaluation_score(evaluator, logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_score += score

    stop = default_timer()
    time = stop - start

    total_loss /= step + 1
    total_score /= step + 1

    return total_loss, total_score, time


def validate(
    model: nn.Module,
    device: torch.device,
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    evaluator: Evaluator,
    g: pyg.data.Data,
    mask: torch.Tensor,
    batch_size: int,
    num_workers: int,
) -> tuple[float]:
    model.eval()

    start = default_timer()

    inputs = g.x
    labels = g.y[mask]

    with torch.inference_mode():
        logits = model.inference(
            g, inputs, batch_size, num_workers, device)[mask]

        loss = loss_function(logits, labels)
        score = utils.get_evaluation_score(evaluator, logits, labels)

    stop = default_timer()
    time = stop - start

    loss = loss.item()

    return loss, score, time


def run(args: argparse.ArgumentParser) -> None:
    torch.manual_seed(args.seed)

    processed_dataset = utils.process_dataset(
        args.dataset,
        args.dataset_root,
        reverse_edges=args.graph_reverse_edges,
        self_loop=args.graph_self_loop,
    )

    activations = {'leaky_relu': F.leaky_relu, 'relu': F.relu}

    train_dataloader = pyg.loader.NeighborSampler(
        processed_dataset.g.edge_index,
        args.fanouts,
        node_idx=processed_dataset.train_idx,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )

    model = GAT(
        processed_dataset.in_feats,
        args.hidden_feats,
        processed_dataset.out_feats,
        args.num_heads,
        args.num_layers,
        batch_norm=args.batch_norm,
        input_dropout=args.input_dropout,
        dropout=args.dropout,
        activation=activations[args.activation],
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_function = nn.CrossEntropyLoss()

    train_times = []
    inference_times = []

    print(f'## Started training ##')

    for epoch in range(args.num_epochs):
        train_loss, train_score, train_time = train(
            model,
            args.device,
            optimizer,
            loss_function,
            processed_dataset.evaluator,
            train_dataloader,
            g=processed_dataset.g,
        )
        valid_loss, valid_score, valid_time = validate(
            model,
            args.device,
            loss_function,
            processed_dataset.evaluator,
            processed_dataset.g,
            processed_dataset.valid_idx,
            args.eval_batch_size,
            args.num_workers,
        )

        # checkpoint.create(
        #     epoch,
        #     train_time,
        #     valid_time,
        #     train_loss,
        #     valid_loss,
        #     train_score,
        #     valid_score,
        #     model,
        # )

        print(
            f'Epoch: {epoch + 1:03} '
            f'Train Loss: {train_loss:.2f} '
            f'Valid Loss: {valid_loss:.2f} '
            f'Train Score: {train_score:.4f} '
            f'Valid Score: {valid_score:.4f} '
            f'Train Epoch Time: {train_time:.2f} '
            f'Valid Epoch Time: {valid_time:.2f}'
        )

        if 9 <= epoch <= 19:
            train_times.append(train_time)
            inference_times.append(valid_time)

    if args.test_validation:
        # model.load_state_dict(checkpoint.best_epoch_model_parameters)

        test_loss, test_score, test_time = validate(
            model,
            args.device,
            loss_function,
            processed_dataset.evaluator,
            processed_dataset.g,
            processed_dataset.test_idx,
            args.eval_batch_size,
            args.num_workers,
        )

        print(
            f'Test Loss: {test_loss:.2f} '
            f'Test Score: {test_score * 100:.2f} % '
            f'Test Epoch Time: {test_time:.2f}'
        )

    print(f'## Finished training ##')

    print(f'Average train epoch time: {np.mean(train_times)}')
    print(f'Average inference epoch time: {np.mean(inference_times)}')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('GAT PyG Benchmarking')

    argparser.add_argument('--device', default='cpu', type=str,
                           choices=['cpu', 'cuda'])
    argparser.add_argument('--dataset', default='ogbn-products', type=str,
                           choices=['ogbn-arxiv', 'ogbn-products'])
    argparser.add_argument('--dataset-root', default='dataset', type=str)
    argparser.add_argument('--graph-reverse-edges', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--graph-self-loop', default=True,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--num-epochs', default=20, type=int)
    argparser.add_argument('--lr', default=0.003, type=float)
    argparser.add_argument('--hidden-feats', default=256, type=int)
    argparser.add_argument('--num-layers', default=3, type=int)
    argparser.add_argument('--num-heads', default=3, type=int)
    argparser.add_argument('--batch-norm', default=False,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--input-dropout', default=0.1, type=float)
    argparser.add_argument('--dropout', default=0.5, type=float)
    argparser.add_argument('--activation', default='relu',
                           type=str, choices=['leaky_relu', 'relu'])
    argparser.add_argument('--batch-size', default=1024, type=int)
    argparser.add_argument('--eval-batch-size', default=4096, type=int)
    argparser.add_argument('--fanouts', default=[5, 10, 15],
                           nargs='+', type=str)
    argparser.add_argument('--num-workers', default=4, type=int)
    argparser.add_argument('--early-stopping-patience', default=10, type=int)
    argparser.add_argument('--early-stopping-monitor',
                           default='loss', type=str)
    argparser.add_argument('--test-validation', default=True,
                           action=argparse.BooleanOptionalAction)
    argparser.add_argument('--seed', default=13, type=int)

    args = argparser.parse_args()

    run(args)
