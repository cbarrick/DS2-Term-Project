#!/usr/bin/env python3
import argparse
import logging

import numpy as np
import sklearn.metrics

import torch
import torch.autograd as A
import torch.nn as N
import torch.nn.functional as F
import torch.optim as O

from datasets import NucleiLoader
from datasets import EpitheliumLoader
from train import EWCTrainer
from models import AlexNet


logger = logging.getLogger()


def main(
    n_folds=5,
    batch_size=64,
    epochs=100,
    cuda=None,
    dry_run=False,
    log_level='DEBUG',
):
    logging.basicConfig(
        level=log_level,
        style='{',
        format='[{levelname:.4}][{asctime}][{name}:{lineno}] {msg}',
    )

    net = AlexNet(2)
    opt = O.Adam(net.parameters())
    loss = N.CrossEntropyLoss()
    model = EWCTrainer(net, opt, loss, cuda=cuda, dry_run=dry_run)

    tasks = {
        'nuclei': NucleiLoader(k=n_folds),
        'epithelium': EpitheliumLoader(k=n_folds),
    }

    metrics = {
        'f-measure': sklearn.metrics.f1_score,
        'precision': sklearn.metrics.precision_score,
        'recal': sklearn.metrics.recall_score,
        'log-loss': sklearn.metrics.log_loss,
    }

    for f in range(n_folds):
        print(f'================================ Fold {f} ================================')
        model.reset()

        for task, loader in tasks.items():
            print(f'-------- Training on {task} --------')
            train, validation, _ = loader.load(f, batch_size=batch_size)
            model.fit(train, validation, max_epochs=epochs)
            model.consolidate(validation)
            print()

        for task, loader in tasks.items():
            print(f'-------- Scoring {task} --------')
            _, _, test = loader.load(f, batch_size=batch_size)
            for metric, criteria in metrics.items():
                z = model.test(test, criteria)
                print(f'{metric}:', z)
            print()

        if dry_run:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the EWC experiment.')
    parser.add_argument('-k', '--n-folds', metavar='N', type=int, default=5, help='the number of cross-validation folds')
    parser.add_argument('-b', '--batch-size', metavar='N', type=int, default=64, help='the batch size')
    parser.add_argument('-e', '--epochs', metavar='N', type=int, default=100, help='the maximum number of epochs per task')
    parser.add_argument('-c', '--cuda', metavar='N', type=int, default=None, help='use the Nth cuda device')
    parser.add_argument('-d', '--dry-run', action='store_true', help='do a dry run to check for errors')
    parser.add_argument('-l', '--log-level', help='set the log level')
    args = parser.parse_args()
    main(**vars(args))
