"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

The module that contains utility functions
"""
from typing import Iterable
import argparse
import random
import numpy as np
import torch

def reset_seed(
    seed: int=7
):
    """
    Reset the random variables with the given seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_kwargs(
    args: Iterable[str],
    value: bool=False,
    default: object=None,
    kwargs=None,
):
    """
    Check the validity of kwargs
    """
    missing = [arg for arg in args if arg not in kwargs]
    if len(missing) > 0:
        raise TypeError(f'Required arguments {missing} are missing.')

    if value:
        not_updated = [arg for arg in args if kwargs[arg] is default]
        if len(not_updated) > 0:
            raise ValueError(f'Required argument values for {not_updated} are missing.')

def parse_arguments(
    return_default: bool=False,
) -> object:
    """
    Parse the arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed', type=int, default=7,
         help='Random seed'
    )
    parser.add_argument(
        '--gpu', type=int, default=0,
        help='GPU to use'
    )
    parser.add_argument(
        '--antecedent_len', type=int, default=4,
        help='Maximum antecedent length'
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=512,
        help='The hidden dimension of DNN base.'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=1e-5,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.0,
        help='Weight decay'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.95,
        help='Gamma for ExponentialLR scheduler'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Batch Size'
    )
    parser.add_argument(
        '--num_workers', type=int, default=8,
        help='Number of workers for preprocessing'
    )
    parser.add_argument(
        '--dataset', type=str, default='yelp',
        help='Dataset'
    )
    parser.add_argument(
        '--base', type=str, default='bert',
        help='Base Model'
    )
    parser.add_argument(
        '--epochs', type=int, default=10,
        help='Number of epochs'
    )
    parser.add_argument(
        '--num_atoms', type=int, default=5000,
        help='Number of words to use for atoms, Only for NLP datasets.'
    )
    parser.add_argument(
        '--min_df', type=int, default=200,
        help='Minimum data frequency for an antecedent '
    )
    parser.add_argument(
        '--max_df', type=float, default=0.95,
        help='Maximum data frequency for an antecedent'
    )
    parser.add_argument(
        '--pretrain_samples', type=int, default=10000,
        help='Number of samples for pretraining of consequent estimator.'
    )
    parser.add_argument(
        '--only_eval', action='store_true',
        help='Do only evaluation'
    )
    parser.add_argument(
        '--result_dir', type=str, default='result',
        help='The directory name to save results'
    )
    parser.add_argument(
        '--save_dir', type=str, default='save_dir',
        help='The directory to save interim files'
    )

    if return_default:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args
