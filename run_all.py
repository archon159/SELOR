"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.

Run all files that need to train a selor model with given arguments
"""
import os

from selor_utils import utils

if __name__ == "__main__":
    cur_args = utils.parse_arguments().__dict__
    default_args = utils.parse_arguments(return_default=True).__dict__

    option = []
    for key, value in cur_args.items():
         # Always show dataset, base model, gpu, seed
        if value is not default_args[key] or key in ['dataset', 'base', 'gpu', 'seed']:
            option.append(f'--{key} {value}')
    option = ' '.join(option)

    files_to_run = [
        'base.py',
        'extract_base_embedding.py',
        'build_atom_pool.py',
        'sample_antecedents.py',
        'pretrain_consequent_estimator.py',
        'selor.py'
    ]

    for file in files_to_run:
        cmd = f'python3 {file} {option}'
        print(cmd)
        os.system(cmd)
        print()
