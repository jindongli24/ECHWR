import argparse
import json
import os
from copy import deepcopy

import yaml


def train_cv(cfgs: dict, path_main: str) -> None:
    '''Orchestrates cross-validation training.

    This function reads the number of folds from the dataset configuration,
    generates temporary YAML configuration files for each fold, and executes
    the training script sequentially for all folds using a chained shell
    command.

    Args:
        cfgs: Dictionary containing the base training configuration.
        path_main: File path to the main Python training script (e.g.,
            'main.py').
    '''
    with open(os.path.join(cfgs['dir_dataset'], 'train.json'), 'r') as f:
        num_fd = json.load(f)['info']['num_fold']

    dir_temp = f'temp_{os.path.basename(cfgs["dir_work"])}'
    os.makedirs(dir_temp, exist_ok=True)

    command = []
    separator = ' && '

    for i in range(num_fd):
        cfgs_new = deepcopy(cfgs)
        cfgs_new['idx_fold'] = i
        path_temp = os.path.join(dir_temp, f'f{i}.yaml')

        # replace placeholder '-1' with actual fold index in checkpoint path
        if cfgs_new['checkpoint'] and '-1' in cfgs_new['checkpoint']:
            cfgs_new['checkpoint'] = cfgs_new['checkpoint'].replace(
                '-1', str(i)
            )

        with open(path_temp, 'w') as f:
            yaml.safe_dump(cfgs_new, f, allow_unicode=True)

        command.append(f'python {path_main} -c {path_temp}')

    # chain commands and cleanup temp dir at the end
    command = separator.join(command) + f' && rm -rf {dir_temp}'
    print(command)
    os.system(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run handwriting recognition model with cross validation.'
    )
    parser.add_argument(
        '-c', '--config', help='Path to the YAML file of configuration.'
    )
    parser.add_argument(
        '-m',
        '--main',
        help='Path to the Python script for training.',
        default='main.py',
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfgs = yaml.safe_load(f)

    assert (
        cfgs['idx_fold'] == -1
    ), 'Please use cross-validation training configuration (idx_fold=-1).'

    train_cv(cfgs, args.main)
