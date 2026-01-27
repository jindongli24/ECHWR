import argparse
import json
import os
from glob import glob

import numpy as np
import torch
import yaml
from thop import profile

from echwr.model import ECHWR


def get_mean_std_cv(cfgs: dict, results: dict = {}) -> dict:
    '''Calculates statistics for cross-validation results.

    Iterates through result JSON files in the work directory, extracts the
    best Character Error Rate (CER) and Word Error Rate (WER), and computes
    mean and standard deviation.

    Args:
        cfgs: Configuration dictionary containing 'dir_work' and 'test' keys.
        results: Dictionary to append results to. Defaults to {}.

    Returns:
        Updated dictionary containing 'cer' and 'wer' statistics (raw, mean,
        std).
    '''
    cer, wer = {}, {}

    if paths_result := glob(
        os.path.join(
            cfgs['dir_work'],
            '*',
            'test_20*.json' if cfgs['test'] else 'train_20*.json',
        )
    ):
        for i, path_result in enumerate(sorted(paths_result)):
            with open(path_result, 'r') as f:
                result_fd = json.load(f)

            if cfgs['test']:
                result_best = result_fd['-1']['evaluation']
            else:
                epoch_best = result_fd['best']['character_error_rate'][0]
                result_best = result_fd[str(epoch_best)]['evaluation']

            cer[str(i)] = result_best['character_error_rate']
            wer[str(i)] = result_best['word_error_rate']

        results['cer'] = {
            'raw': cer,
            'mean': np.mean(list(cer.values())).item(),
            'std': np.std(list(cer.values())).item(),
        }
        results['wer'] = {
            'raw': wer,
            'mean': np.mean(list(wer.values())).item(),
            'std': np.std(list(wer.values())).item(),
        }
        results = {k: v for k, v in sorted(results.items())}

    return results


def get_macs_params(cfgs: dict, results: dict = {}) -> dict:
    '''Calculates the computational cost and model size.

    Computes the number of parameters and Multiply-Accumulate operations
    (MACs) using a dummy input.

    Args:
        cfgs: Configuration dictionary containing architecture details and
            channel counts.
        results: Dictionary to append results to. If None, a new dictionary
            is created. Defaults to None.

    Returns:
        Updated dictionary containing 'macs' and 'params'.
    '''
    model = ECHWR(
        cfgs['arch_en'],
        cfgs['arch_de'],
        cfgs['arch_pool'],
        cfgs['arch_txt'],
        cfgs['num_channel'],
        len(cfgs['categories']),
    ).eval()

    # generate dummy input based on dataset type
    x = torch.randn(
        1, cfgs['num_channel'], 1024 if 'word' in cfgs['dir_dataset'] else 4096
    )
    macs, params = profile(model, inputs=(x,))

    results['macs'] = int(macs)
    results['params'] = int(params)
    results = {k: v for k, v in sorted(results.items())}

    return results


def main(path_cfg: str) -> None:
    '''Main execution routine for single-experiment evaluation.

    Loads configuration, creates work directories, calculates cross-validation
    statistics, computes model complexity, and saves the aggregated results to
    JSON.

    Args:
        path_cfg: Path to the configuration YAML file.
    '''
    with open(path_cfg, 'r') as f:
        cfgs = yaml.safe_load(f)

    os.makedirs(cfgs['dir_work'], exist_ok=True)

    path_results = os.path.join(cfgs['dir_work'], 'results.json')

    if os.path.isfile(path_results):
        with open(path_results, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    results = get_mean_std_cv(cfgs, results)
    results = get_macs_params(cfgs, results)

    with open(path_results, 'w') as f:
        json.dump(results, f)

    print(results)


def main_ac(dir_work: str) -> None:
    '''Aggregates evaluation results across multiple dataset directories.

    Summarizes CER and WER across different sub-experiments (assumed to be
    split by folds 0-4) found within the work directory.

    Args:
        dir_work: Path to the root work directory containing sub-experiment
            folders.
    '''
    cer, wer = {'raw': {}}, {'raw': {}}

    for fname in glob(os.path.join(dir_work, '*', 'results.json')):
        with open(fname, 'r') as f:
            result = json.load(f)

        idx_1 = os.path.basename(os.path.dirname(fname))

        for idx_2 in ['0', '1', '2', '3', '4']:
            # check if fold data exists before accessing to avoid keyerror
            if idx_2 in result['cer']['raw']:
                cer['raw'][f'{idx_1}{idx_2}'] = result['cer']['raw'][idx_2]
                wer['raw'][f'{idx_1}{idx_2}'] = result['wer']['raw'][idx_2]

    cer['mean'] = np.mean(list(cer['raw'].values())).item()
    cer['std'] = np.std(list(cer['raw'].values())).item()
    wer['mean'] = np.mean(list(wer['raw'].values())).item()
    wer['std'] = np.std(list(wer['raw'].values())).item()
    results = {'cer': cer, 'wer': wer}

    with open(os.path.join(dir_work, 'results.json'), 'w') as f:
        json.dump(results, f)

    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate handwriting recognition model.'
    )
    parser.add_argument(
        '-c', '--config', help='Path to YAML file of configuration.'
    )
    args = parser.parse_args()

    if os.path.isfile(args.config):
        main(args.config)
    elif os.path.isdir(args.config):
        main_ac(args.config)
