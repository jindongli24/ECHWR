import argparse
import json
import os
from glob import glob

import numpy as np
import torch
import yaml
from scipy import stats
from thop import profile

from echwr.loss import ECHWRLoss
from echwr.model import ECHWR


def get_mean_ci_cv(
    cfgs: dict, results: dict = None, confidence: float = 0.95
) -> dict:
    '''calculates mean and confidence interval for cv results.

    iterates through result json files in the work directory, extracts the
    best character error rate (cer) and word error rate (wer), and computes
    mean and confidence interval bounds.

    Args:
        cfgs: configuration dict with 'dir_work' and 'test' keys.
        results: dict to append results to. defaults to an empty dict.
        confidence: the confidence level to calculate. defaults to 0.95.

    Returns:
        updated dictionary containing 'cer' and 'wer' statistics.
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

        if len(cer) > 1:
            # calculate the t-critical value for n-1 degrees of freedom
            t_crit = stats.t.ppf((1 + confidence) / 2, len(cer) - 1)

            vals_c, vals_w = list(cer.values()), list(wer.values())
            mean_c, mean_w = np.mean(vals_c).item(), np.mean(vals_w).item()
            std_c, std_w = np.std(vals_c).item(), np.std(vals_w).item()

            h_c = float(stats.sem(vals_c) * t_crit)
            h_w = float(stats.sem(vals_w) * t_crit)

            results['cer'] = {
                'raw': cer,
                'mean': mean_c,
                'std': std_c,
                'ci': h_c,
            }
            results['wer'] = {
                'raw': wer,
                'mean': mean_w,
                'std': std_w,
                'ci': h_w,
            }

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
        len_seq=cfgs['len_seq'],
    ).eval()

    # generate dummy input based on dataset type
    x_ts = torch.randn(
        1,
        cfgs['num_channel'],
        (
            1024
            if 'word' in cfgs['dir_dataset']
            or 'equation' in cfgs['dir_dataset']
            else 4096
        ),
    )
    x_txt = torch.randint(
        0,
        len(cfgs['categories']) - 1,
        (1 + cfgs['num_corr_txt'], 1, cfgs['len_context']),
    )
    macs_train, params_train = profile(
        model,
        inputs=(
            x_ts,
            x_txt,
            cfgs['tasks'],
        ),
    )
    macs_infer, params_infer = profile(model, inputs=(x_ts,))

    results['params'] = {
        'train': int(params_train),
        'infer': int(params_infer),
    }
    results['macs'] = {'train': int(macs_train), 'infer': int(macs_infer)}

    return results


def get_train_metrics(
    cfgs: dict, results: dict = {}, num_iters: int = 100
) -> dict:
    '''Calculates training iteration time.

    Benchmarks a dummy forward and backward pass using CUDA events
    to estimate the true time per training iteration.

    Args:
        cfgs: Configuration dictionary containing architecture details.
        results: Dictionary to append results to. Defaults to {}.
        num_iters: Iterations for the benchmark. Defaults to 100.

    Returns:
        Updated dictionary containing 'time_per_iter'.
    '''
    model = (
        ECHWR(
            cfgs['arch_en'],
            cfgs['arch_de'],
            cfgs['arch_pool'],
            cfgs['arch_txt'],
            cfgs['num_channel'],
            len(cfgs['categories']),
            len_context=cfgs['len_context'],
            len_seq=cfgs['len_seq'],
        )
        .train()
        .cuda()
    )

    len_seq = (
        1024
        if 'word' in cfgs['dir_dataset'] or 'equation' in cfgs['dir_dataset']
        else 4096
    )
    x_ts = torch.randn(1, cfgs['num_channel'], len_seq).cuda()
    x_txt = torch.randint(
        0,
        len(cfgs['categories']) - 1,
        (1 + cfgs['num_corr_txt'], 1, cfgs['len_context']),
    ).cuda()

    fn_loss = ECHWRLoss()

    # warm-up runs to stabilize hardware timings
    for _ in range(5):
        model.zero_grad()
        out = model(x_ts, x_txt, cfgs['tasks'])
        loss = fn_loss(
            out,
            x_txt[0],
            cfgs['tasks'],
            (int(len_seq // model.ratio_ds),),
            (int(cfgs['len_context']),),
            model.scale,
        )
        loss.backward()

    # cuda events for accurate hardware timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # benchmark loop
    start_event.record()
    for _ in range(num_iters):
        model.zero_grad()
        out = model(x_ts, x_txt, cfgs['tasks'])
        loss = fn_loss(
            out,
            x_txt[0],
            cfgs['tasks'],
            (int(len_seq // model.ratio_ds),),
            (int(cfgs['len_context']),),
            model.scale,
        )
        loss.backward()
    end_event.record()

    # force cpu to wait for gpu to finish all tasks
    torch.cuda.synchronize()

    # elapsed_time returns milliseconds, convert to seconds
    total_time_sec = start_event.elapsed_time(end_event)
    results['time_per_iter'] = total_time_sec / num_iters

    return results


def main(path_cfg: str, tt: bool = False) -> None:
    '''Main execution routine for single-experiment evaluation.

    Loads configuration, creates work directories, calculates cross-validation
    statistics, computes model complexity, and saves the aggregated results to
    JSON.

    Args:
        path_cfg: Path to the configuration YAML file.
        tt: Whether to calculate the training time. Defaults to False.
    '''
    with open(path_cfg, 'r') as f:
        cfgs = yaml.safe_load(f)

    os.makedirs(cfgs['dir_work'], exist_ok=True)

    path_results = os.path.join(cfgs['dir_work'], 'results.json')
    results = {}

    results = get_mean_ci_cv(cfgs, results)
    results = get_macs_params(cfgs, results)

    if tt:
        results = get_train_metrics(cfgs, results)
    else:
        results['time_per_iter'] = None

    with open(path_results, 'w') as f:
        json.dump(results, f)

    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate handwriting recognition model.'
    )
    parser.add_argument(
        '-c', '--config', help='Path to YAML file of configuration.'
    )
    parser.add_argument(
        '-tt',
        '--training-time',
        action='store_true',
        help='Whether to calculate the training time.',
    )
    args = parser.parse_args()

    main(args.config, args.training_time)
