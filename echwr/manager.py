import argparse
import json
import os
import time
from datetime import datetime

import torch
import yaml
from loguru import logger

from .utils import sec2time

__all__ = ['RunManager']


class RunManager:
    '''Manage work directory, logging, losses, and checkpoints.

    This class handles the boilerplate of creating output directories, saving
    configurations, logging training progress, and serializing model states.
    It employs a "fail fast" strategy: directories are created immediately
    upon instantiation to ensure the environment is valid before training
    starts.

    Args:
        cfgs: Configuration namespace containing experiment settings.

    Attributes:
        cfgs (argparse.Namespace): Configurations.
        metrics (dict): Dictionary storing training metrics.
        preds_test (dict): Dictionary storing test predictions.
        ts (str): Timestamp string for unique file naming.
        dir_ckp (str): Directory for checkpoints.
        dir_vis (str): Directory for visualizations.
        path_metrics (str): Path to the metrics JSON file.
        path_preds_test (str): Path to the test predictions JSON file.
        epoch (int): Current epoch number.
        loss (list): List of losses for the current epoch.
        num_iter (int): Total iterations in the current epoch.
        t_start (float): Start time of the current epoch.
        tag (str): Current phase tag ('train' or 'test').
    '''

    def __init__(self, cfgs: argparse.Namespace) -> None:
        self.cfgs = cfgs
        self.metrics = {}
        self.preds_test = {}

        self.ts = datetime.now().strftime('%Y%m%d%H%M%S')

        # initialize the work directory
        tag = 'test' if self.cfgs.test else 'train'

        self.dir_ckp = os.path.join(
            self.cfgs.dir_work, str(self.cfgs.idx_fold), 'checkpoints'
        )
        self.dir_vis = os.path.join(
            self.cfgs.dir_work, str(self.cfgs.idx_fold), 'visualization'
        )
        path_cfg = os.path.join(
            self.cfgs.dir_work,
            str(self.cfgs.idx_fold),
            f'{tag}_{self.ts}.yaml',
        )
        path_log = os.path.join(
            self.cfgs.dir_work, str(self.cfgs.idx_fold), f'{tag}_{self.ts}.log'
        )
        self.path_metrics = os.path.join(
            self.cfgs.dir_work,
            str(self.cfgs.idx_fold),
            f'{tag}_{self.ts}.json',
        )
        self.path_preds_test = os.path.join(
            self.cfgs.dir_work,
            str(self.cfgs.idx_fold),
            f'{tag}_preds_{self.ts}.json',
        )

        os.makedirs(self.dir_ckp, exist_ok=True)
        os.makedirs(self.dir_vis, exist_ok=True)

        with open(os.path.join(path_cfg), 'w') as f:
            yaml.safe_dump(vars(self.cfgs), f)

        logger.add(path_log)
        logger.info(
            f'Initialized work directory at {self.cfgs.dir_work} '
            f'for fold {self.cfgs.idx_fold}.'
        )

    def check_step(self, scur: int, mode: str) -> bool:
        '''Check whether the current step is desired according to frequency.

        The last step is always True regardless of the frequency.
        CALL AFTER initialize_epoch!

        Args:
            scur: Current step number.
            mode: Type of the step. Options are 'iter', 'eval', and 'save'.

        Returns:
            True if the current step triggers an action, False otherwise.
        '''
        match mode:
            case 'iter':
                return scur % self.cfgs.freq_log == 0 or scur == self.num_iter
            case 'eval':
                return (
                    scur % self.cfgs.freq_eval == 0 or scur == self.cfgs.epoch
                )
            case 'save':
                if self.cfgs.freq_save > 0:
                    return (
                        scur % self.cfgs.freq_save == 0 or scur == self.cfgs.epoch
                    )
                else:
                    return False

    def initialize_epoch(self, epoch: int, num_iter: int, val: bool) -> None:
        '''Initialize the recording variables for a new epoch.

        Args:
            epoch: Epoch number.
            num_iter: Maximum number of iterations of the current epoch.
            val: Whether the current epoch is for validation/test phases.
        '''
        self.epoch = epoch  # epoch life time
        self.loss = []  # epoch life time
        self.num_iter = num_iter  # epoch life time
        self.t_start = time.time()  # epoch life time
        self.tag = 'test' if val else 'train'  # epoch life time

        if not epoch in self.metrics.keys():
            self.metrics[epoch] = {}  # epoch life time

        self.metrics[epoch][self.tag] = []  # epoch life time

    def log(self, message: str) -> None:
        '''Log messages to the configured logger.

        Args:
            message: Message content to log.
        '''
        logger.info(message)

    def save_checkpoint(
        self,
        state_model: dict | None = None,
        state_optimizer: dict | None = None,
        state_lr_scheduler: dict | None = None,
    ) -> None:
        '''Save the model, optimizer, and scheduler states to a checkpoint.

        Args:
            state_model: State dictionary of the model. Defaults to None.
            state_optimizer: State dictionary of the optimizer. Defaults to
                None.
            state_lr_scheduler: State dictionary of the learning rate
                scheduler. Defaults to None.
        '''
        torch.save(
            {
                'epoch': self.epoch,
                'lr_scheduler': state_lr_scheduler,
                'model': state_model,
                'optimizer': state_optimizer,
            },
            os.path.join(self.dir_ckp, f'{self.epoch}.pth'),
        )
        logger.info(f'Saved checkpoint of epoch {self.epoch}')

    def save_results(self) -> None:
        '''Save the cached metrics and predictions to JSON files.'''
        with open(self.path_metrics, 'w') as f:
            json.dump(self.metrics, f)

        with open(self.path_preds_test, 'w') as f:
            json.dump(self.preds_test, f)

    def summarize_epoch(self) -> float:
        '''Summarize and save the results of the epoch.

        CALL AFTER initialize_epoch!

        Returns:
            Average loss of the epoch.
        '''
        t_end = time.time() - self.t_start
        loss_avg = sum(self.loss) / len(self.loss)
        result = {'loss_avg': loss_avg, 'time': t_end}
        logger.info(
            (
                f'{self.tag}, epoch: {self.epoch}, loss avg: {loss_avg:.7f}, '
                f'time: {sec2time(t_end)}'
            )
        )
        self.metrics[self.epoch][self.tag].append(result)
        self.save_results()

        return loss_avg

    def summarize_evaluation(self) -> None:
        '''Find and log the best metrics across all epochs.'''
        results_eval = [
            [epoch, result['evaluation']]
            for epoch, result in self.metrics.items()
            if 'evaluation' in result.keys()
        ]
        metrics = results_eval[0][1].keys()
        best = {metric: [-1, -1] for metric in metrics}  # [epoch, value]

        # iterate all results to get the best of each metrics
        for result in results_eval:
            for metric in metrics:
                if (
                    result[1][metric] < best[metric][1]
                    or best[metric][0] == -1
                ):
                    best[metric] = [result[0], float(result[1][metric])]

        self.metrics['best'] = best
        logger.info(f'best: {best}')
        self.save_results()

    def update_evaluation(
        self,
        result: dict,
        preds: list[str] = None,
        labels: list[str] = None,
    ) -> None:
        '''Update the evaluation results.

        Log predictions and labels if they are given.

        Args:
            result: Evaluation results dictionary.
            preds: List of predictions. Defaults to None.
            labels: List of labels. Defaults to None.
        '''
        self.metrics[self.epoch]['evaluation'] = result
        msg_log = [f'{key}: {val:.7f} ' for key, val in result.items()]
        logger.info(', '.join(msg_log))

        if labels:
            if 'labels' not in self.preds_test.keys():
                self.preds_test['labels'] = labels

        if preds:
            self.preds_test[self.epoch] = preds

        self.save_results()

    def update_iteration(
        self,
        iter: int,
        loss: float,
        lr: float = -1,
    ) -> None:
        '''Update the status of the iteration.

        If the current iteration is desired according to the frequency, logs
        the information. CALL AFTER initialize_epoch!

        Args:
            iter: Iteration number.
            loss: Loss value.
            lr: Current learning rate. Defaults to -1.
        '''
        self.loss.append(loss)

        if self.check_step(iter + 1, 'iter'):
            t_inter = time.time() - self.t_start
            result = {
                'lr': lr,
                'iters': iter + 1,
                'loss': loss,
                'time': t_inter,
            }
            self.metrics[self.epoch][self.tag].append(result)
            logger.info(
                (
                    f'{self.tag}, epoch: {self.epoch}, iters: {iter + 1}/'
                    f'{self.num_iter}, lr: {lr:.7f}, loss: {loss:.7f}, time: '
                    f'{sec2time(t_inter)}'
                )
            )
