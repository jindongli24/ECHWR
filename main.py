import argparse
import os
import warnings

import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from echwr.dataset import ECHWDataset, fn_collate
from echwr.decoder_ctc import BestPath
from echwr.evaluate import evaluate
from echwr.loss import ECHWRLoss
from echwr.manager import RunManager
from echwr.model import ECHWR
from echwr.utils import seed_everything, seed_worker
from echwr.visualize import visualize

warnings.filterwarnings('ignore', category=UserWarning)


def train_one_epoch(
    dataloader: DataLoader,
    model: ECHWR,
    fn_loss: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    lr_scheduler: torch.optim.lr_scheduler.SequentialLR,
    manager: RunManager,
    epoch: int,
) -> None:
    '''Train model for 1 epoch.

    Args:
        dataloader: Dataloader of training set.
        model: Model instance.
        fn_loss: Loss function module.
        optimizer: Optimizer instance.
        scaler: Scaler for mixed-precision training.
        lr_scheduler: Learning rate scheduler.
        manager: Running manager instance.
        epoch: Current epoch number.
    '''
    manager.initialize_epoch(epoch, len(dataloader), False)
    model.train()

    for idx, (seq, txts, len_seq, len_txt) in enumerate(dataloader):
        seq = seq.to(manager.cfgs.device)
        txts = txts.to(manager.cfgs.device)
        optimizer.zero_grad()

        with torch.autocast('cuda', torch.float16):
            outputs = model(seq, txts, manager.cfgs.tasks)
            loss = fn_loss(
                outputs,
                txts[0],
                manager.cfgs.tasks,
                len_seq // model.ratio_ds,
                len_txt,
                model.scale,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        manager.update_iteration(
            idx,
            loss.item(),
            lr_scheduler.get_last_lr()[0],
        )

    manager.summarize_epoch()

    # save checkpoints every freq_save epoch
    if manager.check_step(epoch + 1, 'save'):
        manager.save_checkpoint(model.state_dict())


def test(
    dataloader: DataLoader,
    model: ECHWR,
    fn_loss: nn.Module,
    manager: RunManager,
    ctc_decoder: BestPath,
    epoch: int = None,
) -> None:
    '''Test the model.

    Args:
        dataloader: DataLoader of test set.
        model: Model instance.
        fn_loss: Loss function module.
        manager: Running manager instance.
        ctc_decoder: CTC decoder instance.
        epoch: Epoch number. Defaults to None.
    '''
    preds, labels = [], []
    manager.initialize_epoch(epoch, len(dataloader), True)
    model.eval()

    with torch.no_grad():
        for idx, (seq, txts, len_seq, len_txt) in enumerate(dataloader):
            seq = seq.to(manager.cfgs.device)
            txts = txts.to(manager.cfgs.device)
            outputs = model(seq)
            loss = fn_loss(
                outputs,
                txts[0],
                ['hwr'],
                len_seq // model.rewi.ratio_ds,
                len_txt,
            )
            manager.update_iteration(idx, loss.item())

            # decode and cache results every freq_eval epoch
            if manager.check_step(epoch + 1, 'eval'):
                for pred, len_pred, label in zip(
                    outputs['out_hwr'].cpu(),
                    len_seq // model.rewi.ratio_ds,
                    txts[0].cpu(),
                ):
                    preds.append(ctc_decoder.decode(pred[:len_pred]))
                    labels.append(ctc_decoder.decode(label, True))

    manager.summarize_epoch()

    # evaluate every freq_eval epoch
    if manager.check_step(epoch + 1, 'eval'):
        visualize(
            preds, labels, manager.cfgs.categories[1:], manager.dir_vis, epoch
        )
        results_eval = evaluate(preds, labels)
        manager.update_evaluation(results_eval, preds, labels)


def main(cfgs: argparse.Namespace) -> None:
    '''Main function for training and evaluation.

    Args:
        cfgs: Configurations.
    '''
    # initialize the environment
    manager = RunManager(cfgs)
    seed_everything(cfgs.seed)
    ctc_decoder = BestPath(cfgs.categories)
    model = ECHWR(
        cfgs.arch_en,
        cfgs.arch_de,
        cfgs.arch_pool,
        cfgs.arch_txt,
        cfgs.num_channel,
        len(cfgs.categories),
        len_context=cfgs.len_context,
        len_seq=cfgs.len_seq,
    ).to(cfgs.device)
    dataset_test = ECHWDataset(
        os.path.join(cfgs.dir_dataset, 'val.json'),
        cfgs.idx_fold,
        cfgs.categories,
        model.rewi.ratio_ds,
        len_context=cfgs.len_context,
        len_seq=cfgs.len_seq,
        cache=cfgs.cache,
    )
    dataloader_test = DataLoader(
        dataset_test,
        cfgs.size_batch,
        num_workers=cfgs.num_worker,
        collate_fn=fn_collate,
    )
    fn_loss = ECHWRLoss()
    epoch_start = 0

    if not cfgs.test:
        dataset_train = ECHWDataset(
            os.path.join(cfgs.dir_dataset, 'train.json'),
            cfgs.idx_fold,
            cfgs.categories,
            model.rewi.ratio_ds,
            cfgs.aug,
            cfgs.len_context,
            cfgs.len_seq,
            cfgs.num_corr_txt,
            cfgs.cache,
        )
        dataloader_train = DataLoader(
            dataset_train,
            cfgs.size_batch,
            True,
            num_workers=cfgs.num_worker,
            collate_fn=fn_collate,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(cfgs.seed),
        )
        optimizer = torch.optim.AdamW(
            [
                {'params': model.rewi.parameters(), 'lr': cfgs.lr},
                {
                    'params': [
                        p
                        for name, p in model.named_parameters()
                        if 'rewi' not in name and p.requires_grad
                    ],
                    'lr': cfgs.lr_trans,
                },
            ]
        )
        scaler = GradScaler()
        lr_scheduler = SequentialLR(
            optimizer,
            [
                LinearLR(
                    optimizer,
                    0.01,
                    total_iters=len(dataloader_train) * cfgs.epoch_warmup,
                ),
                CosineAnnealingLR(
                    optimizer,
                    len(dataloader_train) * (cfgs.epoch - cfgs.epoch_warmup),
                ),
            ],
            [len(dataloader_train) * cfgs.epoch_warmup],
        )

    # load checkpoint for testing if given
    if cfgs.checkpoint:
        ckp = torch.load(cfgs.checkpoint, weights_only=False)
        model.load_state_dict(ckp['model'], strict=False)
        manager.log(f'Load checkpoint from {cfgs.checkpoint}')

    # start running
    for e in range(epoch_start, cfgs.epoch):
        if cfgs.test:
            test(
                dataloader_test,
                model,
                fn_loss,
                manager,
                ctc_decoder,
                -1,
            )
            break
        else:
            train_one_epoch(
                dataloader_train,
                model,
                fn_loss,
                optimizer,
                scaler,
                lr_scheduler,
                manager,
                e,
            )
            test(
                dataloader_test,
                model,
                fn_loss,
                manager,
                ctc_decoder,
                e,
            )

    if not cfgs.test:
        manager.summarize_evaluation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run handwriting recognition model.'
    )
    parser.add_argument(
        '-c', '--config', help='Path to YAML file of configuration.'
    )
    args = parser.parse_args()
    # args.config = 'configs/train.yaml'  # ONLY for debugging

    with open(args.config, 'r') as f:
        cfgs = yaml.safe_load(f)
        cfgs = argparse.Namespace(**cfgs)

    main(cfgs)
