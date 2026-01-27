import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
import umap.plot as uplt
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from echwr.dataset import ECHWDataset, fn_collate
from echwr.decoder_ctc import BestPath
from echwr.model import ECHWR


def get_embeds(cfgs: argparse.Namespace) -> dict[str, list[np.ndarray]]:
    model = ECHWR(
        cfgs.arch_en,
        cfgs.arch_de,
        cfgs.arch_pool,
        cfgs.arch_txt,
        cfgs.num_channel,
        len(cfgs.categories),
        len_context=cfgs.len_context,
    )
    dataset = ECHWDataset(
        os.path.join(cfgs.dir_dataset, 'val.json'),
        cfgs.idx_fold,
        cfgs.categories,
        len_context=cfgs.len_context,
        num_corr_txt=cfgs.num_corr_txt,
    )
    dataloader = DataLoader(
        dataset, collate_fn=fn_collate, batch_size=cfgs.size_batch
    )
    ctc_decoder = BestPath(cfgs.categories)

    model.load_state_dict(
        torch.load(cfgs.checkpoint, weights_only=False)['model']
    )
    model.to(cfgs.device).eval()

    embeds_seq = {}
    embeds_txt = {}
    embeds_error = {}

    with torch.no_grad():
        for seq, txt, _, len_txt in tqdm(dataloader):
            label = ctc_decoder.decode(txt[0, 0, : len_txt[0]], True)
            seq, txt = seq.to(cfgs.device), txt.to(cfgs.device)
            outputs = model(seq, txt, tasks=['bc', 'ec'])

            if label in embeds_seq.keys():
                embeds_seq[label].append(outputs['embed_seq'][0].cpu().numpy())
            else:
                embeds_seq[label] = [outputs['embed_seq'][0].cpu().numpy()]

            if not label in embeds_txt.keys():
                embeds_txt[label] = [outputs['embeds_txt'][0, 0].cpu().numpy()]

            if outputs['embeds_txt'].shape[0] > 1:
                embeds = [
                    embed.cpu().numpy()
                    for embed in outputs['embeds_txt'][1:, 0]
                ]

                if label in embeds_error.keys():
                    embeds_error[label].append(embeds)
                else:
                    embeds_error[label] = [embeds]

    embeds_seq = dict(
        sorted(embeds_seq.items(), key=lambda item: len(item[1]), reverse=True)
    )

    if bool(embeds_txt):
        embeds_txt = {k: embeds_txt[k] for k in embeds_seq.keys()}

    if bool(embeds_error):
        embeds_error = {k: embeds_error[k] for k in embeds_seq.keys()}

    return embeds_seq, embeds_txt, embeds_error


def plot_umap(
    num_label: int,
    embeds_seq: dict = {},
    embeds_txt: dict = {},
    embeds_error: dict = {},
    path_save: str = '',
    title: str = '',
) -> None:
    embeds_all = []
    labels_all = []
    types_all = []

    if bool(embeds_seq):
        for label, embeds in list(embeds_seq.items())[:num_label]:
            for embed in embeds:
                embeds_all.append(embed)
                labels_all.append(label)
                types_all.append(0)

    if bool(embeds_txt):
        for label, embeds in list(embeds_txt.items())[:num_label]:
            embeds_all.append(embeds[0])
            labels_all.append(label)
            types_all.append(1)

    if bool(embeds_error):
        for label, embeds in list(embeds_error.items())[:num_label]:
            for embed in embeds:
                for e in embed:
                    embeds_all.append(e)
                    labels_all.append(label + '_error')
                    types_all.append(2)

    embeds_all = np.array(embeds_all)
    labels_all = np.array(labels_all)
    types_all = np.array(types_all)

    reducer = umap.UMAP(random_state=42)
    reducer.fit(embeds_all)

    mask_txt = types_all == 1
    embeds_txt_umap = reducer.embedding_[mask_txt]
    labels_txt = labels_all[mask_txt]
    reducer.embedding_ = reducer.embedding_[~mask_txt]
    labels_all = labels_all[~mask_txt]
    types_all = types_all[~mask_txt]

    unique_labels = np.unique(labels_all)
    cmap = plt.get_cmap('tab20')
    color_key = {label: cmap(i % 20) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(10, 10))
    ax = uplt.points(reducer, labels=labels_all, color_key=color_key)

    if sum(mask_txt):
        ax.scatter(
            embeds_txt_umap[:, 0],
            embeds_txt_umap[:, 1],
            s=500,
            color=[color_key[label] for label in labels_txt],
            alpha=0.2,
        )

    plt.title(title)
    plt.tight_layout()

    if path_save:
        plt.savefig(path_save, dpi=1000)
    else:
        plt.savefig('umap.png')


if __name__ == '__main__':
    path_config = 'configs/draw.yaml'

    with open(path_config, 'r') as f:
        cfgs = yaml.safe_load(f)
        cfgs = argparse.Namespace(**cfgs)

    embeds_seq, embeds_txt, embeds_error = get_embeds(cfgs)
    plot_umap(
        3,
        embeds_seq,
        embeds_txt,
        embeds_error,
        os.path.join(cfgs.dir_work, str(cfgs.idx_fold), 'visualization', 'umap.pdf'),
        cfgs.title,
    )
