import json
import os

import numpy as np
import torch
from loguru import logger
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from .transforms import AddNoise, Drift, Dropout, TimeWarp

__all__ = ['fn_collate', 'ECHWDataset']


def fn_collate(
    batch: list[tuple[list[torch.Tensor], list[torch.Tensor], int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Collate function for aligning the shape of data sequences and labels.

    Aligns dynamic batch sizes and removes samples with duplicated labels
    within the batch.

    Args:
        batch: Input batch data including pre-processed time-series
            sequences, original and corrupted texts, lengths of sequences, and
            lengths of text.

    Returns:
        Aligned batch data.
        - seqs (torch.Tensor): Padded time-series sequences.
            Shape: (size_batch, num_chan, len_seq).
        - txts (torch.Tensor): Original text labels.
            Shape: (num_txt, size_batch, len_txt).
        - lens_seq (torch.Tensor): Lengths of sequences.
        - lens_txt (torch.Tensor): Lengths of text.
    '''
    seqs, txts, lens_seq, lens_txt = [], [], [], []
    txts_seen = set()

    # remove samples with duplicate labels
    for seq, txt, len_seq, len_txt in batch:
        if not tuple(txt[0]) in txts_seen:
            txts_seen.add(tuple(txt[0]))
            seqs.append(seq)
            txts.append(txt)
            lens_seq.append(len_seq)
            lens_txt.append(len_txt)

    # [size_batch, len_seq, num_chan] -> [size_batch, num_chan, len_seq]
    seqs = pad_sequence(seqs, True).permute(0, 2, 1)

    # [size_batch, num_txt, len_txt] -> [num_txt, size_batch, len_txt]
    txts = torch.stack(txts).permute(1, 0, 2)

    lens_seq = torch.tensor(lens_seq)
    lens_txt = torch.tensor(lens_txt)

    return seqs, txts, lens_seq, lens_txt


class ECHWDataset(Dataset):
    '''Dataset for handwriting recognition.

    Handles loading, caching, augmenting, and normalizing time-series
    handwriting data.

    Args:
        path_anno: Path to the annotation file of the dataset.
        idx_fold: Fold index for cross validation.
        categories: List of categories (characters).
        ratio_ds: Downsampling ratio of the model. Defaults to 8.
        aug: Whether to augment data. Defaults to False.
        len_context: Length of the text tensor (max context). Defaults to 64.
        len_seq: Fixed length of the input sequence (0 means dynamic).
            Defaults to 0.
        num_corr_txt: Number of corrupted text sets to generate per sample.
            Defaults to 0.
        cache: Whether to cache the data in RAM to speed up access.
            Defaults to False.

    Attributes:
        dir_ds (str): Directory containing dataset files.
        categories (list[str]): Character list.
        ratio_ds (int): Downsampling ratio.
        len_seq (int): Configured sequence length.
        len_context (int): Configured context length.
        num_corr_txt (int): Number of corrupted texts.
        cache (bool): Caching flag.
        augs (list | None): List of augmentation transforms.
        annos (list): Loaded annotations for the current fold.
        candidates (list[str]): Valid characters for noise generation.
        data_cache (list): In-memory cache of loaded data (if enabled).
    '''

    def __init__(
        self,
        path_anno: str,
        idx_fold: str | int,
        categories: list[str],
        ratio_ds: int = 8,
        aug: bool = False,
        len_context: int = 64,
        len_seq: int = 0,
        num_corr_txt: int = 0,
        cache: bool = False,
    ) -> None:
        self.dir_ds = os.path.dirname(path_anno)
        self.categories = categories
        self.ratio_ds = ratio_ds
        self.len_seq = len_seq
        self.len_context = len_context
        self.num_corr_txt = num_corr_txt
        self.cache = cache

        self.augs = (
            [
                AddNoise(scale=0.05, kind='multiplicative'),
                Drift(0.1, 40, 'multiplicative'),
                Dropout(size=(5, 10), per_channel=True),
                TimeWarp(5, 4),
            ]
            if aug
            else None
        )

        with open(path_anno, 'r') as f:
            annos = json.load(f)
            self.annos = annos['annotations'][str(idx_fold)]
            self.candidates = [
                char for char in annos['categories'] if char != ' '
            ]

        if self.cache:
            self.data_cache = [
                [
                    np.loadtxt(
                        os.path.join(self.dir_ds, anno['filename']),
                        delimiter=';',
                        dtype=np.float32,
                    ),
                    anno['label'],
                ]
                for anno in tqdm(self.annos)
            ]
            logger.info(f'Cached dataset {path_anno}')

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        '''Get item according to index number.

        Args:
            idx: Index of data sequence.

        Returns:
            Tuple containing:
            - seq (torch.Tensor): Processed time-series sequence.
            - txts (torch.Tensor): Stack of original and corrupted text labels.
            - len_seq (int): Length of the sequence.
            - len_txt (int): Length of the original text.
        '''
        # load data
        if self.cache:
            seq, txt = self.data_cache[idx]
        else:
            anno = self.annos[idx]
            seq = np.loadtxt(
                os.path.join(self.dir_ds, anno['filename']),
                delimiter=';',
                dtype=np.float32,
            )
            txt = anno['label']

        # text pre-processing
        if self.num_corr_txt:
            txts = [txt] + self.get_txt_corrupted(
                txt, self.candidates, self.num_corr_txt
            )
        else:
            txts = [txt]

        for i in range(len(txts)):
            txts[i] = [self.categories.index(char) for char in txts[i]]

            if i == 0:
                len_txt = len(txts[0])

            txts[i] = torch.tensor(txts[i], dtype=torch.int32)
            txts[i] = pad(txts[i], (0, self.len_context - len(txts[i])))
            txts[i] = txts[i][: self.len_context]

        txts = torch.stack(txts)

        # sequence pre-processing
        if self.augs is not None:
            for aug in self.augs:
                if np.random.random() < 0.25:
                    seq = aug(seq)

        # normalize
        seq = (seq - np.mean(seq, 0)) / (np.std(seq, 0) + 1e-6)
        seq = torch.from_numpy(seq).to(torch.float32)

        # padding
        if self.len_seq and len(seq) < self.len_seq:
            seq = pad(seq.T, (0, self.len_seq - len(seq))).T

        # make sure the sequence lengths are compatible to the CTC loss
        if len(seq) < (len_pad := len(txt) * 2 * self.ratio_ds):
            seq = pad(seq.T, (0, len_pad - len(seq))).T

        return seq, txts, len(seq), len_txt

    def __len__(self) -> int:
        '''Get number of data sequences in the dataset.

        Returns:
            Number of data sequences.
        '''
        return len(self.annos)

    @staticmethod
    def get_txt_corrupted(
        sentence: str, candidates: list[str], num_set: int = 1
    ) -> list[str]:
        '''Introduce character-level errors per word.

        Generates three error types: insertion, deletion, and substitution.

        Args:
            sentence: The original sentence.
            candidates: List of strings to use for insertion and substitution.
            num_set: Number of corrupted sentence sets to generate. Each set
                has three corrupted texts of different types. Defaults to 1.

        Returns:
            List of sentences with insertion, deletion, and substitution errors.
        '''
        words_orig = sentence.split()
        sents_out = []

        for _ in range(num_set):
            # insertion errors
            words_ins = []

            for word in words_orig:
                pos = np.random.randint(0, len(word) + 1)
                char = candidates[np.random.randint(0, len(candidates))]
                words_ins.append(word[:pos] + char + word[pos:])

            sents_out.append(' '.join(words_ins))

            # deletion errors
            words_del = []

            for word in words_orig:
                pos = np.random.randint(0, len(word))
                words_del.append(word[:pos] + word[pos + 1 :])

            sents_out.append(' '.join(words_del))

            # substitution errors
            words_sub = []

            for word in words_orig:
                pos = np.random.randint(0, len(word))
                eligible = [c for c in candidates if c != word[pos]]
                char = eligible[np.random.randint(0, len(eligible))]
                words_sub.append(word[:pos] + char + word[pos + 1 :])

            sents_out.append(' '.join(words_sub))

        return sents_out
