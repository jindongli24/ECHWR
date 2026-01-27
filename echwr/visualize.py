import os

import jiwer
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['visualize']


def draw_sem(
    refs: list[str],
    hyps: list[str],
    cats: list[str],
    path_save: str = 'mat_se.pdf',
) -> None:
    '''Draws and saves a substitution error matrix (SEM).

    The SEM is a variant of a confusion matrix that isolates substitution
    errors in text sequences. To enhance visual clarity, the grayscale map is
    inverted: darker pixels (black) indicate high error rates (1.0), while
    lighter pixels (white) indicate correct predictions (0.0).

    Args:
        refs: List of ground-truth label strings.
        hyps: List of predicted hypothesis strings.
        cats: List of character categories (characters).
        path_save: File path to save the generated PDF plot. Defaults to
            'mat_se.pdf'.
    '''
    out = jiwer.process_characters(refs, hyps)
    confusion = np.zeros((len(cats), len(cats)))
    count = np.zeros((1, len(cats)))

    cnt_event = {'delete': 0, 'equal': 0, 'insert': 0, 'substitute': 0}

    for results, hyp, ref in zip(
        out.alignments, out.hypotheses, out.references
    ):
        for event in results:
            if event.type in ['substitute']:
                for i in range(event.ref_start_idx, event.ref_end_idx):
                    for j in range(event.hyp_start_idx, event.hyp_end_idx):
                        confusion[cats.index(hyp[j])][cats.index(ref[i])] += 1

            cnt_event[event.type] += (
                event.hyp_end_idx - event.hyp_start_idx
                if event.type != 'delete'
                else event.ref_end_idx - event.ref_start_idx
            )

        for char in ref:
            count[0][cats.index(char)] += 1

    plt.figure(figsize=(10, 10), dpi=300)
    plt.figtext(
        0.5,
        0.01,
        ', '.join([f'{k}: {v}' for k, v in cnt_event.items()]),
        ha='center',
    )
    # invert color: 1 (black) is error, 0 (white) is correct
    plt.imshow(1 - (confusion / (count + 1e-9)), cmap='gray')
    plt.xlabel('Reference')
    plt.ylabel('Hypothesis')
    plt.xticks(np.arange(len(cats)), cats)
    plt.yticks(np.arange(len(cats)), cats)
    plt.title('Substitution Error Matrix')
    plt.tight_layout(rect=[0.0, 0.02, 1, 0.98])
    plt.savefig(path_save)
    plt.close()


def visualize(
    preds: str | list[str],
    labels: str | list[str],
    cats: list[str],
    dir_save: str,
    epoch: int,
    use_sem: bool = True,
) -> None:
    '''Orchestrates result visualization and plotting.

    Args:
        preds: Predicted sentence(s). Can be a single string or list.
        labels: Ground-truth sentence(s). Can be a single string or list.
        cats: List of valid character categories.
        dir_save: Directory path to save visualization files.
        epoch: Current epoch number (used for file naming).
        use_sem: Whether to generate the substitution error matrix. Defaults
            to True.
    '''
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(labels, str):
        labels = [labels]

    if use_sem:
        path_save = os.path.join(dir_save, f'mat_se_{epoch}.pdf')
        draw_sem(labels, preds, cats, path_save)
