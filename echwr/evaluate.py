import jiwer
import Levenshtein
import numpy as np

__all__ = ['evaluate']


def get_levenshtein_distance(
    preds: list[str], labels: list[str]
) -> tuple[float, float]:
    '''Calculate the average Levenshtein distance and label length.

    Args:
        preds: List of predicted strings.
        labels: List of ground-truth strings.

    Returns:
        Tuple containing:
        - Mean Levenshtein distance (float).
        - Mean length of ground-truth labels (float).
    '''
    dist_leven = []
    len_label_avg = []

    for pred, label in zip(preds, labels):
        dist = Levenshtein.distance(pred, label)
        dist_leven.append(dist)
        len_label_avg.append(len(label))

    dist_leven = np.mean(dist_leven)
    len_label_avg = np.mean(len_label_avg)

    return dist_leven, len_label_avg


def evaluate(
    preds: str | list[str],
    labels: str | list[str],
    use_ld: bool = True,
    use_cer: bool = True,
    use_wer: bool = True,
) -> dict:
    '''Evaluates prediction accuracy using standard metrics.

    Args:
        preds: Predicted sentence(s).
        labels: Ground-truth sentence(s).
        use_ld: Whether to calculate Levenshtein distance and average length.
            Defaults to True.
        use_cer: Whether to calculate Character Error Rate (CER). Defaults to
            True.
        use_wer: Whether to calculate Word Error Rate (WER). Defaults to True.

    Returns:
        Dictionary containing evaluation results. Keys are:
        'levenshtein_distance', 'average_sentence_length',
        'character_error_rate', 'word_error_rate'. Skipped metrics return -1.
    '''
    if isinstance(preds, str):
        preds = [preds]

    if isinstance(labels, str):
        labels = [labels]

    if use_ld:
        dist_leven, len_sent_avg = get_levenshtein_distance(preds, labels)
    else:
        dist_leven, len_sent_avg = -1, -1

    cer = jiwer.cer(labels, preds) if use_cer else -1
    wer = jiwer.wer(labels, preds) if use_wer else -1

    return {
        'levenshtein_distance': dist_leven,
        'average_sentence_length': len_sent_avg,
        'character_error_rate': cer,
        'word_error_rate': wer,
    }