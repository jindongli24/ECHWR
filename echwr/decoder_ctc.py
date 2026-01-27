import torch

__all__ = ['BestPath']


class BestPath:
    '''Greedy CTC decoder for converting model outputs to sequences.

    This class performs greedy decoding: selecting the label with the highest
    probability at each time step and then merging adjacent repeated tokens.

    Note:
        This implementation assumes that the blank/separator token is handled
        implicitly by the `categories` mapping (e.g., if index 0 is blank,
        `categories[0]` should be an empty string) or that explicit blank
        removal is not required for this specific use case.

    Args:
        categories: List of characters/tokens mapping indices to symbols.

    Attributes:
        categories (list[str]): The vocabulary mapping.
    '''

    def __init__(self, categories: list[str]) -> None:
        self.categories = categories

    def decode(self, seq: torch.Tensor, label: bool = False) -> str:
        '''Decodes the input sequence into a string.

        If `label` is False, it performs argmax and removes consecutive
        repetitive values (CTC collapse). If `label` is True, it treats the
        input as a target sequence and maps indices directly.

        Args:
            seq: Model outputs or labels.
                Shape for prediction: (length_sequence, number_categories).
                Shape for label: (length_sequence).
            label: Whether the input sequence is a label sequence (skips
                argmax and collapse steps). Defaults to False.

        Returns:
            The decoded sentence string.
        '''
        if not label:
            seq = torch.argmax(seq, dim=-1)
            seq = torch.unique_consecutive(seq, dim=-1)

        # map indices to characters
        seq = [self.categories[i] for i in seq]
        seq = ''.join(seq)

        return seq
