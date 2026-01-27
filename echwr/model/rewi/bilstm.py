import torch
import torch.nn as nn

__all__ = ['BiLSTM']


class BiLSTM(nn.Module):
    '''Bidirectional LSTM module for sequence classification.

    This module processes a sequence of features using a Bi-LSTM, projects the
    hidden states to the target class dimension, and applies Softmax to yield
    probabilities.

    Args:
        size_in: Number of input channels (feature dimension).
        num_cls: Number of output categories (classes).
        hidden_size: Hidden state dimension of the LSTM. Defaults to 128.
        num_layers: Number of stacked LSTM layers. Defaults to 3.
        r_drop: Dropout probability for the LSTM layers. Defaults to 0.2.

    Attributes:
        lstm (nn.LSTM): The bidirectional LSTM backbone.
        fc (nn.Linear): Fully connected layer mapping from (2 * hidden_size)
            to num_cls.
        softmax (nn.Softmax): Activation layer to convert logits to
            probabilities.
    '''

    def __init__(
        self,
        size_in: int,
        num_cls: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        r_drop: float = 0.2,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            size_in,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=r_drop,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, num_cls)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Performs the forward pass.

        Args:
            x: Input tensor of shape (batch_size, len_seq, size_in).

        Returns:
            Output tensor of probabilities with shape (batch_size, len_seq,
            num_cls).

            Note:
                Since this output has gone through Softmax, use `nn.NLLLoss`
                for training. Do not use `nn.CrossEntropyLoss` (which expects
                raw logits).
        '''
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = self.softmax(x)

        return x
