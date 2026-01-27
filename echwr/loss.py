import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ECHWRLoss']


class ECHWRLoss(nn.Module):
    '''Error Contrastive Handwriting Recognition Loss.

    This composite loss function combines Connectionist Temporal Classification
    (CTC) loss with probability smoothing, Batch-wise Contrastive (BC) loss,
    and Error-based Contrastive (EC) loss.

    Args:
        alpha_smooth: Smooth factor for input probability smoothing. If 0, use
            original probabilities. Defaults to 1e-6.
        blank: Index of the blank label for CTC. Defaults to 0.
        reduction: Specifies the reduction to apply to the output. Options:
            'none', 'mean', 'sum'. Defaults to 'mean'.
        zero_infinity: Whether to zero infinite losses and associated
            gradients. Defaults to False.


    Attributes:
        alpha_smooth (float): Probability smoothing factor.
        blank (int): Blank label index.
        reduction (str): Reduction mode.
        zero_infinity (bool): Infinite loss handling flag.
    '''

    def __init__(
        self,
        alpha_smooth: float = 1e-6,
        blank: int = 0,
        reduction: str = 'mean',
        zero_infinity: bool = False,
    ) -> None:
        super().__init__()

        self.alpha_smooth = alpha_smooth
        self.blank = blank
        self.reduction = reduction
        self.zero_infinity = zero_infinity

    def forward(
        self,
        preds: dict[str, torch.Tensor],
        targets: torch.Tensor,
        tasks: list[str],
        lens_input: torch.Tensor = None,
        lens_target: torch.Tensor = None,
        scale: torch.Tensor = None,
    ) -> torch.Tensor:
        '''Calculates the sum of active losses.

        Args:
            preds: Dictionary containing model outputs:
                * 'out_hwr': HWR logits (batch, len_seq, num_cls).
                * 'embed_seq': Time-series embeddings (batch, dim_embed).
                * 'embeds_txt': Text embeddings (num_txt, batch, dim_embed).
            targets: Concatenated targets for CTC loss. Shape: (sum(target_lengths)).
            tasks: List of losses to calculate. Options:
                * 'hwr': CTC loss for handwriting recognition.
                * 'bc': Batch-wise contrastive loss (CLIP-style).
                * 'ec': Error-based contrastive loss (Hard negatives).
            lens_input: Lengths of input sequences for CTC. Defaults to None.
            lens_target: Lengths of target sequences for CTC. Defaults to None.
            scale: Learnable temperature scaling factor for contrastive losses.
                Defaults to None.

        Returns:
            The scalar sum of all computed loss values.
        '''
        losses = []

        # ctc loss
        if (
            'hwr' in tasks
            and 'out_hwr' in preds
            and lens_input is not None
            and lens_target is not None
        ):
            # (batch, len_seq, num_cls) -> (len_seq, batch, num_cls)
            probs = preds['out_hwr'].permute((1, 0, 2))

            if self.alpha_smooth:
                probs = self.smooth_probs(probs, self.alpha_smooth)

            probs = probs.log()
            loss_hwr = nn.functional.ctc_loss(
                probs,
                targets,
                lens_input,
                lens_target,
                self.blank,
                self.reduction,
                self.zero_infinity,
            )
            losses.append(loss_hwr)

        # contrastive loss
        if 'embed_seq' in preds and 'embeds_txt' in preds:
            # in-batch contrastive (bc) loss
            if 'bc' in tasks:
                # similarity between time-series seq and gt text (index 0)
                logits_per_seq = (
                    scale * preds['embed_seq'] @ preds['embeds_txt'][0].t()
                )  # [B, B]
                logits_per_txt = logits_per_seq.t()  # [B, B]

                labels_clip = torch.arange(
                    logits_per_seq.size(0), device=preds['embed_seq'].device
                )
                loss_i2t = F.cross_entropy(logits_per_seq, labels_clip)
                loss_t2i = F.cross_entropy(logits_per_txt, labels_clip)
                loss_bc = 0.5 * (loss_i2t + loss_t2i)
                losses.append(loss_bc)

            # error-based contrastive (ec) loss
            if 'ec' in tasks and len(preds['embeds_txt']) > 1:
                # logits shape: [batch, num_txt_candidates]
                logits = torch.einsum(
                    'nbd,bd->bn', preds['embeds_txt'], preds['embed_seq']
                )
                logits *= scale

                # positive is always at index 0
                labels_etc = torch.zeros(
                    logits.size(0),
                    dtype=torch.long,
                    device=preds['embed_seq'].device,
                )
                loss_ec = F.cross_entropy(logits, labels_etc)
                losses.append(loss_ec)

        return sum(losses)

    @staticmethod
    def smooth_probs(probs: torch.Tensor, alpha: float = 1e-6) -> torch.Tensor:
        '''Smooths a probability distribution.

        Mixes the input distribution with a uniform distribution to prevent
        overconfidence.

        Args:
            probs: Probability distribution. Shape: (batch, len_seq, num_cls).
            alpha: Smoothing factor. Defaults to 1e-6.

        Returns:
            Smoothed probability distribution with the same shape as input.
        '''
        num_cls = probs.shape[-1]
        distr_uni = torch.full_like(probs, 1.0 / num_cls)
        probs = (1 - alpha) * probs + alpha * distr_uni

        # ensure probabilities sum to 1
        probs /= probs.sum(dim=-1, keepdim=True)

        return probs
