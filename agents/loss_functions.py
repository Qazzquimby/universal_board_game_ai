import torch
import torch.nn.functional as F


def entropy_adjusted_cross_entropy_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    Calculates cross-entropy loss and subtracts target entropy.
    This makes the loss tend to 0 for a perfect prediction.
    """
    log_probs = F.log_softmax(logits, dim=1)

    # 1. Substitute -inf with 0.0 in log_probs where target is 0 to preven nan
    safe_log_probs = torch.where(targets > 0, log_probs, torch.zeros_like(log_probs))
    cross_entropy_term = targets * safe_log_probs
    cross_entropy = -torch.sum(cross_entropy_term, dim=1).mean()

    with torch.no_grad():
        target_entropy = torch.sum(torch.special.entr(targets), dim=1).mean()

    return cross_entropy - target_entropy
