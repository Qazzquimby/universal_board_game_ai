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
    return F.kl_div(log_probs, targets, reduction="batchmean")

    # log_probs = F.log_softmax(
    #     logits, dim=1
    # )  # missing .exp()? # should be using kl_div?
    # cross_entropy_term = targets * log_probs
    # cross_entropy_term = torch.nan_to_num(cross_entropy_term, nan=0.0)
    # cross_entropy = -torch.sum(cross_entropy_term, dim=1).mean()
    #
    # eps = 1e-9
    # with torch.no_grad():
    #     target_entropy_term = targets * torch.log(targets + eps)
    #     target_entropy_term = torch.nan_to_num(target_entropy_term, nan=0.0)
    #     target_entropy = -torch.sum(target_entropy_term, dim=1).mean()
    #
    # return cross_entropy - target_entropy
