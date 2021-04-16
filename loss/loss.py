import torch


def batch_dice_loss(true_val, pred_val, epsilon=1e-8):
    """
    Dice coefficient loss
    ---
    Equivalent to F1 score. Often used when there is
    imbalance in dataset (non-crack pixels outnumber
    crack pixels by 65:1).

    Args:
        true_val: a tensor of shape [N, 1, H, W]
        predicted_val: a tensor of shape [N, 1, H, W]
    Returns:
        dice_loss: the Dice loss.
    """
    # Flattened from [N, 1, H, W] to [N, H*W]
    true_val = true_val.flatten(start_dim=1)
    pred_val = pred_val.flatten(start_dim=1)

    numerator = 2. * (pred_val * true_val).sum(dim=1)
    denominator = (pred_val).sum(dim=1) + (true_val).sum(dim=1)

    return torch.mean(1 - ((numerator + epsilon) / (denominator + epsilon)))