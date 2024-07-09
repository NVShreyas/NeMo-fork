# LM loss using megatron core

import torch

try:
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
    from megatron.core import parallel_state, tensor_parallel

    HAVE_MEGATRON_CORE=True

except (ImportError, ModuleNotFoundError):
    
    HAVE_MEGATRON_CORE=False


def language_model_loss(labels: torch.Tensor, 
                        logits: torch.Tensor,
                        cross_entropy_loss_fusion=False) -> torch.Tensor:
    """Computes the language model loss (Cross entropy across vocabulary)

    Args:
        labels (Tensor): The labels of dimension [batch size, seq length]
        logits (Tensor): The final logits returned by the output layer of the transformer model
        cross_entropy_loss_fusion (bool): Whether to enable cross entropy fusion when logits are split 
                                        across TP ranks.
    Returns:
        Tensor: Loss tensor of dimensions [batch size, sequence_length]
    """
    # [b s] => [s b]
    labels = labels.transpose(0, 1).contiguous()
    if cross_entropy_loss_fusion:
        loss = fused_vocab_parallel_cross_entropy(logits, labels)
    else:
        loss = tensor_parallel.vocab_parallel_cross_entropy(logits, labels)

    # [s b] => [b, s]
    loss = loss.transpose(0, 1).contiguous()
    return loss