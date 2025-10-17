import torch
import numpy as np

def perplexity_batch_UNSTABLE(bat_likelihoods:torch.Tensor, eos_poss:torch.Tensor, eps=1e-10):
    """
    - Args:
        bat_likelihoods: (batch_size, seq_len) likelihood for each position at each example
        eos_poss: (batch_size) position indices(int) where <EOS> located at each sentence 
    """
    _bli = bat_likelihoods.detach().cpu()
    bs, slen = _bli.shape

    normed_seq_likes = torch.zeros(bs)
    # seems like easier if we just perform row-wise loop
    # instead of vectoring the operations
    for i, row in enumerate(_bli):
        n_used = eos_poss[i].item() + 1 # (position + 1) to get count
        normed_seq_likes[i] = torch.pow(torch.prod(row[:n_used]), 1/n_used)
    bat_perpl = 1/(normed_seq_likes+eps)
    return bat_perpl

def perplexity_batch(bat_likelihoods:torch.Tensor, eos_poss:torch.Tensor):
    """
    Compute per-sequence perplexity for a batch of predicted token likelihoods.

    Perplexity is computed up to the <EOS> position for each sequence individually.
    This function assumes that all tokens beyond <EOS> are to be ignored.

    :param bat_likelihoods: Per-token likelihoods of shape (batch_size, seq_len).
                            Each value should be the model's probability of the true token.
    :param eos_poss: Index of <EOS> token in each sequence (0-based).
                     Shape is (batch_size,).
    :return: Per-sequence perplexity values for each batch example.
    """
    _bli = bat_likelihoods.detach().cpu()
    bs, slen = _bli.shape

    bat_perpl = torch.zeros(bs)
    # seems like easier if we just perform row-wise loop
    # instead of vectoring the operations
    for i, row in enumerate(_bli):
        n_used = eos_poss[i].item() + 1 # (position + 1) to get count
        sum_nll = (-torch.log(row[:n_used])).sum()
        bat_perpl[i] = torch.exp(sum_nll/n_used)
    return bat_perpl
