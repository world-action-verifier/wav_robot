"""Sample-weighted microbatch accumulation; caller clips/steps once per batch."""
import torch


def backward_microbatches(count, microbatch, compute_mean_loss, scaler):
    if count < 1 or microbatch < 1:
        raise ValueError('Batch sizes must be positive')
    total, chunks = 0.0, 0
    for start in range(0, count, microbatch):
        end = min(start + microbatch, count)
        loss = compute_mean_loss(start, end)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError('Microbatch loss must be a finite scalar')
        weight = (end-start)/count
        scaler.scale(loss * weight).backward()
        total += float(loss.detach()) * weight
        chunks += 1
    return total, chunks
