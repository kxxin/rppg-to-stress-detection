"""A single training/validation epoch, separate from command-line orchestration."""
import torch
from tqdm import tqdm


def run_epoch(model, loader, criterion, device, fs, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total, samples = 0.0, 0
    with torch.set_grad_enabled(training):
        progress = tqdm(loader, desc="Train" if training else "Validate")
        for batch in progress:
            video = batch["video"].to(device)
            target = batch["target"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(video)
            loss = criterion(prediction, target, fs)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; inspect this batch before continuing")
            if training:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient norm")
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            count = len(video)
            total += float(loss.detach()) * count
            samples += count
            progress.set_postfix(loss=total / samples)
    if not samples:
        raise ValueError("No batches in epoch")
    return total / samples
