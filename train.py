"""Train RhythmMamba on a CSV manifest; test subjects are never evaluated here."""
import argparse
import inspect
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data import PulseDataset, manifest_hash, read_manifest, subject_splits
from engine import run_epoch
from losses import HybridLoss
from model import RhythmMamba
from utils import CHECKPOINT_FORMAT, cuda_device, new_directory, set_seed, write_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", type=Path, help="Optional CSV with subject,split columns")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--embed-dim", type=int, default=96)
    parser.add_argument("--init-weights", type=Path, help="Optional upstream state_dict (.pth); explicit fine-tuning")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive; workers must be nonnegative")
    device = cuda_device(args.device)
    set_seed(args.seed)
    rows = read_manifest(args.manifest)
    splits = subject_splits(rows, args.seed, args.splits)
    train_rows = [r for r in rows if splits[r["subject"]] == "train"]
    valid_rows = [r for r in rows if splits[r["subject"]] == "valid"]
    loaders = [DataLoader(PulseDataset(part), batch_size=args.batch_size,
                          shuffle=shuffle, num_workers=args.workers,
                          pin_memory=True, drop_last=False)
               for part, shuffle in ((train_rows, True), (valid_rows, False))]
    config = dict(depth=args.depth, embed_dim=args.embed_dim, mlp_ratio=2, drop_path_rate=0.1)
    model = RhythmMamba(**config).to(device)
    if args.init_weights:
        options = {"weights_only": True} if "weights_only" in inspect.signature(torch.load).parameters else {}
        state = torch.load(args.init_weights, map_location="cpu", **options)
        state = state.get("state_dict", state.get("model", state))
        state = {k.removeprefix("module.") if hasattr(k, "removeprefix") else
                 k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    out = new_directory(args.out)
    write_csv(out / "splits.csv", [dict(subject=s, split=splits[s]) for s in sorted(splits)])
    print("Subjects:", {p: sum(v == p for v in splits.values()) for p in ("train", "valid", "test")})
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(loaders[0]), pct_start=0.3)
    criterion = HybridLoss()
    best, history = float("inf"), []
    for epoch in range(args.epochs):
        train_loss = run_epoch(model, loaders[0], criterion, device, rows[0]["fps"], optimizer, scheduler)
        valid_loss = run_epoch(model, loaders[1], criterion, device, rows[0]["fps"])
        improved = valid_loss < best
        best = min(best, valid_loss)
        history.append(dict(epoch=epoch + 1, train_loss=train_loss, valid_loss=valid_loss,
                            learning_rate=optimizer.param_groups[0]["lr"]))
        write_csv(out / "history.csv", history)
        checkpoint = dict(format=CHECKPOINT_FORMAT, model=model.state_dict(),
                          model_config=config, epoch=epoch + 1, best_valid_loss=best,
                          optimizer=optimizer.state_dict(),
                          # PyTorch 2.1 includes a bound annealing method in this
                          # dictionary. Keep only serializable scheduler values
                          # so weights_only checkpoint loading remains possible.
                          scheduler={k: v for k, v in scheduler.state_dict().items() if not callable(v)},
                          splits=splits, manifest_sha256=manifest_hash(args.manifest),
                          data_config={k: rows[0][k] for k in ("frames", "height", "width", "fps")},
                          args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                          torch_version=str(torch.__version__))
        torch.save(checkpoint, out / "last.pt")
        if improved:
            torch.save(checkpoint, out / "best.pt")
        print(f"Epoch {epoch + 1}: train={train_loss:.6f}, valid={valid_loss:.6f}, best={best:.6f}")
    print("Best validation checkpoint:", out / "best.pt")


if __name__ == "__main__":
    main()
