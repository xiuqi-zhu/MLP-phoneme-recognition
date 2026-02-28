"""
Entry script: build data, model, train and test.
Usage:
  python -m scripts.main --root /path/to/data [--epochs 30] [--no-train] [--submit]
"""
import argparse
import gc
import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scripts.config import PHONEMES, get_config
from scripts.dataset import AudioDataset, AudioTestDataset
from scripts.model import Network
from scripts.train import train, eval as run_eval
from scripts.test import test


def main():
    parser = argparse.ArgumentParser(description="HW1 P2 Frame-Level Speech Recognition")
    parser.add_argument("--root", type=str, default="/content/11785-f25-hw1p2", help="Root directory of dataset")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config batch_size")
    parser.add_argument("--no-train", action="store_true", help="Skip training, run test/submit only")
    parser.add_argument("--submit", action="store_true", help="Write submission.csv after test")
    parser.add_argument("--out-csv", type=str, default="./submission.csv", help="Output path for submission CSV")
    args = parser.parse_args()

    config = get_config()
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Datasets and DataLoaders
    train_data = AudioDataset(root=args.root, context=config["context"], partition="train-clean-100", config=config)
    val_data = AudioDataset(root=args.root, context=config["context"], partition="dev-clean", config=config)
    test_data = AudioTestDataset(root=args.root, context=config["context"], partition="test-clean")

    train_loader = torch.utils.data.DataLoader(
        train_data,
        num_workers=4,
        batch_size=config["batch_size"],
        pin_memory=True,
        shuffle=True,
        collate_fn=train_data.collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        num_workers=0,
        batch_size=config["batch_size"],
        pin_memory=True,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        num_workers=0,
        batch_size=config["batch_size"],
        pin_memory=True,
        shuffle=False,
    )

    input_size = (2 * config["context"] + 1) * 28
    num_classes = len(train_data.phonemes)

    model = Network(input_size, num_classes, config=config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params < 20_000_000, "Exceeds 20M params."
    print(f"Model parameters: {n_params}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=True) if device.type == "cuda" else None

    if not args.no_train:
        torch.cuda.empty_cache()
        gc.collect()

        for epoch in range(config["epochs"]):
            print(f"\nEpoch {epoch + 1}/{config['epochs']}")
            curr_lr = float(optimizer.param_groups[0]["lr"])
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, scaler=scaler)
            val_loss, val_acc = run_eval(model, val_loader, criterion, device)
            print(f"\tTrain Acc {train_acc*100:.04f}%\tTrain Loss {train_loss:.04f}\t LR {curr_lr:.07f}")
            print(f"\tVal Acc {val_acc*100:.04f}%\tVal Loss {val_loss:.04f}")
            scheduler.step(val_loss)

    if args.submit or args.no_train:
        predictions = test(model, test_loader, PHONEMES, device, output_path=args.out_csv)
        print(f"Generated {len(predictions)} predictions.")


if __name__ == "__main__":
    main()
