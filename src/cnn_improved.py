"""
Train and evaluate a stronger CNN baseline on the Stanford Cars splits used by this project.

This version keeps the original `src/cnn.py` baseline intact and adds:
- a larger pretrained backbone (ResNet-34)
- stronger data augmentation
- end-to-end fine-tuning by default
- AdamW + cosine LR decay
- label smoothing

Usage examples:
    python src/cnn_improved.py --epochs 10 --batch-size 32
    python src/cnn_improved.py --epochs 12 --lr 0.0003 --eval-test
    python src/cnn_improved.py --eval-test-only
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from progress_utils import progress_bar

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ENTRYPOINT_DIR = ROOT / "entrypoint"

sys.path.insert(0, str(ENTRYPOINT_DIR))
from load import test_ds, train_ds, val_ds  # noqa: E402

NUM_CLASSES = 196
MODEL_NAME = "resnet34"
MODEL_PATH = DATA_DIR / "best_cnn_improved.pt"
HISTORY_PATH = DATA_DIR / "cnn_improved_history.json"
RESULTS_PATH = DATA_DIR / "cnn_improved_results.json"
TEST_PREDS_PATH = DATA_DIR / "cnn_improved_test_predictions.npz"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CarsDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = sample["image"].convert("RGB")
        label = int(sample["label"])
        return self.transform(image), label


def get_transforms(image_size):
    resize_size = int(round(image_size * 1.14))
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, eval_transform


def get_loaders(image_size, batch_size, num_workers):
    train_transform, eval_transform = get_transforms(image_size)
    train_loader = DataLoader(
        CarsDataset(train_ds, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        CarsDataset(val_ds, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        CarsDataset(test_ds, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


def build_model(num_classes, pretrained, freeze_backbone):
    weights = models.ResNet34_Weights.DEFAULT if pretrained else None
    model = models.resnet34(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def topk_correct(logits, labels, ks=(1, 5)):
    max_k = max(ks)
    _, preds = logits.topk(max_k, dim=1)
    preds = preds.t()
    correct = preds.eq(labels.view(1, -1).expand_as(preds))
    return [correct[:k].reshape(-1).float().sum().item() for k in ks]


def run_epoch(model, loader, criterion, device, optimizer=None, desc=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_examples = 0
    top1_correct = 0.0
    top5_correct = 0.0

    iterator = progress_bar(loader, desc=desc, total=len(loader), leave=False)
    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        top1, top5 = topk_correct(logits, labels)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        top1_correct += top1
        top5_correct += top5

        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                loss=f"{total_loss / total_examples:.4f}",
                top1=f"{top1_correct / total_examples:.4f}",
            )

    return {
        "loss": total_loss / total_examples,
        "top1": top1_correct / total_examples,
        "top5": top5_correct / total_examples,
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_top5 = []

    for images, labels in progress_bar(loader, desc="Predict", total=len(loader), leave=False):
        images = images.to(device)
        logits = model(images)
        top5 = logits.topk(5, dim=1).indices.cpu().numpy()

        all_labels.append(labels.numpy())
        all_preds.append(top5[:, 0])
        all_top5.append(top5)

    return {
        "y_true": np.concatenate(all_labels),
        "y_pred": np.concatenate(all_preds),
        "top5_pred": np.concatenate(all_top5),
    }


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_checkpoint(model, device):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run cnn_improved.py first.")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def train(args, device):
    train_loader, val_loader, test_loader = get_loaders(
        args.image_size,
        args.batch_size,
        args.num_workers,
    )
    model = build_model(NUM_CLASSES, args.pretrained, args.freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_top1 = -1.0
    history = []

    epoch_iter = progress_bar(range(1, args.epochs + 1), desc="Epochs")

    for epoch in epoch_iter:
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, desc=f"Train {epoch}/{args.epochs}")
        val_metrics = run_epoch(model, val_loader, criterion, device, desc=f"Val {epoch}/{args.epochs}")
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
        }
        history.append(record)
        save_json(HISTORY_PATH, history)

        print(
            f"Epoch {epoch}/{args.epochs} | lr {current_lr:.6f} | "
            f"train loss {train_metrics['loss']:.4f}, top1 {train_metrics['top1']:.4f}, top5 {train_metrics['top5']:.4f} | "
            f"val loss {val_metrics['loss']:.4f}, top1 {val_metrics['top1']:.4f}, top5 {val_metrics['top5']:.4f}"
        )

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": MODEL_NAME,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                MODEL_PATH,
            )
            print(f"Saved best model to {MODEL_PATH}")

        if hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(val_top1=f"{val_metrics['top1']:.4f}", best=f"{best_val_top1:.4f}")

        scheduler.step()

    results = {
        "model_name": MODEL_NAME,
        "best_val_top1": best_val_top1,
        "best_checkpoint": str(MODEL_PATH),
        "history": str(HISTORY_PATH),
    }

    if args.eval_test:
        results["test"] = evaluate_test(args, device, model=model, test_loader=test_loader)

    save_json(RESULTS_PATH, results)


def evaluate_test(args, device, model=None, test_loader=None):
    if test_loader is None:
        _, _, test_loader = get_loaders(args.image_size, args.batch_size, args.num_workers)
    if model is None:
        model = build_model(NUM_CLASSES, args.pretrained, args.freeze_backbone).to(device)
    checkpoint = load_checkpoint(model, device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    test_metrics = run_epoch(model, test_loader, criterion, device, desc="Test")
    preds = predict(model, test_loader, device)

    TEST_PREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        TEST_PREDS_PATH,
        y_true=preds["y_true"],
        y_pred=preds["y_pred"],
        top5_pred=preds["top5_pred"],
    )

    print(
        f"Test | loss {test_metrics['loss']:.4f}, "
        f"top1 {test_metrics['top1']:.4f}, top5 {test_metrics['top5']:.4f}"
    )
    print(f"Saved test predictions to {TEST_PREDS_PATH}")

    return {
        "checkpoint_epoch": checkpoint["epoch"],
        "loss": test_metrics["loss"],
        "top1": test_metrics["top1"],
        "top5": test_metrics["top5"],
        "predictions": str(TEST_PREDS_PATH),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate an upgraded CNN on Stanford Cars.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--eval-test-only", action="store_true")
    parser.set_defaults(pretrained=True, freeze_backbone=False)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(
        f"Model: {MODEL_NAME}; pretrained: {args.pretrained}; "
        f"freeze_backbone: {args.freeze_backbone}"
    )

    if args.eval_test_only:
        results = {"model_name": MODEL_NAME, "test": evaluate_test(args, device)}
        save_json(RESULTS_PATH, results)
    else:
        train(args, device)


if __name__ == "__main__":
    main()
