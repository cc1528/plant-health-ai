import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "splits"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR   = DATA_DIR / "val"
TEST_DIR  = DATA_DIR / "test"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4  # you can set 0 on Windows if you get DataLoader issues


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders():
    # data augmentations for training only
    train_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    # validation / test: no crazy augments
    eval_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=str(TRAIN_DIR), transform=train_tfms)
    val_dataset   = datasets.ImageFolder(root=str(VAL_DIR),   transform=eval_tfms)
    test_dataset  = datasets.ImageFolder(root=str(TEST_DIR),  transform=eval_tfms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    class_names = train_dataset.classes  # ['early_blight','healthy','late_blight'] etc.
    return train_loader, val_loader, test_loader, class_names


def build_model(num_classes, device):
    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze the backbone first (optional, good for fast convergence)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final layer (fc) with our classifier head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    model = model.to(device)
    return model


def evaluate(model, dataloader, device, loss_fn):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() * labels.size(0)

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / total
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def train():
    device = get_device()
    print(f"[INFO] Using device: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    num_classes = len(class_names)
    print(f"[INFO] Classes: {class_names} ({num_classes} classes)")

    model = build_model(num_classes, device)

    # Only parameters in the new head are trainable (fc layer)
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 20)

        # TRAIN LOOP
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in tqdm(train_loader, desc="Training", ncols=80):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(outputs, dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total if running_total > 0 else 0.0

        # VALIDATION
        val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)

        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

        # checkpoint if improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, MODEL_DIR / "model_best.pth")
            print(f"[INFO]  New best model saved with val_acc={best_val_acc:.4f}")

    # after training loop, evaluate on test
    print("\n[INFO] Loading best weights and evaluating on test set...")
    model.load_state_dict(best_model_wts)
    test_loss, test_acc = evaluate(model, test_loader, device, loss_fn)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")

    # save final model state_dict too (optional)
    torch.save(model.state_dict(), MODEL_DIR / "model_final.pth")
    print("[INFO] Training complete. Models stored in models/ folder.")


if __name__ == "__main__":
    train()
