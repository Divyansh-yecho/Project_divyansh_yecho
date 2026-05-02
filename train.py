import torch
import torch.nn as nn
import torch.optim as optim

from config import learning_rate, scheduler_patience, scheduler_factor, device, CHECKPOINT_PATH


def run_epoch(model, loader, optimizer, criterion, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            if training:
                optimizer.zero_grad()

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def train_brain_tumor_model(model, num_epochs, train_loader, loss_fn, optimizer, val_loader=None):
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=scheduler_patience, factor=scheduler_factor
    )

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"Training on {device} for {num_epochs} epochs\n")

    for epoch in range(num_epochs):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, loss_fn, training=True)

        val_loss, val_acc = 0.0, 0.0
        if val_loader is not None:
            val_loss, val_acc = run_epoch(model, val_loader, optimizer, loss_fn, training=False)
            scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        saved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            saved = "  [saved]"

        print(
            f"Epoch {epoch+1:02d}/{num_epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  "
            f"lr={current_lr:.2e}{saved}"
        )

    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }
    return model, history


def build_optimizer_and_criterion(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    return criterion, optimizer
