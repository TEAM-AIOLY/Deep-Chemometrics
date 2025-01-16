import torch
import torcheval.metrics
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.python.keras.saving.saved_model.load import metrics

def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, save_path=None, save_interval=10, early_stop=True, classification=False, scheduler=None):
    import os
    import numpy as np
    import torch
    import torcheval.metrics
    import matplotlib.pyplot as plt

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        best_model_path = save_path + f'_best.pth'

    train_losses = []
    val_losses = []
    val_metrics = []

    best_val_metric = np.inf if classification else -np.inf

    for epoch in range(num_epochs):
        model.train()
        running_loss = torch.zeros(1, device=device) if classification else torch.zeros(model.out_dims, device=device)

        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).float()
            optimizer.zero_grad()
            outputs = model(inputs[:, None])
            loss = criterion(outputs, targets)
            torch.mean(loss).backward()
            optimizer.step()
            running_loss += torch.mean(loss) * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        if scheduler:
            scheduler.step(epoch_loss)
        train_losses.append(epoch_loss.detach().cpu())

        # Validation loop
        model.eval()
        val_loss = torch.zeros(1, device=device) if classification else torch.zeros(model.out_dims, device=device)
        out = []
        tar = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True).float()
                targets = targets.to(device, non_blocking=True).float()
                outputs = model(inputs[:, None])

                loss = criterion(outputs, targets)
                val_loss += loss.mean() * inputs.size(0)

                out.append(outputs.detach().cpu())
                tar.append(targets.detach().cpu())

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss.detach().cpu())

        all_outputs = torch.cat(out, dim=0)
        all_targets = torch.cat(tar, dim=0)
        metrics = []

        if not classification:
            R2 = [torcheval.metrics.R2Score() for _ in range(model.out_dims)]
            for i in range(model.out_dims):
                R2[i].update(all_targets[:, i], all_outputs[:, i])
                metrics.append(R2[i].compute().item())
            val_metrics.append(metrics)
        else:
            F1 = torcheval.metrics.MulticlassF1Score()
            F1.update(torch.argmax(all_targets, dim=1), torch.argmax(all_outputs, dim=1))
            metrics = F1.compute()
            val_metrics.append(metrics)

        train_loss_str = ', '.join([f'y {i}: {loss:.4f}' for i, loss in enumerate(epoch_loss)])
        val_loss_str = ', '.join([f'y {i}: {loss:.4f}' for i, loss in enumerate(val_loss)])
        metric_str = ', '.join([f'y {i}: {score:.4f}' for i, score in enumerate(metrics)]) if not classification else f'F1 Score: {metrics:.4f}'

        print(f'Epoch {epoch + 1}/{num_epochs} | Train Losses: {train_loss_str} | Validation Losses: {val_loss_str} | Metrics: {metric_str}')

        # Save the best model based on validation metric
        current_val_metric = np.mean(metrics) if not classification else val_loss.mean()
        if save_path and ((classification and current_val_metric < best_val_metric) or (not classification and current_val_metric > best_val_metric)):
            best_val_metric = current_val_metric
            torch.save(model.state_dict(), best_model_path)
            print(f'Model saved at epoch {epoch + 1} to {best_model_path}')

    # Plot metrics
    train_losses_np = [loss.numpy() for loss in train_losses]
    val_losses_np = [loss.numpy() for loss in val_losses]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(train_losses_np, label='Training Loss', color='tab:blue')
    ax1.plot(val_losses_np, label='Validation Loss', color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Metrics', color='tab:green')
    if not classification:
        for i in range(len(val_metrics[0])):
            metric_scores = [scores[i] for scores in val_metrics]
            ax2.plot(metric_scores, label=f'R2 Score y{i}', linestyle='--')
    else:
        ax2.plot(val_metrics, label='F1 Score', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.legend(loc='upper right')

    plt.title('Training and Validation Metrics per Epoch')
    fig.tight_layout()
    plt.show(block=False)

    if save_path:
        return train_losses, val_losses, val_metrics, best_model_path
    else:
        return train_losses, val_losses, val_metrics
