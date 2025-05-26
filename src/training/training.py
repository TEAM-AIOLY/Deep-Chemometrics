# trainer.py
import torch
import torcheval.metrics
from src.utils import Utils
import numpy as np
import os

class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, config, plot=False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(self.config.device)
        self.model.to(self.device)
        self.best_val_metric = -np.inf
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.plot = plot
        self.best_epoch =None

    def train_one_epoch(self):
        self.model.train()
        running_loss = torch.zeros(1, device=self.device) if self.config.classification else torch.zeros(self.model.out_dims, device=self.device)

        self.model.to(self.device)
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device, non_blocking=True).float()
            targets = targets.to(self.device, non_blocking=True).float()

            self.optimizer.zero_grad()
            outputs = self.model(inputs[:, None])
            loss = self.criterion(outputs, targets)
            loss.mean().backward()
            self.optimizer.step()
            running_loss += torch.mean(loss) * inputs.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        return epoch_loss

    def evaluate(self):
        self.model.eval()
        val_loss = torch.zeros(1, device=self.device) if self.config.classification else torch.zeros(self.model.out_dims, device=self.device)
        out, tar = [], []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device, non_blocking=True).float()
                targets = targets.to(self.device, non_blocking=True).float()
                outputs = self.model(inputs[:, None])

                loss = self.criterion(outputs, targets)
                val_loss += loss.mean() * inputs.size(0)

                out.append(outputs.detach().cpu())
                tar.append(targets.detach().cpu())

        val_loss = val_loss / len(self.val_loader.dataset)
        all_outputs = torch.cat(out, dim=0)
        all_targets = torch.cat(tar, dim=0)

        metrics = self.compute_metrics(all_outputs, all_targets)
        return val_loss, metrics

    def compute_metrics(self, outputs, targets):
        metrics = []
        if not self.config.classification:
            R2 = [torcheval.metrics.R2Score() for _ in range(self.model.out_dims)]
            for i in range(self.model.out_dims):
                R2[i].update(targets[:, i], outputs[:, i])
                metrics.append(R2[i].compute().item())
        else:
            F1 = torcheval.metrics.MulticlassF1Score()
            F1.update(torch.argmax(targets, dim=1), torch.argmax(outputs, dim=1))
            metrics = F1.compute()

        return metrics

    def train(self):

        if self.config.save_path ==None :
            best_model_path = None
            print("No save path specified. Model will not be saved.")
        else:
            best_model_path=self.config.save_path + f'_best.pth'

        for epoch in range(self.config.num_epochs):
            epoch_train_loss = self.train_one_epoch()
            self.train_losses.append(epoch_train_loss.detach().cpu())

            val_loss, metrics = self.evaluate()
            self.val_losses.append(val_loss.detach().cpu())
            self.val_metrics.append(metrics)
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} | "
                  f"Train Loss: {epoch_train_loss[0].detach().cpu().numpy():.4f} | "
                  f"Val Loss: {val_loss[0].detach().cpu().numpy():.4f} | "
                  f"Val Mean Metrics: {np.array(metrics).mean():.4f}")

            if best_model_path is not None  :
                self.best_val_metric,best_epoch = Utils.save_model(model=self.model,path=best_model_path,epoch =  epoch,
                                                        best_metric=self.best_val_metric,
                                                        current_metric=metrics,
                                                        classification= self.config.classification)


        if self.plot ==True:
           Utils.plot_losses(self.train_losses, self.val_losses, self.val_metrics, self.config.classification,self.config.max_loss_plot)

        return self.train_losses, self.val_losses, self.val_metrics, best_model_path,best_epoch
