import torch
import torcheval.metrics
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
            
                     
###############################################################################
def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, save_path=None,classification = False, epoch_save_step =None, scheduler=None):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        best_model_path = save_path + f'_best.pth'
        
        
    train_losses = []
    val_losses = []
    val_metrics = []
    
    best_val_metric = np.inf if classification else -np.inf
    best_epoch=-1
    
    if save_path:
        with open(save_path + "_telemetry.txt", "a") as myfile:
            dt = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
            myfile.write("\n\n--------------------------------------------------------------\nTraining "
                            + dt
                            + "\n--------------------------------------------------------------\n")

    for epoch in range(num_epochs):
        model.train()
        running_loss = torch.zeros(1, device=device) if classification else torch.zeros(model.out_dims, device=device)
             
        for inputs, targets in train_loader:
            
            inputs = inputs.to(device,non_blocking=True).float()
            targets = targets.to(device).float() 
            optimizer.zero_grad()  # Zero the parameter gradients
            outputs = model(inputs[:,None])  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            torch.mean(loss).backward()  # Backward pass
            optimizer.step()  # Update weights
        
            running_loss += loss.mean(dim=0) * inputs.size(0)
            
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
               
                inputs = inputs.to(device,non_blocking=True).float()
                targets = targets.to(device,non_blocking=True).float()
                outputs = model(inputs[:,None])
                           
                loss = criterion(outputs, targets) 
                val_loss += loss.mean(dim=0) * inputs.size(0)
                
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
            
        else :
            F1 = torcheval.metrics.MulticlassF1Score()
            F1.update(torch.argmax(all_targets, dim=1), torch.argmax(all_outputs, dim=1))
            metrics = F1.compute()
            val_metrics.append(metrics)

        
        train_loss_str = ', '.join([f'y {i}: {loss:.4f}' for i, loss in enumerate(epoch_loss)])
        val_loss_str = ', '.join([f'y {i}: {loss:.4f}' for i, loss in enumerate(val_loss)])
        metric_str = ', '.join([f'R2 {i}: {score:.4f}' for i, score in enumerate(metrics)]) if not classification else f'F1 Score: {metrics:.4f}'
        
        msg = f'Epoch {epoch + 1}/{num_epochs} | Train Losses: {train_loss_str} | Validation Losses: {val_loss_str} | {metric_str}\n'
        print(msg)
        with open(save_path + "_telemetry.txt", "a") as myfile:
            myfile.write(msg)
        
        if save_path and  epoch_save_step:
           if (epoch + 1) % epoch_save_step == 0:
                checkpoint_path = os.path.join(save_path, f"_epoch{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f'Model saved at epoch {epoch + 1} to {checkpoint_path}')
                
                
        # Save the best model based on validation metric
        current_val_metric = np.mean(metrics) if not classification else val_loss.mean()
        if save_path and ((classification and current_val_metric < best_val_metric) or (not classification and current_val_metric > best_val_metric)):
            best_val_metric = current_val_metric
            torch.save(model.state_dict(), best_model_path)
            best_epoch = epoch + 1
            print(f'Model saved at epoch {epoch + 1} to {best_model_path}')

     


    if save_path:
        return train_losses, val_losses, val_metrics , best_model_path,best_epoch
    else :
        return train_losses, val_losses, val_metrics
###############################################################################
