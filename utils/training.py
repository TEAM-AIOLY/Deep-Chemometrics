import torch
import torcheval.metrics
import numpy as np
import os
import matplotlib.pyplot as plt
            
            
            
###############################################################################
def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, save_path=None,classification = False, epoch_save_step =250):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    
    train_losses = []
    val_losses = []
    val_r2_scores = []
    val_f1_scores = []
    
    if classification:
         min_val_loss = np.inf
    else:
        min_val_loss = -np.inf
        
    best_model_state = None  
    best_epoch = -1  

    for epoch in range(num_epochs):
        model.train()
        if classification:
            running_loss = torch.zeros(1, device=device)
        else :
             running_loss = torch.zeros(model.out_dims, device=device)
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
        train_losses.append(epoch_loss.detach().cpu())       
           
        # Validation loop
        model.eval()
        if classification:
            val_loss = torch.zeros(1, device=device)
        else:
            val_loss = torch.zeros(model.out_dims, device=device)
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
        r2_scores = []
        f1_scores = []
        
        if not classification:
            R2 = [torcheval.metrics.R2Score() for _ in range(model.out_dims)]
            for i in range(model.out_dims):
                R2[i].update(all_targets[:, i],all_outputs[:, i])
                r2_score = R2[i].compute().item()
                r2_scores.append(r2_score)
            
            val_r2_scores.append(r2_scores)
        else :
            F1 = torcheval.metrics.MulticlassF1Score()
            F1.update(torch.argmax(all_targets,dim=1), torch.argmax(all_outputs,dim=1))
            f1_scores = F1.compute()
            val_f1_scores.append(f1_scores)


        
        train_loss_str = ', '.join([f'y {i}: {loss:.4f}' for i, loss in enumerate(epoch_loss)])
        val_loss_str = ', '.join([f'y {i}: {loss:.4f}' for i, loss in enumerate(val_loss)])
        r2_score_str = ', '.join([f'y {i}: {score:.4f}' for i, score in enumerate(r2_scores)])
        
        if classification :
            print(
                f'Epoch {epoch + 1}/{num_epochs} | Train Losses: {train_loss_str} | Validation Losses: {val_loss_str}| F1 Score: {f1_scores}')
        else :
            print(f'Epoch {epoch+1}/{num_epochs} | Train Losses: {train_loss_str} | Validation Losses: {val_loss_str} | R2 Scores: {r2_score_str}')

        
        if save_path:
            if classification:
                current_metric = val_loss.mean()  # Lower is better
                if current_metric < min_val_loss and (epoch + 1) > num_epochs * 0.1:
                    min_val_loss = current_metric
                    best_model_state = model.state_dict().copy()  # Store best model state
                    best_epoch = epoch
            else:
                current_metric = val_loss.mean()
                if current_metric  < min_val_loss and (epoch + 1) > num_epochs * 0.1:
                    min_val_loss = current_metric  
                    best_model_state = model.state_dict().copy()
                    best_epoch = epoch

            # Save checkpoint every 100 epochs
            if (epoch + 1) % epoch_save_step == 0:
                checkpoint_path = os.path.join(save_path, f"checkpoint_epoch{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"💾 Checkpoint saved at epoch {epoch + 1} to {checkpoint_path}")
            
    if save_path and best_model_state is not None:
        best_model_path = os.path.join(save_path, f"best_model_{best_epoch+1}.pth")
        torch.save(best_model_state, best_model_path)
        print(f"✅ Best model saved from epoch {best_epoch + 1} to {best_model_path}")



    if save_path:
        return train_losses, val_losses, val_r2_scores , best_model_state
    else :
        return train_losses, val_losses, val_r2_scores
###############################################################################
