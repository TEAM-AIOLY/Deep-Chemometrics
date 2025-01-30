import torch
import torcheval.metrics
import numpy as np
import os
import matplotlib.pyplot as plt
import pytorch_warmup as warmup
import datetime
            
                     
###############################################################################
def train(model, optimizer, criterion, train_loader, val_loader, num_epochs, save_path=None,classification = False, epoch_save_step =250,doWarmup = False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    

    if doWarmup:
        num_steps = len(train_loader) * num_epochs
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)

    
    train_losses = []
    val_losses = []
    val_r2_scores = []
    val_f1_scores = []
    
    metric = 0.0
    best_metric = -1.0
    
    if save_path:
        with open(save_path + "_telemetry.txt", "a") as myfile:
            dt = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
            myfile.write("\n\n--------------------------------------------------------------\nTraining "
                            + dt
                            + "\n--------------------------------------------------------------\n")

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
            
            if doWarmup:
                with warmup_scheduler.dampening():
                        lr_scheduler.step()
        
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
                R2[i].update(all_targets[:, i], all_outputs[:, i])
                r2_score = R2[i].compute().item()
                r2_scores.append(r2_score)

            val_r2_scores.append(r2_scores)
            metric = r2_scores
        else :

            F1 = torcheval.metrics.MulticlassF1Score()
            F1.update(torch.argmax(all_targets,dim=1), torch.argmax(all_outputs,dim=1))
            f1_scores = F1.compute()
            val_f1_scores.append(f1_scores)
            metric = f1_scores



        
        train_loss_str = ', '.join([f'y {i}: {loss:.4f}' for i, loss in enumerate(epoch_loss)])
        val_loss_str = ', '.join([f'y {i}: {loss:.4f}' for i, loss in enumerate(val_loss)])
        r2_score_str = ', '.join([f'y {i}: {score:.4f}' for i, score in enumerate(r2_scores)])
        
        if classification :
            print(
                f'Epoch {epoch + 1}/{num_epochs} | Train Losses: {train_loss_str} | Validation Losses: {val_loss_str}| F1 Score: {f1_scores}')
            with open(save_path + "_telemetry.txt", "a") as myfile:
                    myfile.write(f'Epoch {epoch + 1}/{num_epochs} | Train Losses: {train_loss_str} | Validation Losses: {val_loss_str} | F1 Score: {f1_scores}\n')
        else :
            print(f'Epoch {epoch+1}/{num_epochs} | Train Losses: {train_loss_str} | Validation Losses: {val_loss_str} | R2 Scores: {r2_score_str}')
            with open(save_path + "_telemetry.txt", "a") as myfile:
                    myfile.write(f'Epoch {epoch + 1}/{num_epochs} | Train Losses: {train_loss_str} | Validation Losses: {val_loss_str} | R2 Score: {r2_score_str}\n')
        
        if save_path and (epoch + 1) % epoch_save_step == 0:
                checkpoint_path = os.path.join(save_path, f"_epoch{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f'Model saved at epoch {epoch + 1} to {checkpoint_path}')
                
                
        if  save_path and metric > best_metric:
            best_save_path = os.path.join(save_path, f'_best.pth')
            best_metric = metric
            torch.save(model.state_dict(), best_save_path)        
            
    if save_path:
        final_save_path = save_path + f'_epoch_{num_epochs}_final.pth'
        torch.save(model.state_dict(), final_save_path)
        print(f"Final model saved at {final_save_path}") 
     


    if save_path:
        return train_losses, val_losses, val_r2_scores , final_save_path
    else :
        return train_losses, val_losses, val_r2_scores
###############################################################################
