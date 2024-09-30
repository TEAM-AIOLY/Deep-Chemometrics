import os
from data.load_dataset_atonce import  SpectralDataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from net.chemtools.PLS import PLS
from net.chemtools.metrics import ccc, r2_score
import pickle
import pyppt as ppt

add_fig=True


if __name__ == "__main__":
    
    #########################################################################################################
    user = os.environ.get('USERNAME')
    if user == 'fabdelghafo':
        data_path = "C:\\00_aioly\\GitHub\datasets\\ossl\\ossl_all_L1_v1.2.csv"
    else:
        data_path = "/home/metz/deepchemometrics/Aioly/data/dataset/oss/ossl_all_L1_v1.2.csv"

    y_labels = ["oc_usda.c729_w.pct","na.ext_usda.a726_cmolc.kg","clay.tot_usda.a334_w.pct","k.ext_usda.a725_cmolc.kg","ph.h2o_usda.a268_index"]  
    dataset_type = "nir"
    
    
    base_path = os.path.join(os.path.dirname(data_path), 'figures', 'pls_multi', f'data_{dataset_type}')
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    plt.rcParams['font.family'] = 'Times New Roman'

    # Instantiate the dataset
    dataset = SpectralDataset(data_path, y_labels, dataset_type)
    spec_dims=dataset.spec_dims

    #########################################################################################################
    
    wavelength = dataset.get_spectral_dimensions()

    X_train=dataset.X_train
    X_val=dataset.X_val
    Y_train=dataset.Y_train
    Y_val=dataset.Y_val
    #########################################################################################################
    ncomp=50
    pls =PLS(ncomp=ncomp)
    #########################################################################################################

    Y_train_log = torch.log1p(Y_train)
    Y_val_log = torch.log1p(Y_val)

    pls.fit(X_train, Y_train_log)

    perf = {label: [] for label in y_labels}  # RMSE for each label
    for lv in range(ncomp):
        Y_pred = pls.predict(X_val, lv)
        for i, target_label in enumerate(y_labels):
            y_pred_label = Y_pred[:, i].unsqueeze(1)
            Y_val_label = Y_val_log[:, i].unsqueeze(1)
            rmse = torch.sqrt(F.mse_loss(y_pred_label, Y_val_label, reduction='none')).mean(dim=0)
            perf[target_label].append(rmse)
              

    # Predict with all latent variables
    Y_pred_final = pls.predict(X_val, ncomp-1)
    
    
    for target_label in y_labels:
        target_index = y_labels.index(target_label)
        
        Y_val_subset = Y_val_log[:, target_index].unsqueeze(1)
        y_pred_label = Y_pred_final[:, target_index].unsqueeze(1)
        
        ccc_value = ccc(Y_val_subset, y_pred_label)
        r2_value = r2_score(Y_val_subset, y_pred_label)
        
        fig = plt.figure()
        plt.plot(range(1, ncomp + 1), [p.item() for p in perf[target_label]], label=target_label, color=default_colors[target_index])
        
        plt.text(0.95, 0.95, f'CCC: {ccc_value:.2f}\nR²: {r2_value:.2f}', 
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                 color='red', fontweight='bold', fontfamily='serif')
        
        plt.xlabel('Latent Variables')
        plt.ylabel('RMSEP')
        plt.title(f'Training RMSE for {target_label} (log x +1)')
        plt.tight_layout()
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   fancybox=True, shadow=True, ncol=5, labelcolor=default_colors[target_index], fontsize=12)
        plt.subplots_adjust(bottom=0.3)
        plt.grid(True)
        # fig.show()
        if add_fig==True:
            ppt.add_figure('Center')
        
       
         
        pdf_path = os.path.join(base_path, f'fig_RMSE_{target_label}.pdf')
        plt.savefig(pdf_path, format='pdf')
        
        svg_path = os.path.join(base_path, f'fig_RMSE_{target_label}.svg')
        plt.savefig(svg_path, format='svg')

        pickle_path = os.path.join(base_path, f'fig_RMSE_{target_label}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(fig, f)
            
        fig, ax = plt.subplots()
        hexbin = ax.hexbin(Y_val_subset.squeeze().numpy(), y_pred_label.squeeze().numpy(), gridsize=50, cmap='inferno', mincnt=1)
        cb = fig.colorbar(hexbin, ax=ax, orientation='vertical')
        cb.set_label('Density')
        lims = [np.min([Y_val_subset.numpy(), y_pred_label.numpy()]), np.max([Y_val_subset.numpy(), y_pred_label.numpy()])]
        ax.plot(lims, lims, 'k-', label= target_label)  
        
        plt.text(0.05, 0.95, f'CCC: {ccc_value:.2f}\nR²: {r2_value:.2f}', 
                 transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                 color='red', fontweight='bold', fontfamily='serif')
        
        ax.set_xlabel('Real Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'Predicted vs Real Values for {target_label} (log x + 1)')
        
        
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                   fancybox=True, shadow=True, ncol=5, fontsize=12,labelcolor =default_colors[target_index])
        plt.tight_layout()
        plt.grid()
        
        # fig.show()
        if add_fig==True:
            ppt.add_figure('Center')
        
        hexbin_pdf_path = os.path.join(base_path, f'fig_hexbin_{target_label}.pdf')
        plt.savefig(hexbin_pdf_path, format='pdf')
        
        hexbin_svg_path = os.path.join(base_path, f'fig_hexbin_{target_label}.svg')
        plt.savefig(hexbin_svg_path, format='svg')

        # Save the figure object using pickle
        hexbin_pickle_path = os.path.join(base_path, f'fig_hexbin_{target_label}.pkl')
        with open(hexbin_pickle_path, 'wb') as f:
            pickle.dump(fig, f)
            
 