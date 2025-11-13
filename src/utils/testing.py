"""
Created on Thu Aug  1 14:58:26 2024

@author: metz
"""
import numpy as np
import pandas as pd

import torch

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import matplotlib as mpl


def RMSEP(y_true, y_pred):
    loss = np.sqrt(np.mean(np.square((y_true - y_pred)), axis=0))
    return loss


def ccc(y_true, y_pred):
    # Means
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)

    # Population variances
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    # Population standard deviations
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    cor = np.corrcoef(y_true, y_pred, rowvar=False)[0][1]

    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    ccc = numerator / denominator
    return (ccc)


def test(model, model_path, test_loader, config, Residual = False , classes = None):
    Y = []
    y_pred = []
    model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    with torch.no_grad():
        for inputs, targets in test_loader:
            Y += targets.to("cpu")
            inputs = inputs.to(device, non_blocking=True).float()
            outputs = model(inputs[:, None])
            y_pred += outputs.to("cpu")

    Y = np.array(Y)
    y_pred = np.array(y_pred)

    if config.classification:
        y_pred = np.exp(y_pred) / np.sum(np.exp(y_pred), axis=1, keepdims=True)
        from sklearn.metrics import classification_report, confusion_matrix
        y_pred = np.argmax(y_pred, axis=1)
        Y = np.argmax(Y, axis=1)
        print(classification_report(Y, y_pred,zero_division=True))
        print(confusion_matrix(Y, y_pred))

    else:
        for i in range(Y.shape[1]):
            ccc_score = ccc(y_pred[:, i], Y[:, i])
            r2_score_ = r2_score(Y[:, i], y_pred[:, i])
            rmsep_score = RMSEP(y_pred[:, i], Y[:, i])

            print(f"CCC: {ccc_score}, R2: {r2_score_}, RMSEP: {rmsep_score}")

    # Creazione figura con due subplot affiancati
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    
    # stile marker
    scatter_kwargs = dict(edgecolors='k', alpha=0.8, s=60, linewidths=0.5)

    # -------------------------------------------------------
    # Prepara mapping colori (categorico vs continuo)
    # -------------------------------------------------------
    cmap = norm = labels = color_list = codes = None
    is_categorical = False

    if classes is not None:
        c_raw = np.asarray(classes)
        if c_raw.shape[0] != Y.shape[0]:
            raise ValueError("`classes` deve avere la stessa lunghezza di Y/y_pred.")

        # categorico se stringhe/oggetti, oppure interi con poche modalità
        n_unique = (pd.unique(c_raw).size
                    if c_raw.dtype.kind in ('U','S','O')
                    else np.unique(c_raw).size)
        is_categorical = (c_raw.dtype.kind in ('U','S','O')) or (np.issubdtype(c_raw.dtype, np.integer) and n_unique <= 20)

        if is_categorical:
            cats   = pd.Categorical(c_raw)   # preserva l'ordine delle categorie
            codes  = cats.codes              # 0..K-1, -1 per NaN
            labels = list(cats.categories)
            K = len(labels)
            base = plt.get_cmap('tab20')     # tavolozza discreta
            color_list = [base(i % base.N) for i in range(K)]
            cmap = mpl.colors.ListedColormap(color_list)
            norm = mpl.colors.BoundaryNorm(np.arange(-0.5, K + 0.5, 1), K)
        else:
            c_vals = c_raw.astype(float)     # continuo

    # -------------------------------------------------------
    # Subplot 1: Predicted vs Observed
    # -------------------------------------------------------
    if classes is None:
        axes[0].scatter(Y[:, i], y_pred[:, i], **scatter_kwargs)
    elif is_categorical:
        m = codes != -1
        axes[0].scatter(Y[m, i], y_pred[m, i],
                        c=codes[m], cmap=cmap, norm=norm, **scatter_kwargs)
        # legenda coerente coi colori dei punti
        handles = [plt.Line2D([0],[0], marker='o', linestyle='',
                            color='w', markerfacecolor=color_list[k],
                            markeredgecolor='k', markersize=8, label=str(lbl))
                for k, lbl in enumerate(labels)]
        axes[0].legend(handles=handles, title="Class")
    else:
        sc0 = axes[0].scatter(Y[:, i], y_pred[:, i],
                            c=c_vals, cmap='viridis', **scatter_kwargs)
        plt.colorbar(sc0, ax=axes[0], label="Class")

    axes[0].plot([Y.min() - 1, Y.max() + 1], [Y.min() - 1, Y.max() + 1], 'r', lw=2)
    axes[0].text(0.95, 0.05,
                f'RMSEP: {rmsep_score:.2f}\nCCC: {ccc_score:.2f}\n$\\mathbf{{R}}^2$: {r2_score_:.2f}',
                transform=axes[0].transAxes, fontsize=12, va='bottom', ha='right',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                color='red', fontweight='bold', fontfamily='serif')
    axes[0].set_xlabel('Observed', fontsize=16)
    axes[0].set_ylabel('Predicted', fontsize=16)
    axes[0].set_title(f'Predicted vs Observed for y{i+1}', fontsize=16)
    axes[0].tick_params(labelsize=14)

    # -------------------------------------------------------
    # Subplot 2: Residui
    # -------------------------------------------------------
    if Residual:
        residuals = Y[:, i] - y_pred[:, i]
        if classes is None:
            axes[1].scatter(Y[:, i], residuals, **scatter_kwargs)
        elif is_categorical:
            m = codes != -1
            axes[1].scatter(Y[m, i], residuals[m],
                            c=codes[m], cmap=cmap, norm=norm, **scatter_kwargs)
            axes[1].legend(handles=handles, title="Classi")
        else:
            sc1 = axes[1].scatter(Y[:, i], residuals,
                                c=c_vals, cmap='viridis', **scatter_kwargs)
            plt.colorbar(sc1, ax=axes[1], label="Classe")
        axes[1].axhline(0, color='r', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Observed', fontsize=16)
        axes[1].set_ylabel('Residuals', fontsize=16)
        axes[1].set_title(f'Residuals for y{i+1}', fontsize=16)
        axes[1].tick_params(labelsize=14)
    else:
        fig.delaxes(axes[1])
        axes = np.array([axes[0]])

    plt.tight_layout()
    plt.show(block=False)