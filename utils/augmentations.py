import numpy as np
import torch
import matplotlib.pyplot as plt

from typing import Union as U, Tuple as T


def aug_slope(spectra):
    spectra = spectra.numpy()
    x = np.arange(200)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, spectra)[0]
    errors = spectra-(m*x + c)
    mv = m - np.random.uniform(-0.002,0.002)
    cv = c -np.random.uniform(-0.05,0.05)
    return torch.tensor(mv*x+cv+errors)

def aug_slope_11(spectra):
    spectra = spectra.numpy()
    x = np.arange(200)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, spectra)[0]
    errors = spectra -(m*x + c)
    mv = m - np.random.uniform(-0.002,0.002,10)
    cv = c -np.random.uniform(-0.05,0.05,10)
    y_out = spectra
    for i in range(10):
        y_out = np.vstack((y_out,mv[i]*x+cv[i]+errors))
    return torch.tensor(y_out)

def emsc(spectra: torch.Tensor, order: int = 2,
         reference: torch.Tensor = None,
         constituents: torch.Tensor = None) -> U[torch.Tensor, T[torch.Tensor, torch.Tensor]]:
    """
    Preprocess all spectra with EMSC
    :param spectra: ndarray of shape [n_samples, n_channels]
    :param wavenumbers: ndarray of shape [n_channels]
    :param order: order of polynomial
    :param reference: reference spectrum
    :param constituents: ndarray of shape [n_consituents, n_channels]
    Except constituents it can also take orthogonal vectors,
    for example from PCA.
    :param return_coefs: if True returns coefficients
    [n_samples, n_coeffs], where n_coeffs = 1 + len(costituents) + (order + 1).
    Order of returned coefficients:
    1) b*reference +                                    # reference coeff
    k) c_0*constituent[0] + ... + c_k*constituent[k] +  # constituents coeffs
    a_0 + a_1*w + a_2*w^2 + ...                         # polynomial coeffs
    :return: preprocessed spectra
    """
    if reference is None:
        reference = torch.mean(spectra, axis=0)
    reference = reference.unsqueeze(1)

    wavenumbers = torch.arange(reference.shape[0],dtype=reference.dtype)

    # squeeze wavenumbers to approx. range [-1; 1]
    # use if else to support uint types
    if wavenumbers[0] > wavenumbers[-1]:
        rng = wavenumbers[0] - wavenumbers[-1]
    else:
        rng = wavenumbers[-1] - wavenumbers[0]
    half_rng = rng / 2
    normalized_wns = (wavenumbers - torch.mean(wavenumbers)) / half_rng

    polynomial_columns = [torch.ones(wavenumbers.shape[0])]
    for j in range(1, order + 1):
        polynomial_columns.append(normalized_wns ** j)
    polynomial_columns = torch.stack(polynomial_columns).T

    # spectrum = X*coefs + residues
    # least squares -> A = (X.T*X)^-1 * X.T; coefs = A * spectrum
    if constituents is None:
        columns = (reference, polynomial_columns)
    else:
        columns = (reference, constituents.T, polynomial_columns)

    X = torch.concatenate(columns, axis=1) # shape 200*4
    A = torch.matmul(torch.linalg.pinv(torch.matmul(X.T, X)), X.T) # shape 4*200

    spectra_columns = spectra.T # shape 200*n 
    coefs = torch.matmul(A, spectra_columns) # shape 4*n, mul, offset, slope, 2nd degree
    residues = spectra_columns - torch.matmul(X, coefs) # 200*n 

    preprocessed_spectra = (reference + residues/coefs[0]).T # n*200

    return preprocessed_spectra, coefs.T, residues.T, X.T

def emsc_pca(spectra: torch.Tensor, dataset: str, order_emsc: int = 2,
         reference: torch.Tensor = None,
         constituents: torch.Tensor = None) : # 1ere composante, mean_c0, std_c0, coefs.T, residues.T, X.T
    
    if reference is None:
        reference = torch.mean(spectra, axis=0)
    reference = reference.unsqueeze(1)
    wavenumbers = torch.arange(reference.shape[0],dtype=reference.dtype)

    # squeeze wavenumbers to approx. range [-1; 1]
    # use if else to support uint types
    if wavenumbers[0] > wavenumbers[-1]:
        rng = wavenumbers[0] - wavenumbers[-1]
    else:
        rng = wavenumbers[-1] - wavenumbers[0]
    half_rng = rng / 2
    normalized_wns = (wavenumbers - torch.mean(wavenumbers)) / half_rng

    polynomial_columns = [torch.ones(wavenumbers.shape[0])]
    for j in range(1, order_emsc + 1):
        polynomial_columns.append(normalized_wns ** j)
    polynomial_columns = torch.stack(polynomial_columns).T

    # compute X and A matrices
    if constituents is None:
        columns = (reference, polynomial_columns)
    else:
        columns = (reference, constituents.T, polynomial_columns)

    # spectrum = X*coefs + residues
    # least squares -> A = (X.T*X)^-1 * X.T; coefs = A * spectrum
    X = torch.concatenate(columns, axis=1) # shape 200*4
    A = torch.matmul(torch.linalg.pinv(torch.matmul(X.T, X)), X.T) # shape 4*200
    spectra_columns = spectra.T # shape 200*n 
    coefs = torch.matmul(A, spectra_columns) # shape 4*n, mul, offset, slope, 2nd degree
    residues = spectra_columns - torch.matmul(X, coefs) # 200*n 
    coefs = coefs.T # shape n*4
    residues = residues.T # shape n*200
    X = X.T

    # PCA
    order_pca = order_emsc+2 # max_order
    coefs_mean = torch.mean(coefs,dim=0)
    coefs_c = coefs - coefs_mean
    U,S,V = torch.pca_lowrank(coefs_c,q=order_pca) # True to center the data
    pc0 = V[:,0]
    explained_variance = (S ** 2) / (coefs.size(0) -1 )
    explained_var_ratio = explained_variance / explained_variance.sum()
    scores_pc0 = U[:,0]*S[0]
    scores_pc0_mean = torch.mean(scores_pc0)
    scores_pc0_std = torch.std(scores_pc0)
    # save histo
    plt.hist(scores_pc0)
    plt.title(f"var explained : {explained_var_ratio[0]:.2f}")
    plt.savefig(f"score_1stcomp_{dataset}.png")

    return pc0, scores_pc0_mean, scores_pc0_std, coefs, coefs_mean, residues, X

def emsa(coefs, residues, coefs_std, X, device):
    # dim coefs = (batch, 4) dim residues = (batch, 200), dim X = (4,200)
    deviations = torch.normal(mean=0, std=coefs_std.unsqueeze(0).repeat(coefs.shape[0],1)).to(device) # sample n*4 deviations 
    new_coefs = coefs + deviations # create new coefs
    mask = new_coefs[:,0]<0
    new_coefs[mask,0] = -new_coefs[mask,0] # make sure multiplicative effect is positive
    return torch.matmul(new_coefs, X) + residues * new_coefs[:,0].unsqueeze(1) / coefs[:,0].unsqueeze(1)

def emsa_pca(coefs, coefs_mean, residues, pc0, scores_pc0_mean, scores_pc0_std, X, device):
    new_pc0_scores = torch.normal(mean=0, std=scores_pc0_std.repeat(coefs.shape[0])).to(device) # sample nb_batch deviations
    new_coefs = torch.matmul(new_pc0_scores.unsqueeze(1),pc0.unsqueeze(0)) +coefs_mean
    return torch.matmul(new_coefs, X) + residues * new_coefs[:,0].unsqueeze(1) / coefs[:,0].unsqueeze(1)
    
class data_augmentation:
    def __init__(self, slope = 0.1, offset = 0.1, multiplicative = 0.1, noise = 0.1, shift = 0.1):
        self.slope = slope
        self.offset = offset
        self.multiplicative = multiplicative
        self.noise = noise
        self.shift = shift

    def __call__(self, X):
        X_aug = np.zeros_like(X)
        X_slope = (np.random.uniform(-self.slope, self.slope, (X.shape[0],1)))*np.arange(X.shape[1])
        X_offset = np.random.uniform(-self.offset, self.offset, (X.shape[0],1))
        X_mul = X * (1-np.random.uniform(-self.multiplicative, self.multiplicative, (X.shape[0],1)))
        X_aug = X_mul + X_offset + X_slope + np.random.normal(0, self.noise, X.shape)
        return X_aug