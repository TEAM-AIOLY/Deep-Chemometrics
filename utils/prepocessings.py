import torch
import numpy as np
from scipy.signal import savgol_filter
from abc import ABC, abstractmethod
import sys

class Preprocessing(ABC):
    @abstractmethod
    def apply_preprocessing(self, x:torch.Tensor) -> torch.Tensor:
        pass

    def correct(self, x:torch.Tensor) -> torch.Tensor :
        x_corr = self.apply_preprocessing(x)
        if x_corr.shape != x.shape:
            sys.exit(f"size missmatch between x and its correction, the preprocessing should conserve spetral dimension, dim of x : {x.shape}, dim of x_corr : {x_corr.shape}")
        else :
            return x_corr
        
class SNV(Preprocessing):
    def apply_preprocessing(self, x:torch.Tensor) -> torch.Tensor:
        m = torch.mean(x,dim=1).unsqueeze(1)
        print(m.shape)
        std = torch.std(x,dim=1).unsqueeze(1)
        return (x-m) / std
    
    def correct(self, x):
        return self.apply_preprocessing(x)
    
class EMSC(Preprocessing):
    def __init__(self,order=2):
        self.order = order
        self.X = None
        self.A = None

    def fit(self, x:torch.Tensor):
        self.reference = torch.mean(x, axis=0)
        ref = self.reference.unsqueeze(1)
        wavenumbers = torch.arange(ref.shape[0],dtype=ref.dtype)
        # squeeze wavenumbers to approx. range [-1; 1]
        # use if else to support uint types
        if wavenumbers[0] > wavenumbers[-1]:
            rng = wavenumbers[0] - wavenumbers[-1]
        else:
            rng = wavenumbers[-1] - wavenumbers[0]
        half_rng = rng / 2
        normalized_wns = (wavenumbers - torch.mean(wavenumbers)) / half_rng

        polynomial_columns = [torch.ones(wavenumbers.shape[0])]
        for j in range(1, self.order + 1):
            polynomial_columns.append(normalized_wns ** j)
        polynomial_columns = torch.stack(polynomial_columns).T

        # spectrum = X*coefs + residues
        # least squares -> A = (X.T*X)^-1 * X.T; coefs = A * spectrum
        columns = (ref, polynomial_columns)

        self.X = torch.concatenate(columns, axis=1) # shape nb_lambda*4
        self.A = torch.matmul(torch.linalg.pinv(torch.matmul(self.X.T, self.X)), self.X.T) # shape 4*200
        
    def get_coefs_residues_X(self, x:torch.Tensor,device):
        A = self.A.to(device)
        X = self.X.to(device)
        x = x.to(device)
        coefs = torch.matmul(A, x.T) # shape (4*nb_lambda)*(nb_lambda*n)=(4*n), mul, offset, slope, 2nd degree
        residues = x - torch.matmul(X, coefs).T
        return coefs.T, residues, X.T
    
    def apply_preprocessing(self, x:torch.Tensor) -> torch.Tensor:
        # spectrum = X*coefs + residues
        # least squares -> A = (X.T*X)^-1 * X.T; coefs = A * spectrum 
        coefs = torch.matmul(self.A, x.T) # shape (4*nb_lambda)*(nb_lambda*n)=(4*n), mul, offset, slope, 2nd degree
        residues = x - torch.matmul(self.X, coefs).T # n*nb_lambda 

        preprocessed_x = residues/coefs[0].unsqueeze(1) + self.reference  # n*nb_lambda
        return preprocessed_x
    
    def correct(self, x):
        return super().correct(x)
    
class Sav_Gol(Preprocessing):
    def __init__(self, window_size, poly_order, derivative):
        self.window_size = window_size
        self.poly_order = poly_order
        self.derivative = derivative

    def apply_preprocessing(self, x:torch.Tensor) -> torch.Tensor:
        xnp = x.numpy()
        xnp_corr = savgol_filter(xnp,self.window_size, self.poly_order, self.derivative)
        return torch.tensor(xnp_corr)

def loadPreprocessing(pp,x_train) -> Preprocessing:
    if pp["name"] == "SNV":
        return SNV()
    elif pp["name"]=="EMSC" :
        if "order" in pp:
            emsc = EMSC(order=pp["order"])
            emsc.fit(x_train)
            return emsc
        else :
            emsc = EMSC()
            emsc.fit(x_train)
            return emsc
    elif pp["name"]=="SavGol":
        return Sav_Gol(pp["window_size"], pp["poly_order"], pp["derivative"])
    else : 
        sys.exit("for now only snv, emsc and SavGol are implemented, for emsc you can specify the order in config file, order is 2 by default")