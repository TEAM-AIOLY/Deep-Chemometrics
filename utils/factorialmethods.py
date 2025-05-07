import torch

from typing import Union as U
from abc import ABC, abstractmethod
import sys
from utils import constants

class FactorialMethod(ABC):
    @abstractmethod
    def fit_fm(self, residues:torch.Tensor,device) -> U[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    # def apply_factorial_method(self, residues:torch.tensor,nb_comp=int) -> U[torch.tensor, torch.tensor, torch.tensor]:
    #     scores, loadings, res_fm = self.apply_factorial_method(residues)
    #     if scores.shape != (residues.shape[0],nb_comp) or loadings.shape != (nb_comp, residues.shape[1]):
    #         sys.exit(f"the factorial method should return scores, loadings and remaining residues of dim {(residues.shape[0],nb_comp)}, {(nb_comp, residues.shape[1])} and {residues.shape}, dims are {scores.shape}, {loadings.shape} and {res_fm.shape}")
    #     else :
    #         return scores, loadings, res_fm        

class PCA(FactorialMethod):
    def __init__(self,lim_explained_var:float=0.9,nb_comp=None): #by default use lim_explained_var, if nb_comb is not none use nb_comb
        super().__init__()
        self.lim_explained_var = lim_explained_var
        self.nb_comp=nb_comp

    def fit_fm(self, residues:torch.Tensor,device) -> U[torch.Tensor, torch.Tensor, torch.Tensor]:
        residues = residues.to(device)
        residues_mean = torch.mean(residues,dim=0)
        residues_c = residues - residues_mean
        U,S,V = torch.pca_lowrank(residues_c,q=torch.min(torch.tensor(residues_c.shape))) # by default nb_comp=min(6,nb_indiv,nb_var)
        explained_var_per_comp = (S ** 2) / (residues_c.size(0) -1)
        explained_var_ratio = explained_var_per_comp / explained_var_per_comp.sum()
        explained_var = explained_var_ratio[0]
        if self.nb_comp != None:
            loadings = V[:,:self.nb_comp]
            scores = U[:,:self.nb_comp]*S[:self.nb_comp]
        else :
            last_comp = 0
            while explained_var<self.lim_explained_var and last_comp<(min(residues_c.shape)-1):
                last_comp += 1
                explained_var += explained_var_ratio[last_comp]
            loadings = V[:,:last_comp+1]
            scores = U[:,:last_comp+1]*S[:last_comp+1]
        return loadings.T, scores, residues_mean
    
class MVN(FactorialMethod):
    def __init__(self):
        super().__init__()

    def fit_fm(self, residues:torch.Tensor,device) -> U[torch.Tensor, torch.Tensor, torch.Tensor]:
        residues = residues.to(device)
        residues_mean = torch.mean(residues,dim=0)
        residues_cov = torch.cov(residues.T)
        return residues_cov, None, residues_mean # None because other factorial methods also return scores
    
def loadFactorialMethod(fm,dim:tuple)->FactorialMethod:
    if fm['name'] == constants.PCA :
        if "nb_comp" in fm:
            if fm['nb_comp'] == "all":
                return PCA(min(dim))
            else: 
                nb_comp = int(fm['nb_comp'])
                if nb_comp > min(dim):
                    sys.exit("number of component should be smaller or equal to the minimum between number of variables and number of individues, use 'all' to use all components")
                return (PCA(nb_comp=nb_comp))
        elif "var_exp" in fm:
            var_exp = float(fm['var_exp'])
            if var_exp > 1. :
                sys.exit("variance explained should be smaller than 1")
            return PCA(lim_explained_var=var_exp)
        else : 
            return PCA(min(dim))
    elif fm['name'] == constants.MVN:
        return MVN()
    else : 
        sys.exit("for now only PCA and MVN are available, for PCA default use all components, you can specify a number of components with nb_comp or the variance of the dataset you want with var_exp")