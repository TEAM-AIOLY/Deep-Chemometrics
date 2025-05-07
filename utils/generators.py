import torch

from abc import ABC, abstractmethod
from utils.constants import *


class Generator(ABC):
    @abstractmethod
    def generate_residues(self,nb:int,loadings:torch.Tensor,scores:torch.Tensor=None,mean_residues:torch.Tensor=None) -> torch.Tensor:
        pass

class GaussianGeneratorWithScores(Generator):
    def __init__(self, loadings:torch.Tensor, scores:torch.Tensor, residues_mean:torch.Tensor) -> torch.Tensor:
        super().__init__()
        self.loadings = loadings.type(torch.float32)
        self.residues_mean = residues_mean.type(torch.float32)
        self.scores_std = torch.std(scores,dim=0).type(torch.float32)
        
    def generate_residues(self, idx):
        nb = idx.shape[0]
        new_scores = torch.normal(mean=0, std=self.scores_std.unsqueeze(0).repeat(nb,1)) # sample nb scores from gaussians
        residues = torch.matmul(new_scores,self.loadings) + self.residues_mean
        return residues

class LocalGaussianGeneratorWithScores(Generator):
    def __init__(self, loadings:torch.Tensor, scores:torch.Tensor, residues_mean:torch.Tensor, alpha:float=0.1) -> torch.Tensor:
        super().__init__()
        self.loadings = loadings.type(torch.float32)
        self.residues_mean = residues_mean.type(torch.float32)
        self.scores = scores.type(torch.float32)
        self.scores_std = torch.std(scores,dim=0).type(torch.float32)
        self.alpha=alpha
        
    def generate_residues(self, idx):
        deviations = torch.normal(mean=0, std=(self.alpha*self.scores_std).unsqueeze(0).repeat(len(idx),1)) # lamda parameter to setup
        new_scores = self.scores[idx,:] + deviations # old_scores + deviations
        residues = torch.matmul(new_scores,self.loadings) + self.residues_mean
        return residues

class MVNGenerator(Generator):
    def __init__(self, loadings:torch.Tensor, residues_mean:torch.Tensor) -> torch.Tensor:
        super().__init__()
        self.mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc=residues_mean,covariance_matrix=loadings)
        
    def generate_residues(self, idx):
        nb = idx.shape[0]
        residues = torch.tensor(self.mvn.sample((nb,)),dtype=torch.float32)
        return residues

class EMSA():
    def __init__(self, coefs, residues, X):
        self.coefs = coefs # bs*(order+1)
        self.residues = residues # bs*nb_lambda 
        self.coefs_std = torch.std(coefs,axis=0) # (order+1)
        self.X = X # (order+1)*nb_lambda

    def generate_emsa(self, idx):
        coefs = self.coefs[idx,:]
        residues = self.residues[idx,:]
        deviations = torch.normal(mean=0, std=self.coefs_std.unsqueeze(0).repeat(len(idx),1)) # bs*(order+1) deviations 
        new_coefs = coefs + deviations # # bs*(order+1)
        mask = new_coefs[:,0]<0 
        new_coefs[mask,0] = -new_coefs[mask,0] # make sure multiplicative effect is positive
        return (torch.matmul(new_coefs, self.X) + residues * new_coefs[:,0].unsqueeze(1) / coefs[:,0].unsqueeze(1)).type(torch.float32)
        

def loadGenerator(gen:str, loadings:torch.Tensor, residues_mean:torch.Tensor, scores:torch.Tensor=None):
    if gen["name"]=="localGaussian":
        if "alpha" in gen:
            return LocalGaussianGeneratorWithScores(loadings,scores,residues_mean,gen["alpha"])
        else:
            return LocalGaussianGeneratorWithScores(loadings,scores,residues_mean)
    elif gen["name"]=="gaussian":
        return GaussianGeneratorWithScores(loadings,scores,residues_mean)
    elif gen["name"]==MVN:
        return MVNGenerator(loadings,residues_mean)