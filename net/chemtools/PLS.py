import torch
from net.chemtools.metrics import ccc,r2_score
import torch.nn.functional as F



class PLS:
    def __init__(self, ncomp, weights=None):
        self.ncomp = ncomp
        self.weights = weights

    def fit(self, X, Y):
        # Convert input arrays to PyTorch tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64).clone().detach()
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float64).clone().detach()

        n, zp = X.shape
        q = Y.shape[1]

        if self.weights is None:
            self.weights = torch.ones(n, dtype=torch.float64) / n
        else:
            self.weights = torch.tensor(self.weights, dtype=torch.float64).clone().detach() / torch.sum(self.weights)

        self.xmeans = torch.sum(self.weights[:, None] * X, dim=0)
        X = X - self.xmeans

        self.ymeans = torch.sum(self.weights[:, None] * Y, dim=0)
        Y = Y - self.ymeans

        self.T = torch.zeros((n, self.ncomp), dtype=torch.float64)
        self.R = torch.zeros((zp, self.ncomp), dtype=torch.float64)
        self.W = torch.zeros((zp, self.ncomp), dtype=torch.float64)
        self.P = torch.zeros((zp, self.ncomp), dtype=torch.float64)
        self.C = torch.zeros((q, self.ncomp), dtype=torch.float64)
        self.TT = torch.zeros(self.ncomp, dtype=torch.float64)

        Xd = self.weights[:, None] * X
        tXY = Xd.T @ Y

        for a in range(self.ncomp):
            if q == 1:
                w = tXY[...,0]
            else:
                u, _, _ = torch.svd(tXY.T, some=False)
                u = u[:, 0]
                w = tXY @ u

            w = w / torch.sqrt(torch.sum(w * w))

            r = w.clone()
            if a > 0:
                for j in range(a):
                    r = r - torch.sum(self.P[:, j] * w) * self.R[:, j]

            t = X @ r
            tt = torch.sum(self.weights * t * t)

            c = (tXY.T @ r) / tt
            p = (Xd.T @ t) / tt

            tXY = tXY - (p[:, None] @ c[None]) * tt

            self.T[:, a] = t
            self.P[:, a] = p
            self.W[:, a] = w
            self.R[:, a] = r
            self.C[:, a] = c
            self.TT[a] = tt

    def transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64)
        else:
            X = X.clone().detach().to(dtype=torch.float64)
        X = X - self.xmeans
        T_new = X @ self.R
        return T_new

    def fit_transform(self, X, Y):
        self.fit(X, Y)
        return self.T

    def get_params(self):
        return {
            "T": self.T,
            "P": self.P,
            "W": self.W,
            "C": self.C,
            "R": self.R,
            "TT": self.TT,
            "xmeans": self.xmeans,
            "ymeans": self.ymeans,
            "weights": self.weights,
            "T.ortorcho": True
        }

    def predict(self, X, nlv=None):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64)
        else:
            X = X.clone().detach().to(dtype=torch.float64)
            
        X = X - self.xmeans

        if nlv is None:
            nlv = self.ncomp
        else:
            nlv = min(nlv, self.ncomp)

        B = self.W[:, :nlv] @ torch.inverse(self.P[:, :nlv].T @ self.W[:, :nlv]) @ self.C[:, :nlv].T
        predictions = X @ B + self.ymeans
        return predictions



class LDA:
    def __init__(self, prior="unif"):
        """
        Initialize the LDA class.
        Args:
            prior (str): Prior probabilities. Options are "unif" (uniform) or "prop" (proportional).
        """
        self.prior = prior
        self.ct = None  # Class means
        self.W = None   # Within-class covariance matrix
        self.wprior = None  # Weighted priors
        self.lev = None  # Class labels
        self.ni = None   # Number of samples per class
    def fit(self, X, y):
        """
        Fit the LDA model.

        Args:
            X (torch.Tensor): Predictor variables (n_samples, n_features).
            y (torch.Tensor): Class labels (n_samples, n_classes), one-hot encoded.
        """
        # Ensure X and y are tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64)
        else:
            X = X.clone().detach().to(dtype=torch.float64)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float64)
        else:
            y = y.clone().detach()

        # Ensure y is 2D and one-hot encoded
        if y.ndim != 2:
            raise ValueError("y must be a 2D tensor (one-hot encoded).")

        n, p = X.shape  # Number of samples and features
        n_classes = y.shape[1]  # Number of classes

        # Compute class sample sizes (sum of one-hot encoded values for each class)
        self.ni = y.sum(dim=0)  # Shape: (n_classes,)

        # Compute class means (weighted by one-hot encoding)
        self.ct = (y.T @ X) / self.ni[:, None]  # Shape: (n_classes, n_features)

        # Compute prior probabilities
        if self.prior == "unif":
            self.wprior = torch.ones(n_classes, dtype=torch.float64) / n_classes
        elif self.prior == "prop":
            self.wprior = self.ni / self.ni.sum()

        # Compute within-class scatter matrix
        W = torch.zeros((p, p), dtype=torch.float64)
        for i in range(n_classes):
            diff = X - self.ct[i]  # Difference between samples and class mean
            weighted_diff = (y[:, i][:, None] * diff)  # Weight by one-hot encoding
            W += weighted_diff.T @ diff
        self.W = W / (n - n_classes)  # Normalize by degrees of freedom
   
    def predict(self, X):
        """
        Predict class probabilities for new data.

        Args:
            X (torch.Tensor): New predictor variables (n_samples, n_features).

        Returns:
            torch.Tensor: Posterior probabilities for each class (n_samples, n_classes).
        """
        # Ensure X is a tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64)
        else:
            X = X.clone().detach().to(dtype=torch.float64)
            
        m, p = X.shape  # Number of samples and features
        n_classes = self.ct.shape[0]  # Number of classes

        # Compute discriminant scores
        ds = torch.zeros((m, n_classes), dtype=torch.float64)
        inv_W = torch.inverse(self.W)
        for i in range(n_classes):
            diff = X - self.ct[i]
            ds[:, i] = -0.5 * torch.sum(diff @ inv_W * diff, dim=1)  # Quadratic term
            ds[:, i] += torch.log(self.wprior[i])  # Add log-prior

        # Compute posterior probabilities
        posterior = torch.exp(ds - torch.max(ds, dim=1, keepdim=True).values)  # Avoid numerical underflow
        posterior /= posterior.sum(dim=1, keepdim=True)

        return posterior


class PLSDA(PLS):
    def __init__(self, ncomp, weights=None, prior="unif"):
        super().__init__(ncomp=ncomp, weights=weights)
        self.lda = LDA(prior=prior)
        self.proj = None  # final linear discriminant(s)

    def fit(self, X, Y):
        # Step 1: run PLS to get T scores
        super().fit(X, Y)
        T = self.T  # Shape: (n_samples, n_components)

        # Step 2: LDA in the latent space
        self.lda.fit(T, Y)

        # Step 3: compute projection vector(s)
        # W: (n_features, n_components), lda.W: (n_components, n_components)
        # We project to LDA directions in T-space: final proj = W @ lda_vector
        eigvals, eigvecs = torch.linalg.eigh(self.lda.W)  # from LDA
        top_eigvec = eigvecs[:, -1]  # take the leading one

        self.proj = self.W @ top_eigvec  # shape: (n_features,)
        return self

    def transform(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float64)
        else:
            X = X.clone().detach().to(dtype=torch.float64)
        X = X - self.xmeans
        scores = X @ self.proj
        return scores

    def predict(self, X):
        # Project T manually then call lda.predict
        T_new = super().transform(X)
        return self.lda.predict(T_new)








def getknn(Xr, Xu, k, diss="euclidean"):
    n = Xr.shape[0]
    m = Xu.shape[0]

    if diss == "euclidean":
        distances = torch.cdist(Xu, Xr)
    elif diss == "mahalanobis":
        cov = torch.cov(Xr.T)
        cov_inv = torch.inverse(cov)
        diff = Xu.unsqueeze(1) - Xr.unsqueeze(0)
        distances = torch.sqrt(torch.sum(torch.matmul(diff, cov_inv) * diff, dim=2))
    elif diss == "correlation":
        Xr_mean = torch.mean(Xr, dim=0)
        Xu_mean = torch.mean(Xu, dim=0)
        Xr_centered = Xr - Xr_mean
        Xu_centered = Xu - Xu_mean
        Xr_norm = torch.norm(Xr_centered, dim=1)
        Xu_norm = torch.norm(Xu_centered, dim=1)
        distances = 1 - torch.matmul(Xu_centered, Xr_centered.T) / (Xu_norm.unsqueeze(1) * Xr_norm.unsqueeze(0))
    else:
        raise ValueError(f"Unknown distance type: {diss}")

    knn_indices = torch.argsort(distances, dim=1)[:, :k]
    knn_distances = torch.gather(distances, 1, knn_indices)

    return {"listnn": knn_indices, "listd": knn_distances}


def k_fold_cross_validation(X_train, Y_train, ncomp, k_folds=5):
    fold_rmsecv = [[] for _ in range(k_folds)] 
    fold_ccc = []
    fold_r2 = []
    
    fold_size = X_train.shape[0] // k_folds
    indices  = torch.randperm(X_train.shape[0])
    
    for fold in range(k_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < k_folds - 1 else X_train.shape[0]

        val_indices = indices[val_start:val_end]
        train_indices = torch.cat([indices[:val_start], indices[val_end:]])

        X_train_fold = X_train[train_indices]
        Y_train_fold = Y_train[train_indices]
        X_val_fold = X_train[val_indices]
        Y_val_fold = Y_train[val_indices]

        pls = PLS(ncomp=ncomp)
        pls.fit(X_train_fold, Y_train_fold)
        
        
        perf = []
        for lv in range(ncomp):
            y_pred = pls.predict(X_val_fold, lv)
            rmse = torch.sqrt(F.mse_loss(y_pred, Y_val_fold, reduction='none')).mean(dim=0)
            perf.append(rmse)

        y_pred_final = pls.predict(X_val_fold, ncomp - 1)
        ccc_value = ccc(Y_val_fold, y_pred_final)
        r2_value = r2_score(Y_val_fold, y_pred_final)
        
        fold_rmsecv[fold] = perf
        fold_ccc.append(ccc_value)
        fold_r2.append(r2_value)

    return fold_rmsecv, fold_ccc, fold_r2
    

        