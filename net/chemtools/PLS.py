import torch
from net.chemtools.metrics import ccc,r2_score
import torch.nn.functional as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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



class PLSDA(PLS):
    def __init__(self, ncomp, weights=None):
        super().__init__(ncomp, weights)
        self.lda = None  # LDA model will be initialized after PLS fitting
        self.lda_scores = None  # LDA scores (projections)
        self.lda_loadings = None  # LDA loadings (discriminant vectors)

    def fit(self, X, Y, class_labels):
        """
        Fit the PLS model and then perform LDA on the PLS scores.
        
        Args:
            X (torch.Tensor or np.ndarray): Predictor variables.
            Y (torch.Tensor or np.ndarray): Response variables.
            class_labels (array-like): Class labels for LDA (conjunctive labels).
        """
        if not isinstance(Y, torch.Tensor):
            Y = torch.tensor(Y, dtype=torch.float64)    
        # Convert disjunctive labels (Y) to conjunctive labels if needed
        if Y.ndim > 1:  # Check if Y is one-hot encoded
            class_labels = torch.argmax(Y, dim=1).numpy()  # Convert to class indices
        else:
            class_labels = class_labels  # Use provided class_labels directly
        # Fit the PLS model
        super().fit(X, Y)

        # Perform LDA on the PLS scores
        scores = self.T.numpy()  # Convert PLS scores to NumPy for LDA
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(scores, class_labels)

       
        # LDA scores
        self.lda_loadings =self.lda.coef_  # LDA loadings (discriminant vectors)
        self.lda_scores = scores @ self.lda_loadings.T

    def predict(self, X, nlv=None):
        """
        Predict class labels using the LDA model on new data.
        
        Args:
            X (torch.Tensor or np.ndarray): New predictor variables.
            nlv (int, optional): Number of latent variables to use. Defaults to all components.
        
        Returns:
            np.ndarray: Predicted class labels.
        """
        # Transform the new data into PLS scores
        scores = super().transform(X).numpy()

        # Use the LDA model to predict class labels
        predictions = self.lda.predict(scores)
        return predictions

    def predict_proba(self, X, nlv=None):
        """
        Predict class probabilities using the LDA model on new data.
        
        Args:
            X (torch.Tensor or np.ndarray): New predictor variables.
            nlv (int, optional): Number of latent variables to use. Defaults to all components.
        
        Returns:
            np.ndarray: Predicted class probabilities.
        """
        # Transform the new data into PLS scores
        scores = super().transform(X).numpy()

        # Use the LDA model to predict class probabilities
        probabilities = self.lda.predict_proba(scores)
        return probabilities

    def get_lda_scores(self):
        """
        Get the LDA scores (projections of the data onto the discriminant vectors).
        
        Returns:
            np.ndarray: LDA scores.
        """
        return self.lda_scores

    def get_lda_loadings(self):
        """
        Get the LDA loadings (discriminant vectors).
        
        Returns:
            np.ndarray: LDA loadings.
        """
        return self.lda_loadings





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
    

        