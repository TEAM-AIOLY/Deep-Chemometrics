import torch
from net.chemtools.metrics import ccc,r2_score
from sklearn.preprocessing import StandardScaler


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
    def __init__(self):
        self.W = []  # Store projection vectors for each class
        self.class_thresholds = []  # Store thresholds for each class (binary classification)
        self.device = torch.device("cpu")

    def _to_tensor(self, arr):
        """Convert numpy array to tensor if not already a tensor."""
        if isinstance(arr, torch.Tensor):
            return arr.detach().clone()
        return torch.tensor(arr, dtype=torch.float32)

    def fit(self, X, Y):
        """
        Fit the LDA model for each class in Y.
        Each column of Y is treated as a separate binary classification problem.
        """
        X = self._to_tensor(X).to(self.device)
        Y = self._to_tensor(Y).to(self.device)

        N, D = X.shape  # N: number of samples, D: number of features
        _, C = Y.shape  # C: number of classes

        self.W = []
        self.class_thresholds = []

        # Iterate over each class (binary classification for that class)
        for c in range(C):
            self._fit_single_class(X, Y[:, c], c)

        return self

    def _fit_single_class(self, X, y_class, class_idx):
        """
        Fit the LDA model for a single class (binary classification for that class).
        This includes calculating the within-class scatter matrix and the projection matrix.
        """
        N, D = X.shape  # N: number of samples, D: number of features

        # Separate the samples based on class labels (binary: 1 or 0)
        class_samples_pos = X[y_class == 1]
        class_samples_neg = X[y_class == 0]

        # Calculate class means
        mean_pos = class_samples_pos.mean(dim=0)
        mean_neg = class_samples_neg.mean(dim=0)

        # Compute the within-class scatter matrix (Sw)
        Sw = torch.zeros(D, D, device=self.device)
        for x in class_samples_pos:
            diff = (x - mean_pos).unsqueeze(1)
            Sw += diff @ diff.T
        for x in class_samples_neg:
            diff = (x - mean_neg).unsqueeze(1)
            Sw += diff @ diff.T

        # Regularize Sw (add small value to diagonal)
        Sw += 1e-6 * torch.eye(D, device=self.device)

        # Compute the between-class scatter matrix (Sb)
        mean_all = X.mean(dim=0)
        diff_pos = (mean_pos - mean_all).unsqueeze(1)
        diff_neg = (mean_neg - mean_all).unsqueeze(1)
        Sb = (diff_pos @ diff_pos.T) + (diff_neg @ diff_neg.T)

        # Solve the generalized eigenvalue problem Sw^{-1} Sb
        eigvals, eigvecs = torch.linalg.eig(torch.linalg.pinv(Sw) @ Sb)
        eigvals = eigvals.real
        eigvecs = eigvecs.real

        # Sort eigenvectors by eigenvalues (descending order)
        idx = torch.argsort(eigvals, descending=True)

        # Store the projection vector for this class (we use the top eigenvector)
        self.W.append(eigvecs[:, idx[0]].unsqueeze(1))  # One vector per class

        # Calculate the threshold: mean of the projected positive samples
        Z_pos = (class_samples_pos @ self.W[-1]).squeeze()
        if Z_pos.numel() == 0:
            threshold = float('inf')
        else:
            threshold = Z_pos.mean()  # Threshold as the mean of positive class projection
        self.class_thresholds.append(threshold)

    def transform(self, X):
        """
        Transform the input data X using the learned projection for each class.
        Returns the transformed data for each class as a list.
        """
        X = self._to_tensor(X).to(self.device)
        transformed_data = []

        for W in self.W:
            Z = X @ W  # Project the data using the class-specific projection matrix
            transformed_data.append(Z)

        return torch.cat(transformed_data, dim=1)  # Concatenate projections for all classes

    def predict(self, X):
        """
        Predict the multi-label output using thresholds learned during fit.
        Each class is treated as a separate binary classification problem.
        """
        X = self._to_tensor(X).to(self.device)
        predictions = []

        # Iterate over each class (column)
        for i, W in enumerate(self.W):
            Z = X @ W  # Project the data
            pred = (Z.squeeze() >= self.class_thresholds[i]).float()  # Apply threshold
            predictions.append(pred.unsqueeze(1))  # Keep predictions for this class

        # Stack predictions for all classes
        return torch.cat(predictions, dim=1)



class MLDA:
    def __init__(self):
        self.projections = None  # List of weight vectors
        self.device = torch.device("cpu")

    def _to_tensor(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().clone()
        return torch.tensor(arr, dtype=torch.float32)

    def fit(self, X, Y):
        """
        Fit one-vs-all LDA projections for multilabel data.
        """
        X = self._to_tensor(X)
        Y = self._to_tensor(Y)

        if Y.ndim != 2:
            raise ValueError("Y must be a 2D binary label matrix.")

        self.device = X.device
        N, D = X.shape
        _, C = Y.shape
        projections_list = []

        for c in range(C):
            pos_mask = Y[:, c] == 1
            neg_mask = Y[:, c] == 0

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                self.projections.append(torch.zeros(D, device=self.device))
                continue

            X_pos = X[pos_mask]
            X_neg = X[neg_mask]

            mu_pos = X_pos.mean(dim=0)
            mu_neg = X_neg.mean(dim=0)

            X_pos_c = X_pos - mu_pos
            X_neg_c = X_neg - mu_neg

            Sw = (
                X_pos_c.T @ X_pos_c +
                X_neg_c.T @ X_neg_c +
                1e-6 * torch.eye(D, device=self.device)
            )

            mean_diff = (mu_pos - mu_neg).unsqueeze(1)
            w = torch.linalg.pinv(Sw) @ mean_diff
            # w = w / torch.norm(w)
            projections_list.append(w.squeeze())
        self.projections = torch.stack(projections_list) 

        return self

    def transform(self, X):
        """
        Project X using each class-wise LDA direction.
        Returns tensor of shape [N, C]
        """
        X = self._to_tensor(X).to(self.device)
        if self.projections is None:
            raise RuntimeError("Call .fit() before .transform().")

        # return torch.stack([X @ w for w in self.projections], dim=1)
        W = self.projections  # shape [C, D]
        return X @ W.T 

    def predict(self, X, threshold=0.0):
        """
        Predict multilabel output: 1 if projection > threshold else 0.
        """
        projections = self.transform(X)
        return (projections > threshold).int()
    
    
    
    

class MLDA_v2:
    def __init__(self):
        self.projections = None  # Projection matrix [C, D]
        self.device = torch.device("cpu")

    def _to_tensor(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().clone().to(self.device)
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def fit(self, X, Y):
        """
        Fit MLDA using direct generalized eigen-solver:
        eigvals, eigvecs = torch.linalg.eig(Sb, Sw)
        """
        X = self._to_tensor(X)
        Y = self._to_tensor(Y)

        if Y.ndim != 2:
            raise ValueError("Y must be a 2D binary label matrix.")

        N, D = X.shape
        _, C = Y.shape

        # Compute class means μ_c
        mu_c = []
        for c in range(C):
            mask = Y[:, c] == 1
            mu = X[mask].mean(dim=0) if mask.sum() > 0 else torch.zeros(D, device=self.device)
            mu_c.append(mu)
        mu_c = torch.stack(mu_c)  # [C, D]

        # Overall mean μ
        mu = X.mean(dim=0)  # [D]

        # Between-class scatter Sb
        Sb = torch.zeros((D, D), device=self.device)
        for c in range(C):
            diff = (mu_c[c] - mu).unsqueeze(1)
            Sb += diff @ diff.T

        # Within-class scatter Sw
        Sw = torch.zeros((D, D), device=self.device)
        for i in range(N):
            for c in range(C):
                if Y[i, c] == 1:
                    diff = (X[i] - mu_c[c]).unsqueeze(1)
                    Sw += diff @ diff.T
        Sw += 1e-6 * torch.eye(D, device=self.device)

        # Direct generalized eigen-solve
        eigvals, eigvecs = torch.linalg.eig(Sb, Sw)

        # Sort by descending real parts
        real_vals = eigvals.real
        idx = torch.argsort(real_vals, descending=True)
        eigvecs = eigvecs.real[:, idx].float()

        # Store top-C projections
        self.projections = eigvecs[:, :C].T  # [C, D]

        return self

    def transform(self, X):
        X = self._to_tensor(X)
        if self.projections is None:
            raise RuntimeError("Call .fit() before .transform().")
        return X @ self.projections.T  # [N, C]

    def predict(self, X, threshold=0.0):
        return (self.transform(X) > threshold).int()



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
    

        
        
    
# class LDA:
#     def __init__(self):
#         self.num_classes = None
#         self.means = None
#         self.weights = None
#         self.device = torch.device("cpu")
#         self.scaler = StandardScaler() 
    
#     def _to_tensor(self, arr):
#         """Convert numpy array to tensor if not already a tensor."""
#         if isinstance(arr, torch.Tensor):
#             return arr.detach().clone().to(dtype=torch.float32)
#         return torch.tensor(arr, dtype=torch.float32)

#     def fit(self, X, Y):
#         """
#         Fit the LDA model to the data using eigen decomposition.
        
#         Args:
#             X: Tensor of shape (n_samples, n_features) with the input data.
#             Y: Tensor of shape (n_samples,) with the class labels (one-hot encoded).
#         """
#         X = self._to_tensor(X).to(self.device)
#         Y = self._to_tensor(Y).to(self.device)

#         N, D = X.shape  # N: number of samples, D: number of features
#         _, C = Y.shape  # C: number of classes (one-hot encoded)
#         self.num_classes=C
        
#         X = torch.tensor(self.scaler.fit_transform(X.cpu().numpy()), dtype=torch.float32, device=self.device)

#         # Compute the mean for each class
#         means = torch.zeros((C, D), dtype=torch.float32, device=self.device)
#         for i in range(C):
#             means[i] = torch.mean(X[Y[:, i] == 1], dim=0)

#         # Compute the within-class scatter matrix (Sw)
#         Sw = torch.zeros((D, D), dtype=torch.float32, device=self.device)
#         for i in range(C):
#             class_data = X[Y[:, i] == 1]
#             diff = class_data - means[i]
#             Sw += diff.T @ diff

#         b = torch.zeros((D, D), dtype=torch.float32, device=self.device)

#         # Compute the overall mean of the dataset
#         overall_mean = torch.mean(X, dim=0)

#         # Loop through each class
#         for i in range(C):
#             # Compute the difference between the class mean and the overall mean
#             mean_diff = means[i] - overall_mean  # This is a (D,) vector
#             # Update the between-class scatter matrix using matrix multiplication
#             Sb += torch.sum(Y[:, i] == 1).item() * (mean_diff.unsqueeze(1) @ mean_diff.unsqueeze(0))

#         # Solve the generalized eigenvalue problem: Sw^-1 * Sb
#         eigvals, eigvecs = torch.linalg.eig(Sb, Sw)
#         eigvals = eigvals.real
#         eigvecs = eigvecs.real

#         # Sort the eigenvectors by eigenvalues in descending order and take the eigenvectors with the largest eigenvalues
#         sorted_indices = torch.argsort(eigvals, descending=True)
#         self.weights = eigvecs[:, sorted_indices][:,:C-1]
#         self.weights =(self.weights).float()
    

#         self.means = means
#         self.Sw = Sw
#         self.Sb = Sb

#     def transform(self, X):
#         """
#         Project the data onto the LDA space.
        
#         Args:
#             X: Tensor of shape (n_samples, n_features) with the input data to transform.
        
#         Returns:
#             Projected data of shape (n_samples, n_classes-1).
#         """
#         X = self._to_tensor(X).to(self.device)
#         X = torch.tensor(self.scaler.transform(X.cpu().numpy()), dtype=torch.float32, device=self.device)
#         return X @ self.weights # Project onto the LDA space

#     def predict(self, X):
#         X_proj = self.transform(X)  # (n_samples, n_components)

#         # Project the class means into the same LDA space
#         means_proj = self.means @ self.weights  # (n_classes, n_components)

#         distances = torch.cdist(X_proj, means_proj)
#         pred_classes = torch.argmin(distances, dim=1)
#         one_hot_preds = torch.nn.functional.one_hot(pred_classes, num_classes=self.num_classes).float()
#         return one_hot_preds


class QDA:
    def __init__(self):
        self.num_classes = None
        self.means = None
        self.covariances = None
        self.priors = None
        self.device = torch.device("cpu")

    def _to_tensor(self, arr):
        """Convert numpy array to tensor if not already a tensor."""
        if isinstance(arr, torch.Tensor):
            return arr.detach().clone().to(dtype=torch.float32)
        return torch.tensor(arr, dtype=torch.float32)

    def fit(self, X, Y):
        """
        Fit the QDA model to the data by calculating the class-wise means, covariances, and priors.
        
        Args:
            X: Tensor of shape (n_samples, n_features) with the input data.
            Y: Tensor of shape (n_samples, n_classes) with one-hot encoded class labels.
        """
        X = self._to_tensor(X).to(self.device)
        Y = self._to_tensor(Y).to(self.device)

        N, D = X.shape  # N: number of samples, D: number of features
        _, C = Y.shape  # C: number of classes (one-hot encoded)
        
        self.num_classes = C

        # Initialize placeholders for class means, covariances, and priors
        self.means = torch.zeros((C, D), dtype=torch.float32, device=self.device)
        self.covariances = torch.zeros((C, D, D), dtype=torch.float32, device=self.device)
        self.priors = torch.zeros(C, dtype=torch.float32, device=self.device)

        # Loop through each class to calculate means, covariances, and priors
        for i in range(C):
            class_data = X[Y[:, i] == 1]  # Get the data points for class i
            self.means[i] = torch.mean(class_data, dim=0)
            self.covariances[i] = torch.cov(class_data.T)  # Covariance matrix for class i
            self.priors[i] = class_data.shape[0] / N  # Prior probability of class i

    def transform(self, X):
        """
        Project the data into the QDA space. For QDA, transformation isn't a direct projection
        like in LDA, but we still need a consistent method to handle the data.

        Args:
            X: Tensor of shape (n_samples, n_features) with the input data to transform.

        Returns:
            Transformed data (n_samples, n_classes) based on QDA decision function.
        """
        X = self._to_tensor(X).to(self.device)
        N = X.shape[0]
        
        # Initialize a tensor to store the discriminant function values for each class
        scores = torch.zeros((N, self.num_classes), dtype=torch.float32, device=self.device)
        
        for k in range(self.num_classes):
            mean = self.means[k]
            cov = self.covariances[k]
            prior = self.priors[k]

            # Calculate the discriminant function for class k
            diff = X - mean
            inv_cov = torch.pinverse(cov)  # Inverse of the covariance matrix
            log_det_cov = torch.logdet(cov)  # Log determinant of covariance

            # Compute the quadratic term (x - mean)^T * inv(S) * (x - mean)
            quadratic_term = torch.sum(diff @ inv_cov * diff, dim=1)
            scores[:, k] = -0.5 * log_det_cov - 0.5 * quadratic_term + torch.log(prior)

        return scores

    def predict(self, X):
        """
        Predict the class labels for the given data X using the QDA model.
        
        Args:
            X: Tensor of shape (n_samples, n_features) with the input data to classify.
        
        Returns:
            Predicted class labels as a tensor.
        """
        scores = self.transform(X)  # Get the discriminant function scores
        predicted_classes = torch.argmax(scores, dim=1)  # Assign the class with the highest score
        return predicted_classes

    def predict_proba(self, X):
        """
        Predict the class probabilities for the given data X using the QDA model.
        
        Args:
            X: Tensor of shape (n_samples, n_features) with the input data to classify.
        
        Returns:
            Predicted class probabilities.
        """
        scores = self.transform(X)  # Get the discriminant function scores
        exp_scores = torch.exp(scores)  # Apply exponentiation to the scores
        probs = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)  # Softmax to get probabilities
        return probs




