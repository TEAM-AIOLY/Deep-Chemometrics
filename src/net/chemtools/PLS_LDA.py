# file: src/net/chemtools/PLS_LDA.py
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class PLS_LDA(BaseEstimator, ClassifierMixin):
    """
    PLS-DA: PLS (X -> scores T) + LDA su T.
    Parametri:
      - n_components: numero di latent variables (LV) per il PLS.
      - scale: passa a PLSRegression (False se già scali nel pipeline).
      - lda_kwargs: dict opzionale per configurare LDA (es. {'solver': 'lsqr'}).

      TODO: aggiungere gestione classi sbilanciate in LDA (class_weight, priors).
            CRITERIO MAX VALUES OF Y SCORES
    """
    def __init__(self, n_components=2, scale=False, lda_kwargs=None):
        self.n_components = n_components
        self.scale = scale
        self.lda_kwargs = lda_kwargs  # <-- DEVE essere assegnato, senza modificarlo

        # inizializzati in fit
        self._enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.pls_ = None
        self.lda_ = None
        self.classes_ = None

    # --------- Fit / Predict ----------
    def fit(self, X, y):
        y = np.asarray(y)
        if y.ndim != 1:
            y = y.ravel()

        # Dummy matrix
        Y = self._enc.fit_transform(y.reshape(-1, 1))

        # Ricrea PLS model
        self.pls_ = PLSRegression(n_components=self.n_components, scale=self.scale)
        self.pls_.fit(X, Y)

        # Scores PLS 
        T = self.pls_.x_scores_

        # LDA for classification on T
        lda_kwargs_local = {} if self.lda_kwargs is None else dict(self.lda_kwargs)  # <-- copia locale

        self.lda_ = LinearDiscriminantAnalysis(**lda_kwargs_local).fit(T, y)

        self.classes_ = self.lda_.classes_
        return self

    def _transform_to_scores(self, X):
        # Obtain PLS scores T from X
        if self.pls_ is None:
            raise RuntimeError("Model not fitted yet.")
        return self.pls_.transform(X)

    def decision_function(self, X):
        T = self._transform_to_scores(X)
        return self.lda_.decision_function(T)

    def predict_proba(self, X):
        T = self._transform_to_scores(X)
        
        return self.lda_.predict_proba(T)

    def predict(self, X):
        # Obtain predictions
        T = self._transform_to_scores(X)
        return self.lda_.predict(T)

