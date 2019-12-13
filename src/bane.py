"""BANE procedure class. """

import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.linalg import inv
from sklearn.decomposition import TruncatedSVD

class BANE(object):
    """
    Binarized Attributed Network Embedding Class (ICDM 2018).
    """
    def __init__(self, args, P, X):
        """
        Setting up a BANE model.
        :param args: Arguments object.
        :param P: Normalized connectivity matrix.
        :param X: Feature matrix.
        """
        self.args = args
        self.P = P
        self.X = X

    def fit(self):
        """
        Creating a BANE embedding.
        1. Running SVD.
        2. Running power iterations and CDC.
        """
        print("\nFitting BANE model.\nBase SVD fitting started.")
        self.fit_base_SVD_model()
        print("SVD completed.\nFitting binary model.\n")
        self.binary_optimize()

    def fit_base_SVD_model(self):
        """
        Reducing the dimensionality with SVD in the 1st step.
        """
        self.P = self.P.dot(self.X)
        self.model = TruncatedSVD(n_components=self.args.dimensions,
                                  n_iter=70,
                                  random_state=42)

        self.model.fit(self.P)
        self.P = self.model.fit_transform(self.P)

    def update_G(self):
        """
        Updating the kernel matrix.
        """
        self.G = np.dot(self.B.transpose(), self.B)
        self.G = self.G + self.args.alpha*np.eye(self.args.dimensions)
        self.G = inv(self.G)
        self.G = self.G.dot(self.B.transpose()).dot(self.P)

    def update_Q(self):
        """
        Updating the rescaled target matrix.
        """
        self.Q = self.G.dot(self.P.transpose()).transpose()

    def update_B(self):
        """
        Updating the embedding matrix.
        """
        for _ in tqdm(range(self.args.approximation_rounds), desc="Inner approximation:"):
            for d in range(self.args.dimensions):
                sel = [x for x in range(self.args.dimensions) if x != d]
                self.B[:, d] = self.Q[:, d]-self.B[:, sel].dot(self.G[sel, :]).dot(self.G[:, d]).transpose()
                self.B[:, d] = np.sign(self.B[:, d])

    def binary_optimize(self):
        """
        Starting 2nd optimization phase with power iterations and CCD.
        """
        self.B = np.sign(np.random.normal(size=(self.P.shape[0], self.args.dimensions)))
        for _ in tqdm(range(self.args.binarization_rounds), desc="Iteration", leave=True):
            self.update_G()
            self.update_Q()
            self.update_B()

    def save_embedding(self):
        """
        Saving the embedding.
        """
        self.out = np.concatenate([np.array(range(self.B.shape[0])).reshape(-1, 1), self.B], axis=1)
        self.out = pd.DataFrame(self.out, columns=["id"]+["x_"+str(d) for d in range(self.args.dimensions)])
        self.out.to_csv(self.args.output_path, index=None)
        print("\n\nModel saved.")
