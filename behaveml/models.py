""" Basic video tracking and behavior class that houses data """

import numpy as np
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.metrics import f1_score

def _logit(p):
    return np.log(p / (1 - p))

def _sample_prob_simplex(n=4):
    x = sorted(np.append(np.random.uniform(size = n-1), [0,1]))
    y = np.diff(np.array(x))
    return y

try:
    import ssm 
    class HMMSklearn(ssm.HMM):

        def __init__(self, D, C = 11):
            """HMM model from Linderman state-space model package ssm, tweaked slightly to fit with sklearn syntax
            
            Args:
                D: number of behavioral categories
                C: number of bins to discretize
            """

            self.D = D 
            self.C = C
            super().__init__(D, D + 1, observations = 'categorical', observation_kwargs = {'C': C})

        def fit(self, X, y):

            preds = np.argmax(X, axis = -1)
            X = np.hstack(((X*(self.C-1)).astype(int), np.atleast_2d((preds).astype(int)).T))

            N = len(y)
            transition_matrix = np.ones((self.D, self.D))

            for idx in range(N):
                if idx == 0: continue
                transition_matrix[y[idx-1], y[idx]] += 1
            
            for j in range(self.D):
                transition_matrix[j] /= np.sum(transition_matrix[j]) 

            self.transitions.params = [np.log(transition_matrix)]

            emission_dist = np.ones((self.D, self.D+1, self.C))
            for i in range(self.D):
                for j in range(self.D+1):
                    for k in range(self.C):
                        ct = np.sum(X[(y == i),j] == k)
                        emission_dist[i, j, k] = max(1, ct)
                    emission_dist[i,j,:] /= np.sum(emission_dist[i,j,:])

            self.observations.params = _logit(emission_dist)

        def predict(self, X):
            preds = np.argmax(X, axis = -1)
            X = np.hstack(((X*(self.C-1)).astype(int), np.atleast_2d((preds).astype(int)).T))
            return self.most_likely_states(X)

except ImportError:
    print("Couldn't find ssm module. HMMSklearn model not available. See install instructions: https://github.com/lindermanlab/ssm")

class F1Optimizer(ClassifierMixin):
    def __init__(self, N = 1000, labels = [1]):
        self.N = N
        self.labels = labels

    def fit(self, X, y): #train_labels, train_pred_prob):

        self.dim_x = X.shape[1]

        f = lambda w: f1_score(y, np.argmax((X*w), axis = -1))#, average = 'macro', labels = self.labels)

        w_star = np.ones(self.dim_x)/self.dim_x
        f_star = 0

        for _ in range(self.N):
            w = _sample_prob_simplex(n = self.dim_x)
            f_curr = f(w)
            if f_curr > f_star:
                w_star = w
                f_star = f_curr

        self.w_star = w_star
        self.f_star = f_star

    def predict(self, X):
        return np.argmax(X*self.w_star, axis = -1)

    def predict_proba(self, X):
        return X*self.w_star

    def transform(self, X):
        return self.predict_proba(X)

    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X)

class ModelTransformer(ClassifierMixin):
    def __init__(self, Model, *args, **kwargs):
        self.model = Model(*args, **kwargs)    
        
    def fit(self, X, y):
        self.model.fit(X, y)    
        
    def transform(self, X):
        return self.model.predict_proba(X)
                
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)