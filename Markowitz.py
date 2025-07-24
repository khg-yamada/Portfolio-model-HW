
import numpy as np
import pandas as pd

class EqualWeightPortfolio:
    def get_weights(self, returns: pd.DataFrame) -> np.ndarray:
        m = returns.shape[1]
        return np.ones(m) / m

class RiskParity:
    def get_weights(self, returns: pd.DataFrame) -> np.ndarray:
        vol = returns.std()
        inv_vol = 1 / vol
        weights = inv_vol / inv_vol.sum()
        return weights.values

class MeanVariancePortfolio:
    def __init__(self, gamma: float = 10.0):
        self.gamma = gamma

    def get_weights(self, returns: pd.DataFrame) -> np.ndarray:
        mu = returns.mean().values
        Sigma = returns.cov().values
        m = len(mu)

        import cvxpy as cp
        w = cp.Variable(m)
        ret = mu @ w
        risk = cp.quad_form(w, Sigma)
        prob = cp.Problem(cp.Maximize(ret - self.gamma / 2 * risk),
                          [cp.sum(w) == 1, w >= 0])
        prob.solve()
        return w.value
