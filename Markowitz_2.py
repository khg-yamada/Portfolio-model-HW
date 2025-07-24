
import numpy as np
import pandas as pd

class MyPortfolio:
    def get_weights(self, returns: pd.DataFrame) -> np.ndarray:
        mu = returns.mean()
        vol = returns.std()
        score = mu / vol
        weights = score / score.sum()
        return weights.values
