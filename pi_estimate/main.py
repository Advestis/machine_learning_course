import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import math


def is_in_circle(x_):
    return np.sqrt(((x_-0.5) ** 2).sum(axis=1)) < 0.5


def estimate(n_):
    x = np.random.uniform(0, 1, size=(n_, 2))
    in_circle = x[is_in_circle(x)]
    pi_ = 4 * in_circle.shape[0] / n_
    return pi_


colname = "estimation of $\\pi$"

tqdm.pandas()
estimation = pd.DataFrame(columns=["n"])
estimation["n"] = np.arange(1000, 1000000, 1000).astype(int)
pis = estimation["n"].progress_apply(lambda x: estimate(x))
pis.name = colname
estimation = pd.concat([estimation, pis], axis=1)
estimation = estimation.set_index("n")
print(f"Final estimation (n={estimation.index[-1]}): {estimation.iloc[-1][colname]}")
estimation.plot()
plt.hlines(math.pi, estimation.index[0], estimation.index[-1], color="black", label="Real value of $\\pi$")
plt.legend()
plt.gcf().savefig("plot.pdf")
