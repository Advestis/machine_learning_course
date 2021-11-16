import numpy as np
import pandas as pd

if __name__ == '__main__':
    nsamples = 1000
    a = 10
    b = 15
    xstart = -10
    xend = 10
    noise = 15
    x = np.linspace(xstart, xend, nsamples)
    y = a * x + b + np.random.normal(0, noise, nsamples)
    df = pd.DataFrame(columns=["x", "y"], data=np.array([x, y]).T)
    # noinspection PyTypeChecker
    df.to_csv("data.csv")
