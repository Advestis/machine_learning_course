import numpy as np
import pandas as pd

if __name__ == '__main__':
    nsamples = 1000
    a = 10
    b = -1000
    xstart = -20
    xend = 300
    noise = 60
    x = np.linspace(xstart, xend, nsamples)
    y = a * x + b + np.random.normal(0, noise, nsamples)
    df = pd.DataFrame(columns=["x", "y"], data=np.array([x, y]).T)
    # noinspection PyTypeChecker
    df.to_csv("data.csv")
