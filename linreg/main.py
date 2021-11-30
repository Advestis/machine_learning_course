import pandas as pd
import numpy as np
from regressor import LinearRegressor

if __name__ == '__main__':
    df = pd.read_csv("data.csv")
    x = df["x"]
    y = df["y"]

    expected_stddev = 60

    step = 100
    nbins = 20

    regressor = LinearRegressor(x, y)
    regressor.fit("analytic")
    regressor.plot("fit_result_analytic.pdf")
    regressor.max_epoch = 4000

    regressor.fit("numeric", theta=np.array([1., 1.]))
    regressor.plot("fit_result_numeric.pdf")
    regressor.save_summary("fit_result_numeric_summary.csv")
    regressor.save_summary("fit_result_numeric_summary_normalized.csv", "normalized")
    regressor.plot_summary("fit_result_numeric_summary.pdf")
    regressor.plot_summary("fit_result_numeric_summary_normalized.pdf", "normalized")
    regressor.animate("fit_animation.gif", step=step, nbins=nbins)
