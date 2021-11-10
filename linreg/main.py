import pandas as pd
import numpy as np
from regressor import LinearRegressor

if __name__ == '__main__':
    df = pd.read_csv("data.csv")
    x = df["x"]
    y = df["y"]

    expected_stddev = 60

    regressor = LinearRegressor(x, y)
    regressor.alpha = 0.00000005
    regressor.fit("analytic")
    regressor.plot("fit_result_analytic.pdf")
    regressor.expected_y_std = 60
    regressor.fit("numeric", theta=np.array([1., 1.]))
    regressor.plot("fit_result_numeric.pdf")
    regressor.plot_summary("fit_result_numeric_summary.pdf")
