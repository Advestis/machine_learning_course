import pandas as pd
from regressor import LinearRegressor

if __name__ == '__main__':
    df = pd.read_csv("data.csv")
    x = df["x"]
    y = df["y"]
    regressor = LinearRegressor(x, y)
    regressor.fit()
    regressor.plot("fit_result.pdf")
