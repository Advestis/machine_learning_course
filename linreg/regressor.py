import pandas as pd
import matplotlib.pyplot as plt


class LinearRegressor:
    def __init__(self, x: pd.Series, y: pd.Series):

        if len(x) != len(y):
            raise ValueError("x and y must have the same size")

        self.x = x
        self.y = y
        self.pred_y = None
        self.n = len(x)
        print(f"There are {self.n} observations")
        self.rsquared = None
        self.a = None
        self.b = None

    def fit(self):
        print(f"fitting...")
        ss_xy = (self.y * self.x).sum() - self.x.sum() * self.y.sum() / self.n
        ss_xx = (self.x * self.x).sum() - (self.x.sum() ** 2) / self.n
        self.b = ss_xy / ss_xx
        self.a = self.y.mean() - self.b * self.x.mean()

        self.pred_y = self.a + self.b * self.x
        rss = ((self.y - self.pred_y) ** 2).sum()
        tss = ((self.y - self.y.mean()) ** 2).sum()
        self.rsquared = 1 - (rss / tss)
        print(f"...done. Results are : \n - a={self.a}\n - a={self.b}\n - r**2={self.rsquared}")

    def plot(self, path: str = None):
        print(f"plotting...")
        plt.scatter(self.x, self.y, color="blue", marker="o", s=30, label="data")
        if self.a is not None:
            plt.plot(
                self.x,
                self.a + self.b * self.x,
                color="red",
                label=f"${round(self.b, 3)}x + {round(self.a, 3)}$\n$r^2={round(self.rsquared, 3)}$",
            )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc="upper left")
        if path is not None:
            plt.savefig(path)
        print(f"...figure saved in {path}")
