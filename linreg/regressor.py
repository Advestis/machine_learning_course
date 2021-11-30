import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def check_expected_kwargs(kwargs, expected, allow_unexpected: bool = True):
    for e in expected:
        if e not in kwargs:
            raise ValueError(f"Argument '{e}' is required.")
    if not allow_unexpected:
        for k in kwargs:
            if k not in expected:
                raise ValueError(f"Unexpected argument '{k}'")


def compute_rsquared(y, pred_y):
    rss = ((y - pred_y) ** 2).sum()
    tss = ((y - y.mean()) ** 2).sum()
    return 1 - (rss / tss)


class LinearRegressor:
    def __init__(self, x: pd.Series, y: pd.Series):

        if len(x) != len(y):
            raise ValueError("x and y must have the same size")

        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        self.x = x
        self.y = y
        self.pred_y = None
        self.n = len(x)
        print(f"There are {self.n} observations")
        self.rsquared = None
        self.a = None
        self.b = None
        self.max_epoch = 1e4
        self.learning_rate = 0.001
        self.converged = None
        self.expected_y_std = None
        self.stopping_criterion = None
        self.learning_summary = pd.DataFrame(columns=["loss", "a", "b", "ga", "gb"], dtype=float)
        self.learning_summary.index.name = "epoch"

    def fit(self, method="analytic", **kwargs):
        if method == "analytic":
            self.fit_analytic()
        elif method == "numeric":
            check_expected_kwargs(kwargs, ["theta"], allow_unexpected=False)
            self.fit_numeric(**kwargs)
        else:
            raise ValueError(f"Invalid method '{method}'")

    def fit_analytic(self):
        print(f"fitting...")
        """ Code here """
        self._finish_learning()

    def fit_numeric(self, theta=None):
        def one_fit(a_, b_, x_, y_):
            y_model =
            diffs =
            loss =
            if np.isinf(loss) or np.isnan(loss):
                raise ValueError("loss is invalid")
            g_a =
            g_b =
            return loss, g_a, g_b

        print(f"fitting...")
        """ Code here """

        i = 0
        a, b = theta

        if self.max_epoch is None:
            raise ValueError("max_epoch can not be None")
        while self.max_epoch is None or i < self.max_epoch:
            try:
                self.a =
                self.b =
                loss, g_a, g_b =
                print(f"  a: {self.a} -- b: {self.b}")
                print(f"Epoch {i} | loss {loss}")
                if np.isinf(loss) or np.isnan(loss):
                    raise ValueError("loss is invalid")

                self.learning_summary.loc[i] = [loss, self.a, self.b, g_a, g_b]
                i =
                a =
                b =
            except Exception as e:
                print(f"Stopping the gradient descent because an error occured : {str(e)}")
                break
        self._finish_learning()

    def _finish_learning(self):
        self.pred_y = self.b + self.a * self.x
        self.rsquared = compute_rsquared(self.y, self.pred_y)
        print(f"...done. Results are : \n - a={self.a}\n - b={self.b}\n - r**2={self.rsquared}")


    def plot(self, path: str = None):
        print(f"plotting...")
        plt.scatter(self.x, self.y, color="blue", marker="o", s=10, label="data")
        if self.a is not None:
            plt.plot(
                self.x,
                self.b + self.a * self.x,
                color="red",
                label=f"{round(self.a, 3)}x + {round(self.b, 3)}\nr2={round(self.rsquared, 3)}",
            )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc="upper left")
        plt.grid()
        if path is not None:
            plt.savefig(path)
            print(f"...figure saved in {path}")
            plt.close("all")  # Without this command, all future plots will be made in the same figure
        else:
            return plt.gcf()  # gcf means "get current figure"

    def plot_summary(self, path: str = None):
        if self.learning_summary.empty:
            print("Learning summary is empty")
            return

        df = self.learning_summary

        fig, axis = plt.subplots(2, 2, figsize=(15, 9))
        fig.suptitle("Learning summary", fontsize=25)

        axis[0][0].plot(df.index, df["loss"])
        axis[0][0].set_xlabel("epoch", fontsize=20)
        axis[0][0].set_ylabel("loss", fontsize=20)

        axis[0][1].plot(df.index, df["ga"])
        twinx = axis[0][1].twinx()
        twinx.plot(df.index, df["gb"], color="red", ls="--")
        axis[0][1].set_xlabel("epoch", fontsize=20)
        axis[0][1].set_ylabel("ga", fontsize=20)
        twinx.set_ylabel("gb", fontsize=20)

        axis[1][0].plot(df.index, df["a"])
        axis[1][0].set_xlabel("epoch", fontsize=20)
        axis[1][0].set_ylabel("a", fontsize=20)

        axis[1][1].plot(df.index, df["b"])
        axis[1][1].set_xlabel("epoch", fontsize=20)
        axis[1][1].set_ylabel("b", fontsize=20)

        axis[0][0].grid()
        axis[0][1].grid()
        axis[1][0].grid()
        axis[1][1].grid()

        if path is not None:
            fig.savefig(path)
            print(f"...figure saved in {path}")
            plt.close("all")  # Without this command, all future plots will be made in the same figure
        else:
            return fig

    def save_summary(self, path):
        self.learning_summary.to_csv(path)
