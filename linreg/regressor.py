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


def normalise(values):
    mean = values.mean()
    std = values.std()
    values_normalied = (values - mean) / std
    return values_normalied, mean, std


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
        self.max_epoch = 1e4
        self.alpha = 0.001
        self.converged = None
        self.expected_y_std = None
        self.cost_precicion_stop = None
        self.learning_summary = pd.DataFrame(columns=["cost", "a", "b", "g1", "g2"], dtype=float)
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
        print("sum of y:", self.y.sum())
        print("mean of y:", self.y.mean())
        print("y divided by 2:", self.y / 2)
        """ Code here """
        ss_xy = (self.y * self.x).sum() - self.x.sum() * self.y.sum() / self.n
        ss_xx = (self.x * self.x).sum() - (self.x.sum() ** 2) / self.n
        self.a = ss_xy / ss_xx
        self.b = self.y.mean() - self.a * self.x.mean()
        self._finish_learning()

    def fit_numeric(self, theta=None):
        print(f"fitting...")
        """ Code here """

        x_norm, xmean, xstd = normalise(self.x.values)
        y_norm, ymean, ystd = normalise(self.y.values)

        theta[0] = theta[0] * xstd / ystd
        theta[1] = (theta[1] + theta[0] * xmean * ystd / xstd - ymean) / ystd

        x_t = np.array([x_norm, np.ones(len(self.x))])
        x = x_t.T
        i = 0

        # if self.expected_y_std is not None and self.cost_precicion_stop is None:
        #     self.cost_precicion_stop = 1.1 * (self.expected_y_std - ymean) ** 2 * len(x) / (2 * len(x) * ystd)
        if self.max_epoch is None and self.cost_precicion_stop is None:
            raise ValueError("At least one of max_iterations or cost_precicion_stop must be set for LinearRegressor if "
                             "using numeric fit.")
        if self.cost_precicion_stop is not None:
            self.converged = False
        while self.max_epoch is None or i < self.max_epoch:
            try:
                i += 1
                hypothesis = np.dot(x, theta)
                loss = hypothesis - y_norm
                cost = np.sum(loss ** 2) / (2 * self.n)
                print(f"Epoch {i} | Cost: {cost}")
                if np.isinf(cost) or np.isnan(cost):
                    raise ValueError("Cost is invalid")
                gradient = np.dot(x_t, loss) / self.n
                theta = theta - self.alpha * gradient
                self.a = theta[0] * ystd / xstd
                self.b = ymean + theta[1] * ystd - xmean * theta[0] * ystd / xstd
                print(f"  a: {self.a} -- b: {self.b}")
                self._check_convergence(cost)
                self.learning_summary.loc[i] = [cost, self.a, self.b, gradient[0], gradient[1]]
                if self.converged:
                    break
            except Exception as e:
                print(f"Stopping the gradient descent because an error occured : {str(e)}")
                break
        if not self.converged:
            print("The learning did not converge")
        self._finish_learning()

    def _finish_learning(self):
        self.pred_y = self.b + self.a * self.x
        self.rsquared = compute_rsquared(self.y, self.pred_y)
        print(f"...done. Results are : \n - a={self.a}\n - b={self.b}\n - r**2={self.rsquared}")

    def _check_convergence(self, cost):
        if self.cost_precicion_stop is not None:
            if cost < self.cost_precicion_stop:
                self.converged = True
                print("Learning converged")

    def plot(self, path: str = None):
        print(f"plotting...")
        plt.scatter(self.x, self.y, color="blue", marker="o", s=10, label="data")
        if self.a is not None:
            plt.plot(
                self.x,
                self.b + self.a * self.x,
                color="red",
                label=f"${round(self.a, 3)}x + {round(self.b, 3)}$\n$r^2={round(self.rsquared, 3)}$",
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
            return plt.gcf()

    def plot_summary(self, path: str = None):
        if self.learning_summary.empty:
            print("Learning summary is empty")
            return
        fig, axis = plt.subplots(2, 2, figsize=(15, 9))
        fig.suptitle("Learning summary", fontsize=25)

        axis[0][0].plot(self.learning_summary.index, self.learning_summary["cost"])
        if self.cost_precicion_stop is not None:
            axis[0][0].hlines(self.cost_precicion_stop, 0, self.learning_summary.index[-1], color="red")
        axis[0][0].set_xlabel("epoch", fontsize=20)
        axis[0][0].set_ylabel("Cost", fontsize=20)

        axis[0][1].plot(self.learning_summary.index, self.learning_summary["g1"])
        twinx = axis[0][1].twinx()
        twinx.plot(self.learning_summary.index, self.learning_summary["g2"], color="red", ls="--")
        axis[0][1].set_xlabel("epoch", fontsize=20)
        axis[0][1].set_ylabel("g1", fontsize=20)
        twinx.set_ylabel("g2", fontsize=20)

        axis[1][0].plot(self.learning_summary.index, self.learning_summary["a"])
        axis[1][0].set_xlabel("epoch", fontsize=20)
        axis[1][0].set_ylabel("a", fontsize=20)

        axis[1][1].plot(self.learning_summary.index, self.learning_summary["b"])
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
