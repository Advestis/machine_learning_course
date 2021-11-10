import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
        self.loss_precicion_stop = None
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

        a = theta[0] * xstd / ystd
        b = (theta[1] + a * xmean * ystd / xstd - ymean) / ystd

        i = 0

        # if self.expected_y_std is not None and self.loss_precicion_stop is None:
        #     self.loss_precicion_stop = 1.1 * (self.expected_y_std - ymean) ** 2 * len(x) / (2 * len(x) * ystd)
        if self.max_epoch is None and self.loss_precicion_stop is None:
            raise ValueError(
                "At least one of max_iterations or loss_precicion_stop must be set for LinearRegressor if "
                "using numeric fit."
            )
        if self.loss_precicion_stop is not None:
            self.converged = False
        while self.max_epoch is None or i < self.max_epoch:
            try:
                i += 1
                y_model = a * x_norm + b
                diffs = y_norm - y_model
                loss = np.sum(diffs ** 2) / (2 * self.n)
                print(f"Epoch {i} | loss: {loss}")
                if np.isinf(loss) or np.isnan(loss):
                    raise ValueError("loss is invalid")
                g_a = (-1 / self.n) * (x_norm * diffs).sum()
                g_b = (-1 / self.n) * diffs.sum()
                a = a - self.alpha * g_a
                b = b - self.alpha * g_b
                self.a = a * ystd / xstd
                self.b = ymean + b * ystd - xmean * a * ystd / xstd
                print(f"  a: {self.a} -- b: {self.b}")
                self._check_convergence(loss)
                self.learning_summary.loc[i] = [loss, self.a, self.b, g_a, g_b]
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

    def _check_convergence(self, loss):
        if self.loss_precicion_stop is not None:
            if loss < self.loss_precicion_stop:
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
            return plt.gcf()

    def plot_summary(self, path: str = None):
        if self.learning_summary.empty:
            print("Learning summary is empty")
            return
        fig, axis = plt.subplots(2, 2, figsize=(15, 9))
        fig.suptitle("Learning summary", fontsize=25)

        axis[0][0].plot(self.learning_summary.index, self.learning_summary["loss"])
        if self.loss_precicion_stop is not None:
            axis[0][0].hlines(self.loss_precicion_stop, 0, self.learning_summary.index[-1], color="red")
        axis[0][0].set_xlabel("epoch", fontsize=20)
        axis[0][0].set_ylabel("loss", fontsize=20)

        axis[0][1].plot(self.learning_summary.index, self.learning_summary["ga"])
        twinx = axis[0][1].twinx()
        twinx.plot(self.learning_summary.index, self.learning_summary["gb"], color="red", ls="--")
        axis[0][1].set_xlabel("epoch", fontsize=20)
        axis[0][1].set_ylabel("ga", fontsize=20)
        twinx.set_ylabel("gb", fontsize=20)

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

    def save_summary(self, path):
        self.learning_summary.to_csv(path)

    def animate(self, path: str = None, real_a: float = None, real_b: float = None, step=20, nbins=1000):

        def force_aspect(ax, aspect):
            im = ax.get_images()
            extent = im[0].get_extent()
            ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)

        def make_range(which, real):
            min_ = min(min(self.learning_summary[which]), real)
            max_ = max(max(self.learning_summary[which]), real)

            dist_to_min = abs(real - min_)
            dist_to_max = abs(real - max_)

            if dist_to_min > dist_to_max:
                max_ = real + dist_to_min
            else:
                min_ = real - dist_to_max
            return [min_, max_]

        if real_a is None:
            real_a = self.a
        if real_b is None:
            real_b = self.b

        a_range = make_range("a", real_a)
        b_range = make_range("b", real_b)

        gradient_map = pd.DataFrame(
            index=pd.MultiIndex.from_product([np.linspace(*a_range, nbins), np.linspace(*b_range, nbins)]),
            columns=self.x,
        ).fillna(1)
        slopes = gradient_map.index.get_level_values(0)
        intercepts = gradient_map.index.get_level_values(1)

        gradient_map = gradient_map * gradient_map.columns
        gradient_map = gradient_map.apply(lambda x: slopes * x + intercepts)
        real_y = pd.Series(index=gradient_map.columns, data=real_a * gradient_map.columns + real_b)
        gradient_map = (((gradient_map - real_y) ** 2).sum(axis=1) / (2 * self.n)).unstack()

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        p = axes[0].scatter(self.x, self.y, color="blue", marker="o", s=10)
        a = self.learning_summary.iloc[0]["a"]
        b = self.learning_summary.iloc[0]["b"]
        line = axes[0].plot(
            self.x,
            a * self.x + b,
            color="red",
        )[0]
        tb = f" + {round(self.b, 3)}" if self.b > 0 else f"-{round(self.b, 3)}"
        legend = plt.text(0.1, 0.95, f"Epoch 0: {round(self.a, 3)}x{tb}", transform=axes[0].transAxes)

        colormap = axes[1].imshow(
            gradient_map.values,
            cmap="hot",
            # interpolation="spline16",
            extent=[gradient_map.index[0], gradient_map.index[-1], gradient_map.columns[0], gradient_map.columns[-1]],
            aspect=1,
            origin="lower"
        )
        fig.colorbar(colormap, ax=axes[1], location='right', shrink=0.8)
        force_aspect(axes[1], aspect=1)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[1].set_xlabel("a")
        axes[1].set_ylabel("b")
        axes[1].scatter(
            [self.learning_summary["a"].iloc[0]], [self.learning_summary["b"].iloc[0]], marker="x", color="cyan", s=10
        )
        plt.grid()

        def update(i):
            print(f"Animating {i}...")
            a_ = self.learning_summary.loc[i, "a"]
            b_ = self.learning_summary.loc[i, "b"]
            y = a_ * self.x + b_
            line.set_ydata(y)
            tb_ = f" + {round(b_, 3)}" if b_ > 0 else f"-{round(b_, 3)}"
            legend.set_text(f"Epoch {i}: {round(a_, 3)}x{tb_}")
            axes[1].scatter(
                [self.learning_summary["a"].iloc[i]], [self.learning_summary["b"].iloc[i]], marker="x", color="cyan",
                s=10
            )
            return p, line, legend

        # noinspection PyTypeChecker
        ani = animation.FuncAnimation(fig, update, blit=True, frames=self.learning_summary.index[1::step])

        if path is not None:
            ani.save(path, fps=60)
            print(f"...animation saved in {path}")
            plt.close("all")  # Without this command, all future plots will be made in the same figure
        else:
            return ani
