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

        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        self.x = x
        self.x_norm, self.xmean, self.xstd = normalise(self.x)
        self.y = y
        self.y_norm, self.ymean, self.ystd = normalise(self.y)
        self.pred_y = None
        self.n = len(x)
        print(f"There are {self.n} observations")
        self.rsquared = None
        self.a = None
        self.b = None
        self.max_epoch = 1e4
        self.learning_rate = 0.001
        self.expected_y_std = None
        self.stopping_criterion = None
        self.learning_summary = pd.DataFrame(columns=["loss", "a", "b", "ga", "gb"], dtype=float)
        self.learning_summary_norm = pd.DataFrame(columns=["loss", "a", "b", "ga", "gb"], dtype=float)
        self.learning_summary.index.name = "epoch"

    def convert_params(self, a, b, to="normalized"):
        if to == "normalized":
            aa = a * self.xstd / self.ystd
            bb = (b - self.ymean) / self.ystd + aa * self.xmean / self.xstd
        elif to == "real":
            aa = a * self.ystd / self.xstd
            bb = self.ymean + self.ystd * (b - self.xmean * a / self.xstd)
        else:
            raise ValueError(f"Invalid parameter '{to}' for 'normalized'")
        return aa, bb

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
        x = self.x.reshape(1, self.n)
        ones = np.ones(shape=x.shape)
        x_t = np.concatenate((ones, x))
        x = x_t.T
        y = self.y.reshape(self.n, 1)
        self.b, self.a = np.linalg.inv(x_t @ x) @ x_t @ y
        self.b = self.b[0]
        self.a = self.a[0]
        self._finish_learning()

    def fit_numeric(self, theta=None):
        def one_fit(a_, b_, x_, y_):
            y_model = a_ * x_ + b_
            diffs = y_ - y_model
            loss = np.sum(diffs ** 2) / (2 * self.n)
            if np.isinf(loss) or np.isnan(loss):
                raise ValueError("loss is invalid")
            g_a = (-1 / self.n) * (x_ * diffs).sum()
            g_b = (-1 / self.n) * diffs.sum()
            return loss, g_a, g_b

        print(f"fitting...")
        """ Code here """

        i = 0
        a_real, b_real = theta
        a_norm, b_norm = self.convert_params(a_real, b_real, to="normalized")

        if self.max_epoch is None:
            raise ValueError("max_epoch can not be None")
        while self.max_epoch is None or i < self.max_epoch:
            try:
                self.a = a_real
                self.b = b_real
                loss_real, g_a_real, g_b_real = one_fit(a_real, b_real, self.x, self.y)
                loss_norm, g_a_norm, g_b_norm = one_fit(a_norm, b_norm, self.x_norm, self.y_norm)
                print(f"  a (real, normalized): {self.a}, {a_norm} -- b (real, normalized): {self.b}, {b_norm}")
                print(f"Epoch {i} | loss (real, normalized coordinates): {loss_real}, {loss_norm}")
                if np.isinf(loss_real) or np.isnan(loss_real) or np.isinf(loss_norm) or np.isnan(loss_norm):
                    raise ValueError("loss is invalid")

                self.learning_summary_norm.loc[i] = [loss_norm, a_norm, b_norm, g_a_norm, g_b_norm]
                self.learning_summary.loc[i] = [loss_real, self.a, self.b, g_a_real, g_b_real]

                i += 1
                a_real = a_real - self.learning_rate * g_a_real
                b_real = b_real - self.learning_rate * g_b_real
                a_norm = a_norm - self.learning_rate * g_a_norm
                b_norm = b_norm - self.learning_rate * g_b_norm
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
            return plt.gcf()

    def plot_summary(self, path: str = None, which="real"):
        if self.learning_summary.empty:
            print("Learning summary is empty")
            return

        if which == "real":
            df = self.learning_summary
        else:
            df = self.learning_summary_norm

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

    def save_summary(self, path, which="real"):
        if which == "real":
            self.learning_summary.to_csv(path)
        else:
            self.learning_summary_norm.to_csv(path)

    def animate(self, path: str = None, correct_a: float = None, correct_b: float = None, step=20, nbins=100):
        def get_loss(a_, b_, x_, y_):
            ax = np.outer(a_, x_)
            diff = np.array([ax - y_ + sb for sb in b_])
            loss = np.sum(diff ** 2, axis=2) / (2 * self.n)
            return loss

        def make_range(real, col):
            min_ = min(min(col), real)
            max_ = max(max(col), real)

            dist_to_min = abs(real - min_)
            dist_to_max = abs(real - max_)

            if dist_to_min > dist_to_max:
                max_ = real + dist_to_min
            else:
                min_ = real - dist_to_max
            return [min_, max_]

        if correct_a is None:
            correct_a = self.a
        if correct_b is None:
            correct_b = self.b

        correct_a_norm, correct_b_norm = self.convert_params(correct_a, correct_b, to="normalized")

        a_real = self.learning_summary.iloc[0]["a"]
        b_real = self.learning_summary.iloc[0]["b"]
        a_norm = self.learning_summary_norm.iloc[0]["a"]
        b_norm = self.learning_summary_norm.iloc[0]["b"]

        a_range_real = make_range(correct_a, self.learning_summary["a"])
        b_range_real = make_range(correct_b, self.learning_summary["b"])
        a_range_norm = make_range(correct_a_norm, self.learning_summary_norm["a"])
        b_range_norm = make_range(correct_b_norm, self.learning_summary_norm["b"])
        slopes_real = np.linspace(*a_range_real, nbins)
        slopes_norm = np.linspace(*a_range_norm, nbins)
        intercepts_real = np.linspace(*b_range_real, nbins)
        intercepts_norm = np.linspace(*b_range_norm, nbins)
        slopes_grid_real, intercepts_grid_real = np.meshgrid(slopes_real, intercepts_real)
        slopes_grid_norm, intercepts_grid_norm = np.meshgrid(slopes_norm, intercepts_norm)
        losses_real = get_loss(slopes_real, intercepts_real, self.x, self.y)
        losses_norm = get_loss(slopes_norm, intercepts_norm, self.x_norm, self.y_norm)

        fig = plt.figure(figsize=(15, 15))
        axes = [
            fig.add_subplot(2, 2, 1),
            fig.add_subplot(2, 2, 2, projection="3d"),
            fig.add_subplot(2, 2, 3),
            fig.add_subplot(2, 2, 4, projection="3d"),
        ]

        p_real = axes[0].scatter(self.x, self.y, color="blue", marker="o", s=10)
        axes[0].grid()
        line_real = axes[0].plot(
            self.x,
            a_real * self.x + b_real,
            color="red",
        )[0]
        tb = f" + {round(b_real, 3)}" if b_real >= 0 else f"-{round(b_real, 3)}"
        legend_real = axes[0].text(0.1, 0.95, f"Epoch 0: {round(a_real, 3)}x{tb}", transform=axes[0].transAxes)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")

        p_norm = axes[2].scatter(self.x, self.y, color="blue", marker="o", s=10)
        axes[2].grid()
        line_norm = axes[2].plot(
            self.x,
            a_real * self.x + b_real,
            color="red",
        )[0]
        tb = f" + {round(b_norm, 3)}" if b_norm >= 0 else f"-{round(b_norm, 3)}"
        legend_norm = axes[2].text(0.1, 0.95, f"Epoch 0: {round(a_norm, 3)}x{tb}", transform=axes[2].transAxes)
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")

        axes[1].plot_surface(
            slopes_grid_real, intercepts_grid_real, losses_real, alpha=0.5, antialiased=True, cmap="terrain"
        )
        axes[1].set_xlabel("a")
        axes[1].set_ylabel("b")
        axes[1].set_zlabel("loss")
        axes[1].scatter(
            [slopes_real[0]],
            [intercepts_real[0]],
            [losses_real[0][0]],
            marker="x",
            color="red",
            s=10,
        )

        axes[3].plot_surface(
            slopes_grid_norm, intercepts_grid_norm, losses_norm, alpha=0.5, antialiased=True, cmap="terrain"
        )
        axes[3].set_xlabel("a_N")
        axes[3].set_ylabel("b_N")
        axes[3].set_zlabel("loss_N")
        axes[3].scatter(
            [slopes_norm[0]],
            [intercepts_norm[0]],
            [losses_norm[0][0]],
            marker="x",
            color="red",
            s=10,
        )

        def update(i):
            print(f"Animating {i}...")
            a_real_ = self.learning_summary.loc[i, "a"]
            b_real_ = self.learning_summary.loc[i, "b"]
            a_norm_ = self.learning_summary_norm.loc[i, "a"]
            b_norm_ = self.learning_summary_norm.loc[i, "b"]

            pred_real = a_real_ * self.x + b_real_
            a_norm_real, b_norm_real = self.convert_params(a_norm_, b_norm_, to="real")
            pred_norm_real = a_norm_real * self.x + b_norm_real
            line_real.set_ydata(pred_real)
            line_norm.set_ydata(pred_norm_real)
            tb_ = f" + {round(b_real_, 3)}" if b_real_ > 0 else f"-{round(b_real_, 3)}"
            legend_real.set_text(f"Epoch {i}: {round(a_real_, 3)}x{tb_}")
            tb_ = f" + {round(b_norm_real, 3)}" if b_norm_real > 0 else f"-{round(b_norm_real, 3)}"
            legend_norm.set_text(f"Epoch {i}: {round(a_norm_real, 3)}x{tb_}")

            loss_real = self.learning_summary.loc[i, "loss"]
            loss_norm = self.learning_summary_norm.loc[i, "loss"]

            axes[1].scatter(
                [a_real_],
                [b_real_],
                [loss_real],
                marker="x",
                color="red",
                s=10,
            )

            axes[3].scatter(
                [a_norm_],
                [b_norm_],
                [loss_norm],
                marker="x",
                color="red",
                s=10,
            )
            return p_real, p_norm, line_real, line_norm, legend_real, legend_norm

        # noinspection PyTypeChecker
        ani = animation.FuncAnimation(fig, update, blit=True, frames=self.learning_summary.index[1::step])
        plt.title("Top : Real coordinates. Bottom : normalized coordinates.")

        if path is not None:
            ani.save(path, fps=20, dpi=300)
            print(f"...animation saved in {path}")
            plt.close("all")  # Without this command, all future plots will be made in the same figure
        else:
            return ani
