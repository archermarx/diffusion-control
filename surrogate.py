import numpy as np
from scipy.optimize import minimize
from smt.surrogate_models import KRG
import matplotlib.pyplot as plt
from scipy.stats import norm


class Surrogate:
    def __init__(
        self,
        dim=1,
        bounds=None,
        min_points=None,
        theta0=None,
        corr="matern32",
        optimize_restarts=10,
        acquisition="ei",
        xi=0.0,
        seed=None,
    ):
        self.dim = dim
        self.bounds = bounds
        self.min_points = min_points if min_points is not None else max(2, dim + 1)
        self.theta0 = theta0 if theta0 is not None else [1e-2] * dim
        self.corr = corr
        self.optimize_restarts = optimize_restarts

        if acquisition not in {"mean", "ei"}:
            raise ValueError("acquisition must be 'mean' or 'ei'")

        if xi < 0.0:
            raise ValueError("xi must be nonnegative")

        self.acquisition = acquisition
        self.xi = float(xi)

        self.rng = np.random.default_rng(seed)

        self.X = []
        self.Y = []

        self.model = None
        self.is_trained = False

        self.history = []

    def __call__(self, x) -> float:
        x = self._as_vector(x)

        if len(self.Y) == 0:
            return 0.0

        if not self.is_trained:
            return float(min(self.Y))

        x_scaled = self._scale(x.reshape(1, -1))
        z = self.model.predict_values(x_scaled)
        return float(z[0, 0])

    def update(self, x, y) -> None:
        x = self._as_vector(x)
        y = float(y)

        self.X.append(x)
        self.Y.append(y)

        # Not sure if we should fit every single time update is called or find a way to circumvent this
        # because fitting every time for increasing datapoints could become costly
        self._fit()
        self._store_history()

    def optimize(
        self,
        acquisition=None,
    ) -> tuple[np.ndarray, float]:
        """
        Find a control using either:

            acquisition="mean":
                lowest predicted Kriging mean

            acquisition="ei":
                largest Expected Improvement

        Returns
        -------
        c_best:
            Selected control point.

        z_pred:
            Kriging predicted metric at that control point.
            This is not the EI value.
        """

        mode = self.acquisition if acquisition is None else acquisition

        if mode not in {"mean", "ei"}:
            raise ValueError("acquisition must be 'mean' or 'ei'")

        if len(self.Y) == 0:
            c0 = self._default_control()
            return c0, self(c0)

        best_seen_idx = int(np.argmin(self.Y))
        best_seen = self.X[best_seen_idx].copy()

        # EI is not meaningful until KRG has been trained.
        if not self.is_trained:
            return best_seen, float(self.Y[best_seen_idx])

        bounds = self._get_bounds(best_seen)

        midpoint = np.array(
            [
                0.5 * (lo + hi)
                for lo, hi in bounds
            ],
            dtype=float,
        )

        starts = [midpoint]

        if mode == "mean":
            # For mean minimization, known good controls are useful starts.
            starts.append(best_seen)
            starts.append(self.X[-1].copy())

            for idx in np.argsort(self.Y)[: min(3, len(self.Y))]:
                starts.append(
                    self.X[int(idx)].copy()
                )

        elif mode == "ei":
            # Do not rely primarily on observed points for EI.
            # At observed points, Kriging variance and EI are usually near zero.
            for idx in np.argsort(self.Y)[: min(3, len(self.Y))]:
                center = self.X[int(idx)].copy()

                jitter = np.array(
                    [
                        0.05 * (hi - lo)
                        * self.rng.standard_normal()
                        for lo, hi in bounds
                    ]
                )

                starts.append(
                    self._clip(center + jitter)
                )

        # Add random multistart locations for either acquisition.
        for _ in range(self.optimize_restarts):
            starts.append(
                np.array(
                    [
                        self.rng.uniform(lo, hi)
                        for lo, hi in bounds
                    ]
                )
            )

        objective = lambda c: self._acquisition_objective(
            c,
            mode,
        )

        best_c = self._clip(starts[0])
        best_objective = objective(best_c)

        for start in starts:
            result = minimize(
                objective,
                start,
                method="L-BFGS-B",
                bounds=bounds,
            )

            c_candidate = self._clip(result.x)
            candidate_objective = objective(c_candidate)

            if (
                np.isfinite(candidate_objective)
                and candidate_objective < best_objective
            ):
                best_c = c_candidate
                best_objective = candidate_objective

        # Preserve the existing interface:
        # return the chosen control and its predicted metric.
        z_pred = self(best_c)

        return best_c, float(z_pred)

    def variance(self, x) -> float:
        x = self._as_vector(x)

        if not self.is_trained:
            return float("inf")

        x_scaled = self._scale(x.reshape(1, -1))
        var = self.model.predict_variances(x_scaled)
        return max(0.0, float(var[0, 0]))

    def mean_and_std(self, x) -> tuple[float, float]:
        """
        Return the Kriging predicted mean and standard deviation at x.
        """

        x = self._as_vector(x)

        if not self.is_trained:
            raise RuntimeError(
                "The surrogate must be trained before predicting uncertainty."
            )

        x_scaled = self._scale(x.reshape(1, -1))

        mean = float(
            self.model.predict_values(x_scaled)[0, 0]
        )

        variance = float(
            self.model.predict_variances(x_scaled)[0, 0]
        )

        variance = max(variance, 0.0)
        std = np.sqrt(variance)

        return mean, float(std)

    def expected_improvement(self, x) -> float:
        """
        Expected Improvement acquisition value for minimization.

        A larger value means x is a more useful control point to evaluate next.
        """

        if not self.is_trained:
            return 0.0

        mean, std = self.mean_and_std(x)

        best_observed = float(np.min(self.Y))

        improvement = (
            best_observed
            - mean
            - self.xi
        )

        # Handle points with effectively zero uncertainty.
        if std < 1e-12:
            return max(improvement, 0.0)

        gamma = improvement / std

        ei = (
            improvement * norm.cdf(gamma)
            + std * norm.pdf(gamma)
        )

        return max(0.0, float(ei))

    def _acquisition_objective(self, c, acquisition):
        """
        Objective passed to scipy.optimize.minimize().

        SciPy minimizes, so EI is negated because EI should be maximized.
        """

        if acquisition == "mean":
            return self(c)

        if acquisition == "ei":
            return -self.expected_improvement(c)

        raise ValueError("acquisition must be 'mean' or 'ei'")

    def _fit(self):
        X = np.vstack(self.X)
        Y = np.array(self.Y, dtype=float)

        X_scaled = self._scale(X)
        X_unique, Y_unique = self._remove_duplicate_points(X_scaled, Y)

        if len(Y_unique) < self.min_points:
            self.is_trained = False
            return

        model = KRG(
            theta0=self.theta0,
            corr=self.corr,
            print_global=False,
        )

        model.set_training_values(X_unique, Y_unique.reshape(-1, 1))
        model.train()

        self.model = model
        self.is_trained = True

    def _remove_duplicate_points(self, X, Y):
        rounded = np.round(X, decimals=12)
        X_unique, inverse = np.unique(rounded, axis=0, return_inverse=True)

        Y_unique = np.zeros(len(X_unique))
        counts = np.zeros(len(X_unique))

        for i, group in enumerate(inverse):
            Y_unique[group] += Y[i]
            counts[group] += 1

        Y_unique /= counts
        return X_unique, Y_unique

    def _scale(self, X):
        return self._scale_with_reference(X, self.X)

    def _scale_with_reference(self, X, X_reference):
        X = np.asarray(X, dtype=float)

        if self.bounds is not None:
            lo = np.array([b[0] for b in self.bounds], dtype=float)
            hi = np.array([b[1] for b in self.bounds], dtype=float)
            width = hi - lo
            width[width == 0.0] = 1.0
            return (X - lo) / width

        X_reference = np.vstack(X_reference)
        center = np.mean(X_reference, axis=0)
        scale = np.std(X_reference, axis=0)
        scale[scale < 1e-12] = 1.0
        return (X - center) / scale

    def _get_bounds(self, c_reference):
        if self.bounds is not None:
            return self.bounds

        span = 0.25 * (np.abs(c_reference) + 1.0)
        return list(zip(c_reference - span, c_reference + span))

    def _clip(self, c):
        c = np.asarray(c, dtype=float).reshape(-1)

        if self.bounds is None:
            return c

        lo = np.array([b[0] for b in self.bounds], dtype=float)
        hi = np.array([b[1] for b in self.bounds], dtype=float)

        return np.clip(c, lo, hi)

    def _default_control(self):
        if self.bounds is not None:
            return np.array([0.5 * (lo + hi) for lo, hi in self.bounds], dtype=float)

        return np.zeros(self.dim)

    def _as_vector(self, x):
        x = np.asarray(x, dtype=float).reshape(-1)

        if x.size != self.dim:
            raise ValueError(f"Expected dimension {self.dim}, got {x.size}")

        return x
    
    def plot_1d(
        self,
        ground_truth=None,
        num_points=400,
        errorbar_points=25,
        confidence=2.0,
        show_band=True,
        xlabel="Control c",
        ylabel="Metric z",
        title="Kriging surrogate",
        ground_truth_label="Ground truth",
        filename=None,
        extension=0,
        show=True,
    ):
        """
        Plot a one-dimensional Kriging surrogate and its uncertainty.

        Parameters
        ----------
        ground_truth:
            Either:

            1. A callable:
                ground_truth(c) -> z

            2. A tuple containing existing data arrays:
                (ground_truth_controls, ground_truth_metrics)

            3. None, in which case no ground-truth curve is shown.

        num_points:
            Number of points used to draw the surrogate curve.

        errorbar_points:
            Number of locations where uncertainty error bars are drawn.

        confidence:
            Number of predictive standard deviations used for the uncertainty.
            confidence=2 approximately corresponds to mean ± 2 standard deviations.

        show_band:
            Whether to also show a continuous uncertainty band.

        filename:
            Optional path where the figure is saved.

        show:
            Whether to display the plot interactively.
        """

        if self.dim != 1:
            raise ValueError(
                "plot_1d() only works for a one-dimensional surrogate. "
                "For a multidimensional surrogate, plot a one-dimensional slice."
            )

        if not self.is_trained:
            raise RuntimeError(
                "The surrogate must be trained before it can be plotted."
            )

        # Real control points and measured metric values used to train KRG
        observed_c = np.vstack(self.X)[:, 0]
        observed_z = np.asarray(self.Y, dtype=float)

        # Decide what control range should be plotted
        if self.bounds is not None:
            lower, upper = self.bounds[0]
        else:
            lower = float(np.min(observed_c))
            upper = float(np.max(observed_c))

            span = upper - lower
            padding = 0.2 * span if span > 0 else 0.1

            lower -= padding
            upper += padding

        # Dense grid used to draw the surrogate function
        c_grid = np.linspace(lower - extension, upper + extension, num_points)
        c_grid_matrix = c_grid.reshape(-1, 1)
        c_grid_scaled = self._scale(c_grid_matrix)

        # Kriging predicted mean and variance
        predicted_mean = (
            self.model.predict_values(c_grid_scaled)
            .reshape(-1)
        )

        predicted_variance = (
            self.model.predict_variances(c_grid_scaled)
            .reshape(-1)
        )

        # Small negative variances can occur from numerical roundoff
        predicted_variance = np.maximum(predicted_variance, 0.0)

        # Error bars must use standard deviation, not raw variance
        predicted_std = np.sqrt(predicted_variance)
        uncertainty = confidence * predicted_std

        confidence_lower = predicted_mean - uncertainty
        confidence_upper = predicted_mean + uncertainty

        # Use only a smaller number of points for error bars so the plot
        # does not become overcrowded
        error_indices = np.linspace(
            0,
            num_points - 1,
            min(errorbar_points, num_points),
            dtype=int,
        )

        c_error = c_grid[error_indices]
        mean_error = predicted_mean[error_indices]
        y_error = uncertainty[error_indices]

        # Surrogate-predicted minimum
        c_best, z_best = self.optimize(
            acquisition="mean"
        )

        fig, ax = plt.subplots(figsize=(9, 6))

        # Plot the Kriging mean function
        ax.plot(
            c_grid,
            predicted_mean,
            label="Kriging predicted mean",
            linewidth=2,
        )


        c_ei, z_ei = self.optimize(
            acquisition="ei"
        )

        ax.scatter(
            c_ei[0],
            z_ei,
            marker="D",
            s=90,
            label="EI-selected next point",
            zorder=6,
        )

        # Plot uncertainty as vertical error bars along the surrogate curve
        # ax.errorbar(
        #     c_error,
        #     mean_error,
        #     yerr=y_error,
        #     fmt="none",
        #     capsize=3,
        #     alpha=0.65,
        #     label=(
        #         f"Kriging uncertainty "
        #         f"(±{confidence:g} standard deviations)"
        #     ),
        # )

        # Optional continuous uncertainty band
        if show_band:
            ax.fill_between(
                c_grid,
                confidence_lower,
                confidence_upper,
                alpha=0.15,
                label=(
                    f"Mean ± {confidence:g} standard deviations"
                ),
            )

        # Plot the actual control/metric observations used for training
        ax.scatter(
            observed_c,
            observed_z,
            marker="o",
            s=55,
            label="Observed control points",
            zorder=4,
        )

        # Plot the ground-truth function if one was provided
        if ground_truth is not None:
            if callable(ground_truth):
                ground_truth_c = c_grid
                ground_truth_z = np.asarray(
                    [ground_truth(c) for c in ground_truth_c],
                    dtype=float,
                )
            else:
                if len(ground_truth) != 2:
                    raise ValueError(
                        "ground_truth must be either a callable or "
                        "a tuple of (control_values, metric_values)."
                    )

                ground_truth_c = np.asarray(
                    ground_truth[0],
                    dtype=float,
                ).reshape(-1)

                ground_truth_z = np.asarray(
                    ground_truth[1],
                    dtype=float,
                ).reshape(-1)

                if ground_truth_c.size != ground_truth_z.size:
                    raise ValueError(
                        "Ground-truth control and metric arrays "
                        "must have the same length."
                    )

            ax.plot(
                ground_truth_c,
                ground_truth_z,
                linestyle="--",
                linewidth=2,
                label=ground_truth_label,
            )

        # Show the minimum found by optimize()
        ax.scatter(
            c_best[0],
            z_best,
            marker="*",
            s=180,
            label="Surrogate-predicted minimum",
            zorder=5,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()

        if filename is not None:
            fig.savefig(
                filename,
                dpi=200,
                bbox_inches="tight",
            )

        if show:
            plt.show()

        return fig, ax

    def _store_history(self):
        snapshot = {
            "X": [x.copy() for x in self.X],
            "Y": list(self.Y),
            "is_trained": self.is_trained,
        }
        self.history.append(snapshot)

    def _fit_snapshot_model(self, X_snapshot, Y_snapshot):
        X = np.vstack(X_snapshot)
        Y = np.array(Y_snapshot, dtype=float)

        X_scaled = self._scale_with_reference(X, X_snapshot)
        X_unique, Y_unique = self._remove_duplicate_points(X_scaled, Y)

        if len(Y_unique) < self.min_points:
            return None

        model = KRG(
            theta0=self.theta0,
            corr=self.corr,
            print_global=False,
        )

        model.set_training_values(X_unique, Y_unique.reshape(-1, 1))
        model.train()

        return model

def generate_control_points(bounds, num_points, seed=1, shuffle=True):
    """
    Generate evenly spaced control points across the supplied bounds.

    shuffle=True changes the order in which points are added so the
    surrogate receives points from different parts of the domain instead
    of receiving them strictly from left to right.
    """
    lower, upper = bounds

    points = np.linspace(
        lower,
        upper,
        num_points,
    )

    if shuffle:
        rng = np.random.default_rng(seed)
        points = rng.permutation(points)

    return points.tolist()

# if __name__ == "__main__":
#     try:
#         # Smoke test: known minimum near c = 1.05
#         s = Surrogate(
#             dim=1,
#             bounds=[(-2, 2)],
#             min_points=2,
#             optimize_restarts=5,
#             seed=1,
#         )

#         def ground_truth(c):
#             return (((c - 1.05) ** 2) * ((c + 1.05) ** 2))

#         control_points = [-1.1, -1.0, -0.7, -0.65, -0.4, -0.2, 0, 0.35, 0.4, 0.65, 0.7, 0.75, 0.9, 1.0, 1.1, 1.15, 1.2, 1.25]

#         ext = 0

#         for count, c in enumerate(control_points, start=1):
#             z = ground_truth(c)
#             s.update([c], z)

#             # KRG cannot be plotted until enough distinct points exist.
#             if not s.is_trained:
#                 print(
#                     f"Added point {count}: c={c}, z={z}. "
#                     "Not enough points to train KRG yet."
#                 )
#                 continue

#             fig, ax = s.plot_1d(
#                 ground_truth=ground_truth,
#                 xlabel="Control c",
#                 ylabel="Metric z",
#                 title=f"Kriging surrogate with {count} points added",
#                 filename=f"surrogate_{count}_points.png",
#                 show=False,
#                 extension=ext
#             )

#             # Prevent saved figures from accumulating in memory.
#             plt.close(fig)

#             print(f"Saved surrogate_{count}_points.png")

#         # Optimize only once after all points have been added.
#         c_best, z_best = s.optimize()

#         print("best c:", c_best)
#         print("predicted z:", z_best)
#         print("variance:", s.variance(c_best))

#         fig, ax = s.plot_convergence_1d(
#             ground_truth=ground_truth,
#             xlabel="Control c",
#             ylabel="Metric z",
#             title="Kriging surrogate convergence",
#             filename="surrogate_convergence.png",
#             show_band=True,
#             label_every=1,
#             show=False,
#             extension=ext
#         )

#         plt.close(fig)

#         print("Saved surrogate_convergence.png")

#     except KeyboardInterrupt:
#         plt.close("all")
#         print("\nStopped by user.")

