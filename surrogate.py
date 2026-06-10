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

    def plot_convergence_1d(
        self,
        ground_truth=None,
        num_points=400,
        errorbar_points=25,
        confidence=2.0,
        show_band=False,
        xlabel="Control c",
        ylabel="Metric z",
        title="Kriging surrogate convergence",
        ground_truth_label="Ground truth",
        filename=None,
        show=True,
        extension=0,
        label_every=1,
    ):
        """
        Plot the surrogate after every update on the same figure to show convergence.

        For each stored snapshot:
            - rebuild the surrogate model at that stage
            - plot the surrogate mean curve

        Also plots:
            - final observed control points
            - final uncertainty as error bars
            - optional ground-truth function
            - final surrogate-predicted minimum
        """

        if self.dim != 1:
            raise ValueError(
                "plot_convergence_1d() only works for a one-dimensional surrogate."
            )

        if len(self.history) == 0:
            raise RuntimeError("No history has been recorded yet.")

        final_X = np.vstack(self.X)[:, 0]
        final_Y = np.asarray(self.Y, dtype=float)

        if self.bounds is not None:
            lower, upper = self.bounds[0]
        else:
            lower = float(np.min(final_X))
            upper = float(np.max(final_X))
            span = upper - lower
            padding = 0.2 * span if span > 0 else 0.1
            lower -= padding
            upper += padding

        c_grid = np.linspace(lower - extension, upper + extension, num_points)
        c_grid_matrix = c_grid.reshape(-1, 1)

        fig, ax = plt.subplots(figsize=(9, 6))

        # Plot the ground-truth curve if given
        if ground_truth is not None:
            if callable(ground_truth):
                gt_x = c_grid
                gt_y = np.asarray([ground_truth(c) for c in gt_x], dtype=float)
            else:
                if len(ground_truth) != 2:
                    raise ValueError(
                        "ground_truth must be a callable or a tuple "
                        "(x_values, y_values)."
                    )
                gt_x = np.asarray(ground_truth[0], dtype=float).reshape(-1)
                gt_y = np.asarray(ground_truth[1], dtype=float).reshape(-1)

            ax.plot(
                gt_x,
                gt_y,
                linestyle="--",
                linewidth=2,
                label=ground_truth_label,
            )

        # Plot the surrogate curve from each snapshot
        n_hist = len(self.history)

        for i, snapshot in enumerate(self.history, start=1):
            X_snapshot = snapshot["X"]
            Y_snapshot = snapshot["Y"]

            alpha = 0.15 + 0.75 * (i / n_hist)
            linewidth = 1.0 if i < n_hist else 2.5

            # If not enough unique points yet to train KRG,
            # show a flat "best so far" line
            model = self._fit_snapshot_model(X_snapshot, Y_snapshot)

            if model is None:
                y_flat = np.full_like(c_grid, float(np.min(Y_snapshot)))
                label = (
                    f"after {len(Y_snapshot)} points"
                    if (i % label_every == 0 or i == n_hist)
                    else None
                )
                ax.plot(
                    c_grid,
                    y_flat,
                    linewidth=linewidth,
                    alpha=alpha,
                    label=label,
                )
                continue

            c_grid_scaled = self._scale_with_reference(c_grid_matrix, X_snapshot)
            predicted_mean = model.predict_values(c_grid_scaled).reshape(-1)

            label = (
                f"after {len(Y_snapshot)} points"
                if (i % label_every == 0 or i == n_hist)
                else None
            )

            ax.plot(
                c_grid,
                predicted_mean,
                linewidth=linewidth,
                alpha=alpha,
                label=label,
            )

        # Plot final uncertainty from the final trained model
        if self.is_trained:
            c_grid_scaled = self._scale(c_grid_matrix)
            predicted_mean = self.model.predict_values(c_grid_scaled).reshape(-1)
            predicted_variance = self.model.predict_variances(c_grid_scaled).reshape(-1)
            predicted_variance = np.maximum(predicted_variance, 0.0)
            predicted_std = np.sqrt(predicted_variance)
            yerr = confidence * predicted_std

            # sparse error bars so the plot doesn't get too messy
            error_indices = np.linspace(
                0,
                num_points - 1,
                min(errorbar_points, num_points),
                dtype=int,
            )

            ax.errorbar(
                c_grid[error_indices],
                predicted_mean[error_indices],
                yerr=yerr[error_indices],
                fmt="none",
                capsize=3,
                alpha=0.6,
                label=f"Final uncertainty (±{confidence:g} std)",
            )

            if show_band:
                ax.fill_between(
                    c_grid,
                    predicted_mean - yerr,
                    predicted_mean + yerr,
                    alpha=0.12,
                    label=f"Final mean ± {confidence:g} std",
                )

        # Plot the final observed points
        ax.scatter(
            final_X,
            final_Y,
            marker="o",
            s=55,
            label="Observed points",
            zorder=5,
        )

        # Plot the final predicted minimum
        if self.is_trained:
            c_best, z_best = self.optimize(
                acquisition="mean"
            )
            ax.scatter(
                c_best[0],
                z_best,
                marker="*",
                s=180,
                label="Final predicted minimum",
                zorder=6,
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

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()

        if filename is not None:
            fig.savefig(filename, dpi=200, bbox_inches="tight")

        if show:
            plt.show()

        return fig, ax

    def plot_1d_on_axis(
        self,
        ax,
        ground_truth=None,
        num_points=400,
        errorbar_points=15,
        confidence=2.0,
        show_band=True,
        xlabel="Control c",
        ylabel="Metric z",
        title=None,
        ground_truth_label="Ground truth",
        extension=0.0,
        ei_point=None,
    ):
        """
        Draw the current one-dimensional surrogate on an existing axis.

        This is useful for placing several surrogate-update plots next
        to each other in one figure.
        """

        if self.dim != 1:
            raise ValueError(
                "plot_1d_on_axis() only works for a one-dimensional surrogate."
            )

        if not self.is_trained:
            raise RuntimeError(
                "The surrogate must be trained before it can be plotted."
            )

        observed_c = np.vstack(self.X)[:, 0]
        observed_z = np.asarray(self.Y, dtype=float)

        if self.bounds is not None:
            lower = min(self.bounds[0][0], float(np.min(observed_c)))
            upper = max(self.bounds[0][1], float(np.max(observed_c)))
        else:
            lower = float(np.min(observed_c))
            upper = float(np.max(observed_c))

        c_grid = np.linspace(
            lower - extension,
            upper + extension,
            num_points,
        )

        c_grid_scaled = self._scale(c_grid.reshape(-1, 1))

        predicted_mean = self.model.predict_values(
            c_grid_scaled
        ).reshape(-1)

        predicted_variance = self.model.predict_variances(
            c_grid_scaled
        ).reshape(-1)

        predicted_variance = np.maximum(
            predicted_variance,
            0.0,
        )

        predicted_std = np.sqrt(predicted_variance)
        uncertainty = confidence * predicted_std

        ax.plot(
            c_grid,
            predicted_mean,
            linewidth=2,
            label="Kriging mean",
        )

        if show_band:
            ax.fill_between(
                c_grid,
                predicted_mean - uncertainty,
                predicted_mean + uncertainty,
                alpha=0.15,
                label=f"Mean ± {confidence:g} std",
            )

        error_indices = np.linspace(
            0,
            num_points - 1,
            min(errorbar_points, num_points),
            dtype=int,
        )

        # ax.errorbar(
        #     c_grid[error_indices],
        #     predicted_mean[error_indices],
        #     yerr=uncertainty[error_indices],
        #     fmt="none",
        #     capsize=2,
        #     alpha=0.5,
        #     label="Prediction uncertainty",
        # )

        ax.scatter(
            observed_c,
            observed_z,
            s=40,
            marker="o",
            label="Observed points",
            zorder=4,
        )

        if ground_truth is not None:
            if callable(ground_truth):
                ground_truth_c = c_grid
                ground_truth_z = np.asarray(
                    [ground_truth(c) for c in ground_truth_c],
                    dtype=float,
                )
            else:
                ground_truth_c = np.asarray(
                    ground_truth[0],
                    dtype=float,
                ).reshape(-1)

                ground_truth_z = np.asarray(
                    ground_truth[1],
                    dtype=float,
                ).reshape(-1)

            ax.plot(
                ground_truth_c,
                ground_truth_z,
                linestyle="--",
                linewidth=2,
                label=ground_truth_label,
            )

        # Use the plotted grid to mark the approximate minimum.
        # This avoids running the full multistart optimizer for every subplot.
        best_index = int(np.argmin(predicted_mean))

        ax.scatter(
            c_grid[best_index],
            predicted_mean[best_index],
            marker="*",
            s=130,
            label="Predicted minimum",
            zorder=5,
        )

        if ei_point is not None:
            ei_point = self._as_vector(ei_point)
            ei_predicted_z = self(ei_point)

            ax.scatter(
                ei_point[0],
                ei_predicted_z,
                marker="D",
                s=90,
                label="EI-selected next point",
                zorder=6,
            )

        ax.set_xlim(c_grid[0], c_grid[-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        ax.grid(True, alpha=0.3)

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

def run_progression_test(
        name,
        ground_truth,
        bounds,
        control_points,
        filename,
        min_points=3,
        extension=0.0,
        columns=3,
    ):
        """
        Train a surrogate one control point at a time and place every
        trained stage into one subplot figure.
        """

        surrogate = Surrogate(
            dim=1,
            bounds=[bounds],
            min_points=min_points,
            optimize_restarts=20,
            acquisition="mean",
            xi=0.0,
            seed=1,
        )

        number_of_plots = len(control_points) - min_points + 1

        if number_of_plots <= 0:
            raise ValueError(
                "There must be at least min_points control points."
            )

        columns = min(columns, number_of_plots)
        rows = int(np.ceil(number_of_plots / columns))

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(5 * columns, 4 * rows),
            squeeze=False,
        )

        axes = axes.reshape(-1)
        plot_index = 0

        for count, control in enumerate(control_points, start=1):
            metric = ground_truth(control)
            surrogate.update([control], metric)

            if not surrogate.is_trained:
                continue

            surrogate.plot_1d_on_axis(
                ax=axes[plot_index],
                ground_truth=ground_truth,
                xlabel="Control c",
                ylabel="Function value z",
                title=f"{count} points",
                extension=extension,
            )

            plot_index += 1

        # Hide any unused subplot spaces.
        for axis in axes[plot_index:]:
            axis.axis("off")

        # Create one shared legend instead of repeating a legend in every subplot.
        handles, labels = axes[plot_index - 1].get_legend_handles_labels()

        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
        )

        fig.suptitle(
            f"{name}: Kriging surrogate convergence",
            fontsize=16,
        )

        fig.tight_layout(
            rect=(0.0, 0.09, 1.0, 0.95)
        )

        fig.savefig(
            filename,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        c_best, z_best = surrogate.optimize()

        print(f"\n{name}")
        print("Final predicted best c:", c_best)
        print("Final predicted z:", z_best)
        print("Variance at predicted minimum:", surrogate.variance(c_best))
        print("Saved:", filename)
  
def run_ei_progression_test(
    name,
    ground_truth,
    bounds,
    initial_points,
    ei_iterations,
    filename,
    min_points=3,
    extension=0.0,
    columns=3,
    optimize_restarts=20,
    xi=0.0,
    seed=1,
):
    """
    Start with a small initial design and let Expected Improvement choose
    each subsequent control point.

    Each stage is plotted as a subplot in one output image.
    """

    if len(initial_points) < min_points:
        raise ValueError(
            "The number of initial points must be at least min_points."
        )

    surrogate = Surrogate(
        dim=1,
        bounds=[bounds],
        min_points=min_points,
        optimize_restarts=optimize_restarts,
        acquisition="ei",
        xi=xi,
        seed=seed,
    )

    # Add the initial observations.
    for control in initial_points:
        surrogate.update(
            [control],
            ground_truth(control),
        )

    # One initial plot, then one plot after every EI-selected update.
    number_of_plots = ei_iterations + 1

    columns = min(columns, number_of_plots)
    rows = int(np.ceil(number_of_plots / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5 * columns, 4 * rows),
        squeeze=False,
    )

    axes = axes.reshape(-1)

    selected_controls = []
    selected_actual_values = []
    selected_ei_values = []

    for stage in range(number_of_plots):
        # Select the next point before plotting so the plotted diamond
        # is exactly the point that will be evaluated next.
        c_next = None
        ei_value = None

        if stage < ei_iterations:
            c_next, _ = surrogate.optimize(
                acquisition="ei"
            )

            ei_value = surrogate.expected_improvement(
                c_next
            )

        surrogate.plot_1d_on_axis(
            ax=axes[stage],
            ground_truth=ground_truth,
            xlabel="Control c",
            ylabel="Function value z",
            title=f"{len(surrogate.Y)} points",
            extension=extension,
            ei_point=c_next,
        )

        # Add the EI value to the subplot.
        if c_next is not None:
            axes[stage].text(
                0.03,
                0.97,
                (
                    f"Next c = {c_next[0]:.4f}\n"
                    f"EI = {ei_value:.3e}"
                ),
                transform=axes[stage].transAxes,
                verticalalignment="top",
                bbox={
                    "boxstyle": "round",
                    "alpha": 0.75,
                },
            )

            # Evaluate the real benchmark function and update KRG.
            z_actual = ground_truth(c_next[0])

            selected_controls.append(
                float(c_next[0])
            )
            selected_actual_values.append(
                float(z_actual)
            )
            selected_ei_values.append(
                float(ei_value)
            )

            surrogate.update(
                c_next,
                z_actual,
            )

    # Hide unused subplot spaces.
    for axis in axes[number_of_plots:]:
        axis.axis("off")

    # Create one shared legend.
    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
    )

    fig.suptitle(
        f"{name}: Expected Improvement progression",
        fontsize=16,
    )

    fig.tight_layout(
        rect=(0.0, 0.09, 1.0, 0.95)
    )

    fig.savefig(
        filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # Final estimated minimum of the Kriging mean.
    c_best, z_best = surrogate.optimize(
        acquisition="mean"
    )

    print(f"\n{name} — Expected Improvement")
    print("Final predicted minimum c:", c_best)
    print("Final predicted metric:", z_best)
    print("Saved:", filename)

    return {
        "surrogate": surrogate,
        "selected_controls": np.array(selected_controls),
        "selected_actual_values": np.array(selected_actual_values),
        "selected_ei_values": np.array(selected_ei_values),
    }

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

if __name__ == "__main__":
    try:
        # -------------------------------------------------
        # Optimization test functions
        # -------------------------------------------------

        def quartic(c):
            """
            Double-well quartic.
            Global minima near c = -1.05 and c = 1.05.
            """
            return ((c - 1.05) ** 2) * ((c + 1.05) ** 2)


        # def ackley(c):
        #     """
        #     One-dimensional Ackley function.
        #     Global minimum: c = 0, z = 0.
        #     """
        #     a = 20.0
        #     b = 0.2
        #     frequency = 2.0 * np.pi

        #     return (
        #         -a * np.exp(-b * np.sqrt(c**2))
        #         - np.exp(np.cos(frequency * c))
        #         + a
        #         + np.e
        #     )

        def ackley(c):
            a = 20.0
            b = 0.2

            return (
                -a * np.exp(-b * np.abs(c))
                - np.exp(np.cos(2.0 * np.pi * c))
                + a
                + np.e
            )


        def rastrigin(c):
            """
            One-dimensional Rastrigin function.
            Global minimum: c = 0, z = 0.
            Contains many local minima.
            """
            return c**2 - 10.0 * np.cos(2.0 * np.pi * c) + 10.0


        def forrester(c):
            """
            Common one-dimensional surrogate-model benchmark.
            Defined on approximately [0, 1].
            """
            return ((6.0 * c - 2.0) ** 2) * np.sin(12.0 * c - 4.0)


        # -------------------------------------------------
        # Test-case definitions
        # -------------------------------------------------

        test_cases = [
            {
                "name": "Quartic double well",
                "function": quartic,
                "bounds": (-1.5, 1.5),
                "control_points": generate_control_points(
                    bounds=(-1.5, 1.5),
                    num_points=15,
                    seed=1,
                ),
                "filename": "quartic_progression.png",
                "extension": 0.1,
            },
            {
                "name": "Ackley function",
                "function": ackley,
                "bounds": (-5.0, 5.0),
                "control_points": generate_control_points(
                    bounds=(-5.0, 5.0),
                    num_points=50,
                    seed=2,
                ),
                "filename": "ackley_progression.png",
                "extension": 0.25,
            },
            {
                "name": "Rastrigin function",
                "function": rastrigin,
                "bounds": (-5.12, 5.12),
                "control_points": generate_control_points(
                    bounds=(-5.12, 5.12),
                    num_points=50,
                    seed=3,
                ),
                "filename": "rastrigin_progression.png",
                "extension": 0.2,
            },
            {
                "name": "Forrester function",
                "function": forrester,
                "bounds": (0.0, 1.0),
                "control_points": generate_control_points(
                    bounds=(0.0, 1.0),
                    num_points=50,
                    seed=4,
                ),
                "filename": "forrester_progression.png",
                "extension": 0.03,
            },
        ]

        ei_test_cases = [
            {
                "name": "Quartic double well",
                "function": quartic,
                "bounds": (-1.5, 1.5),
                "initial_points": [
                    -1.5,
                    -0.5,
                    0.5,
                    1.5,
                ],
                "ei_iterations": 12,
                "filename": "quartic_ei_progression.png",
                "extension": 0.1,
                "seed": 1,
            },
            {
                "name": "Ackley function",
                "function": ackley,
                "bounds": (-5.0, 5.0),
                "initial_points": [
                    -5.0,
                    -2.0,
                    2.0,
                    5.0,
                ],
                "ei_iterations": 25,
                "filename": "ackley_ei_progression.png",
                "extension": 0.25,
                "seed": 2,
            },
            {
                "name": "Rastrigin function",
                "function": rastrigin,
                "bounds": (-5.12, 5.12),
                "initial_points": [
                    -5.12,
                    -2.0,
                    2.0,
                    5.12,
                ],
                "ei_iterations": 25,
                "filename": "rastrigin_ei_progression.png",
                "extension": 0.2,
                "seed": 3,
            },
            {
                "name": "Forrester function",
                "function": forrester,
                "bounds": (0.0, 1.0),
                "initial_points": [
                    0.0,
                    0.3,
                    0.7,
                    1.0,
                ],
                "ei_iterations": 16,
                "filename": "forrester_ei_progression.png",
                "extension": 0.03,
                "seed": 4,
            },
        ]

        # -------------------------------------------------
        # Run all benchmark tests
        # -------------------------------------------------

        for test in test_cases:
            run_progression_test(
                name=test["name"],
                ground_truth=test["function"],
                bounds=test["bounds"],
                control_points=test["control_points"],
                filename=test["filename"],
                min_points=3,
                extension=test["extension"],
                columns=3,
            )

        for test in ei_test_cases:
            run_ei_progression_test(
                name=test["name"],
                ground_truth=test["function"],
                bounds=test["bounds"],
                initial_points=test["initial_points"],
                ei_iterations=test["ei_iterations"],
                filename=test["filename"],
                min_points=4,
                extension=test["extension"],
                columns=4,
                optimize_restarts=25,
                xi=0.0,
                seed=test["seed"],
            )

    except KeyboardInterrupt:
        plt.close("all")
        print("\nStopped by user.")