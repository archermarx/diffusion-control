import numpy as np
import matplotlib.pyplot as plt
import time 
import threading

# Sciepy imports
from scipy.optimize import minimize
from scipy.stats import norm

# SMT imports
from smt.surrogate_models import KRG
from smt.applications.ego import EGO
from smt.design_space.design_space import DesignSpace, FloatVariable
from smt.problems import Rosenbrock, Branin

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

        self.design_space = DesignSpace(
            design_variables=[FloatVariable(bounds[0][0], bounds[0][1])], #FloatVariable(self.y_lims[0], self.y_lims[1])], 
            seed=seed)

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

        # Threading
        self.lock = threading.Lock()  # lock for thread safety
        self.is_training = False  # flag to indicate if surrogate is currently being trained
        self.training_thread = None  # will hold the training thread when it's running

    def __call__(self, x) -> float:
        x = self._as_vector(x)

        if len(self.Y) == 0:
            return 0.0

        if not self.is_trained:
            return float(min(self.Y))

        x_scaled = self._scale(x.reshape(1, -1))
        z = self.model.predict_values(x_scaled)
        return float(z[0, 0])

    def update(self, x, y, run_async=True) -> None:
        x = self._as_vector(x)
        y = float(y)

        with self.lock:
            self.X.append(x)
            self.Y.append(y)

        if run_async:
            # For live deployment: don't block the main thread
            if not self.is_training:
                self.is_training = True
                self.training_thread = threading.Thread(target=self._train_surrogate)
                self.training_thread.start()
        else:
            # For offline testing/animations: block and train synchronously
            self.is_training = True
            self._train_surrogate()

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
    

    def _train_surrogate(self):
        #print("Starting surrogate training...") 

        try:    
            with self.lock:
                X = np.vstack(self.X)
                Y = np.array(self.Y, dtype=float)

            X_scaled = self._scale(X)
            X_unique, Y_unique = self._remove_duplicate_points(X_scaled, Y)

            if len(Y_unique) < self.min_points:
                self.is_trained = False
                return

            model = KRG(
                design_space=self.design_space,
                theta0=self.theta0,
                corr=self.corr,
                print_global=False,
            )

            model.set_training_values(X_unique, Y_unique.reshape(-1, 1))
            model.train()

            with self.lock:
                self.model = model
                # ONLY set this to true if we successfully built the model
                self.is_trained = True 
        
        finally:
            #print("Surrogate training completed.") 
            # Only reset the training flag here
            self.is_training = False

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
    
    def negative_EI(x_test):
        # We need the current best physical metric to calculate Expected Improvement
        f_min = np.min(self.Y)

        """Calculates the negative EI because Scipy only minimizes."""
        x_test_reshaped = np.array(x_test).reshape(1, -1)
        pred = self.model.predict_values(x_test_reshaped)
        var = self.model.predict_variances(x_test_reshaped)
        
        var[var == 0.0] = 1e-12  # Prevent divide-by-zero at known sample points
        
        args0 = (f_min - pred) / np.sqrt(var)
        args1 = (f_min - pred) * norm.cdf(args0)
        args2 = np.sqrt(var) * norm.pdf(args0)
        
        ei = args1 + args2
        return -ei[0, 0]  # Return negative EI

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

    def _store_history(self):
        snapshot = {
            "X": [x.copy() for x in self.X],
            "Y": list(self.Y),
            "is_trained": self.is_trained,
        }
        self.history.append(snapshot)

