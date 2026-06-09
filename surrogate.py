import numpy as np
from scipy.optimize import minimize
from smt.surrogate_models import KRG


class Surrogate:
    def __init__(
        self,
        dim=1,
        bounds=None,
        min_points=None,
        theta0=None,
        corr="squar_exp",
        optimize_restarts=10,
        seed=None,
    ):
        self.dim = dim
        self.bounds = bounds
        self.min_points = min_points if min_points is not None else max(2, dim + 1)
        self.theta0 = theta0 if theta0 is not None else [1e-2] * dim
        self.corr = corr
        self.optimize_restarts = optimize_restarts
        self.rng = np.random.default_rng(seed)

        self.X = []
        self.Y = []

        self.model = None
        self.is_trained = False

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

    def optimize(self) -> tuple[np.ndarray, float]:
        if len(self.Y) == 0:
            c0 = self._default_control()
            return c0, self(c0)

        best_seen_idx = int(np.argmin(self.Y))
        best_seen = self.X[best_seen_idx].copy()

        bounds = self._get_bounds(best_seen)

        starts = [best_seen, self.X[-1].copy()]

        for idx in np.argsort(self.Y)[: min(3, len(self.Y))]:
            starts.append(self.X[int(idx)].copy())

        for _ in range(self.optimize_restarts):
            starts.append(np.array([self.rng.uniform(lo, hi) for lo, hi in bounds]))

        best_c = best_seen.copy()
        best_z = self(best_c)

        for start in starts:
            result = minimize(
                lambda c: self(c),
                start,
                method="L-BFGS-B",
                bounds=bounds,
            )

            c_candidate = self._clip(result.x)
            z_candidate = self(c_candidate)

            if z_candidate < best_z:
                best_c = c_candidate
                best_z = z_candidate

        return best_c, float(best_z)

    def variance(self, x) -> float:
        x = self._as_vector(x)

        if not self.is_trained:
            return float("inf")

        x_scaled = self._scale(x.reshape(1, -1))
        var = self.model.predict_variances(x_scaled)
        return max(0.0, float(var[0, 0]))

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
        X = np.asarray(X, dtype=float)

        if self.bounds is not None:
            lo = np.array([b[0] for b in self.bounds], dtype=float)
            hi = np.array([b[1] for b in self.bounds], dtype=float)
            width = hi - lo
            width[width == 0.0] = 1.0
            return (X - lo) / width

        center = np.mean(np.vstack(self.X), axis=0)
        scale = np.std(np.vstack(self.X), axis=0)
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


if __name__ == "__main__":
    # Smoke test: known minimum near c = 1.05
    s = Surrogate(
        dim=1,
        bounds=[(0.75, 1.25)],
        min_points=2,
        optimize_restarts=20,
        seed=1,
    )

    for c in [0.75, 0.9, 1.0, 1.15, 1.25]:
        z = (c - 1.05) ** 2
        s.update([c], z)

    c_best, z_best = s.optimize()

    print("best c:", c_best)
    print("predicted z:", z_best)
    print("variance:", s.variance(c_best))