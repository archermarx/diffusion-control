import numpy as np

class Surrogate:
    def __init__(self, dim=1):
        self.dim=dim

    # Thin plate spline
    # Radial basis function
    # Gaussian Process / kriging <- includes uncertainty

    # Surrogate Modeling Toolbox 
    # Surrogates should incorporate constraints

    def __call__(self, x) -> float:
        return 0.0

    def update(self, x, y) -> None:
        pass

    def optimize(self) -> tuple[np.ndarray, float]:
        return np.zeros(self.dim), 0.0


if __name__ == "__main__":
    # Test out basic surrogate construction functionality
    pass