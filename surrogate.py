import numpy as np
import time 
import threading
from smt.surrogate_models import KRG

class Surrogate:
    def __init__(self, dim=1):
        self.dim=dim

        self.active_model = KRG(theta0=[1e-2] * self.dim, print_global=False)  # surrogate model object
        self.active_model.set_training_values(np.zeros((1, self.dim)), np.zeros((1, 1)))  # initialize surrogate with dummy data
        self.active_model.train() # train surrogate on dummy data to get initial model parameters

        self.X_data = []  # list to store input data [C, Z] and state
        self.Y_data = []  # list to store output data [C_next, Z_next]

        # Threading
        self.lock = threading.Lock()  # lock for thread safety
        self.is_training = False  # flag to indicate if surrogate is currently being trained


    # Thin plate spline
    # Radial basis function
    # Gaussian Process / kriging <- includes uncertainty

    # Surrogate Modeling Toolbox 
    # Surrogates should incorporate constraints

    # Should take in the current state x of the system
    def __call__(self, state_and_control) -> float:
        """FAST THREAD: Called by the control loop. Takes microseconds."""
        x = np.array(state_and_control).reshape(1, -1)  # reshape to 2D array for surrogate input

        y = self.active_model.predict_values(x)  # predict next state using surrogate model

        return float(y[0, 0])  # TODO: Make sure you're just returning the metric (discharge current pk-pk) because the caller already has the control

    # Takes: the input [C, Z] and the output [C_next, Z_next] for training the surrogate
    def update(self, x, y) -> None:
        with self.lock:  # ensure thread safety when accessing surrogate model
            if self.is_training:
                # If surrogate is currently being trained, return a default control (e.g. zeros)
                return 0.0
            else:
                # Otherwise, predict the next state using the surrogate model
                y_pred = self.active_model.predict_values(x)
                # Here you would implement your optimization logic to find the best control based on y_pred
                # For simplicity, we will just return a placeholder control value

    # Might not need
    # Should find the best fit Gaussian Process for the data and return the optimal control
    # Incorporates physical constraints (e.g. max voltage, max mass flow rate)
    def optimize(self) -> tuple[np.ndarray, float]:
        return np.zeros(self.dim), 0.0


if __name__ == "__main__":
    # Test out basic surrogate construction functionality
    pass