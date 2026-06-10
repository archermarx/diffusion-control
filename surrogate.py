import numpy as np
import time 
import threading
import matplotlib.pyplot as plt

# Scipy imports
from scipy.stats import norm

# SMT imports
from smt.surrogate_models import KRG
from smt.applications.ego import EGO
from smt.design_space.design_space import DesignSpace, FloatVariable
from smt.problems import Rosenbrock


# TODO: Add in some way to do constraint handeling for controls that would violate physical constraints?

class Surrogate:
    def __init__(self, dim=2, fun=Rosenbrock(ndim=2), n_iter=20, x_lims=[0, 100], y_lims=[0, 100], seed=42):
        self.dim=dim
        self.fun=fun

        # Optimization
        self.n_iter = n_iter # 10*dim is recommended from EGO paper
        self.x_lims = x_lims
        self.y_lims = y_lims
        self.seed = seed
        self.design_space = DesignSpace(design_variables=[FloatVariable(self.x_lims[0], self.x_lims[1]), FloatVariable(self.y_lims[0], self.y_lims[1])], seed=self.seed)
        self.criterion = "EI"

        self.active_model = KRG(
            design_space=self.design_space,  # <-- Moved here!
            theta0=[1e-2] * self.dim, 
            print_global=False
        )
        self.active_model.set_training_values(np.zeros((1, self.dim)), np.zeros((1, 1)))  # initialize surrogate with dummy data

        self.X_data = []  # list to store input data [C, Z] and state
        self.Y_data = []  # list to store output data [C_next, Z_next]

        self.ego = None


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

        if not self.X_data: return 0.0

        print(f"Surrogate called with input: {state_and_control}")  # debug print

        x = np.array(state_and_control).reshape(1, -1)  # reshape to 2D array for surrogate input

        y = self.active_model.predict_values(x)  # predict next state using surrogate model
        print(f"Predicted values: {y}")  # debug print

        return float(y[0, 0])  # TODO: Make sure you're just returning the metric (discharge current pk-pk) because the caller already has the control

    # Takes: the input [C, Z] and the output [C_next, Z_next] for training the surrogate
    def update(self, current_x, actual_y) -> None:
        """FAST THREAD: Non-blocking call to append data and kick off training."""
        
        # FIX: Sanitize inputs to guarantee uniform 1D shapes regardless of where they came from
        clean_x = np.array(current_x).flatten()
        clean_y = np.array(actual_y).flatten()

        print(f"Updating surrogate with new data: X={clean_x}, Y={clean_y}")  

        with self.lock:  
            self.X_data.append(clean_x)
            self.Y_data.append(clean_y)

        print(f"Current training data size: {len(self.X_data)} samples")  
        
        # Kriging requires at least 2 points to calculate variance and distance
        if len(self.X_data) < 2:
            print("Not enough data to train yet. Need at least 2 points.")
            return
        
        # If already training, skip starting another training thread
        if not self.is_training:
            self.is_training = True
            # Spawn another thread
            training_thread = threading.Thread(target=self._train_surrogate)
            training_thread.start()
    
    def _train_surrogate(self) -> None:
        """SLOW THREAD: Runs in the background to build the new model."""

        print("Starting surrogate training...")  # debug print

        try:
            with self.lock: # should have lock to access global data, but we don't want to hold it for the whole training process
                X_train_shadow = np.array(self.X_data)
                Y_train_shadow = np.array(self.Y_data)

            # Build model
            new_model = KRG(design_space=self.design_space, theta0=[1e-2] * self.dim, print_global=False)
            new_model.set_training_values(X_train_shadow, Y_train_shadow)
            new_model.train()

            print("Surrogate training completed.")  # debug print
            # Update the pointer? does python even have poitner??
            with self.lock:
                self.active_model = new_model  # update surrogate model to the newly trained model

        finally:            
            self.is_training = False  # reset training flag when done
    
    # Might not need
    # Should find the best fit Gaussian Process for the data and return the optimal control
    # Incorporates physical constraints (e.g. max voltage, max mass flow rate)
    def optimize(self) -> tuple[np.ndarray, float]:

        self.ego = EGO(surrogate=self.active_model,
                       criterion=self.criterion,
                       n_iter=self.n_iter,
                       seed=self.seed,
                       xdoe= np.array(self.X_data))
        

        x_opt, y_opt, _, x_data, y_data = self.ego.optimize(fun=self.fun)
        return x_opt, y_opt
    
    def plot_surrogate(self, X_plot, Y_plot, ax, title, x_opt=None):
        def EI(GP, points, f_min):
            pred = GP.predict_values(points)
            var = GP.predict_variances(points)
            var[var == 0.0] = 1e-12 
            args0 = (f_min - pred) / np.sqrt(var)
            args1 = (f_min - pred) * norm.cdf(args0)
            args2 = np.sqrt(var) * norm.pdf(args0)
            return args1 + args2

        Y_GP_plot = self.active_model.predict_values(X_plot)
        Y_GP_plot_var = self.active_model.predict_variances(X_plot)
        Y_EI_plot = EI(self.active_model, X_plot, np.min(self.Y_data))

        x_col = X_plot[:, 0]
        x_data_col = np.array(self.X_data)[:, 0]

        (true_fun,) = ax.plot(x_col, Y_plot, label="True function")
        (doe,) = ax.plot(x_data_col, self.Y_data, linestyle="", marker="s", markersize=10, color="blue", label="DOE (Samples)")
        (gp,) = ax.plot(x_col, Y_GP_plot, linestyle="--", color="g", label="GPR prediction")
        
        sig_plus = Y_GP_plot + 3 * np.sqrt(Y_GP_plot_var)
        sig_moins = Y_GP_plot - 3 * np.sqrt(Y_GP_plot_var)
        
        un_gp = ax.fill_between(
            x_col, sig_plus.flatten(), sig_moins.flatten(), alpha=0.3, color="g", label="99% confidence"
        )
        
        ax1 = ax.twinx()
        (ei,) = ax1.plot(x_col, Y_EI_plot, color="red", label="Expected Improvement")
        
        lines = [true_fun, doe, gp, un_gp, ei]

        # FIX: If we passed an optimal point, plot it as a large star!
        if x_opt is not None:
            x_star = x_opt.flatten()[0]  # Get the 1D slice coordinate
            y_star_pred = self.active_model.predict_values(np.array(x_opt).reshape(1, -1))[0, 0]
            (opt_pt,) = ax.plot(x_star, y_star_pred, '*', markersize=18, color='magenta', label="Intended Opt Point")
            lines.append(opt_pt)
            
        ax.set_title(title)
        ax.set_xlabel("x_0")
        ax.set_ylabel("y")
        ax1.set_ylabel("Expected Improvement (EI)")
        
        # Format the legend cleanly below the plot
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)


if __name__ == "__main__":
    # Test out basic surrogate construction functionality
    # 4 inputs: [V_d, mass_flow, B_field, current_state (discharge current pk-pk)]

    # Test surrogate code with data sampled from Rosenbrock function
    ndim = 2
    problem = Rosenbrock(ndim=ndim)

    num = 100
    x = np.ones((num, ndim))
    x[:, 0] = np.linspace(-2, 2.0, num)
    x[:, 1] = 0.0

    y = problem(x)

    yd = np.empty((num, ndim))
    for i in range(ndim):
        yd[:, i] = problem(x, kx=i).flatten()

    # print(y.shape)
    # print(yd.shape)

    # plt.plot(x[:, 0], y[:, 0])
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.show()



    surrogate = Surrogate(dim=ndim, fun=problem, x_lims=[-2, num])
    
    # Put TWO random points in the surrogate to initialize Kriging math safely
    idx1, idx2 = np.random.choice(num, 2, replace=False)
    surrogate.update(x[idx1], y[idx1])
    surrogate.update(x[idx2], y[idx2])
    
    print("Waiting for background thread to train initial model...")
    time.sleep(1.0) 

    # 1. Ask EGO where it WANTS to go, but don't evaluate it yet
    x_opt, y_opt = surrogate.optimize()
    print(f"Optimal control intended: {x_opt}")

    # 2. Set up the side-by-side figure
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))

    # 3. Plot the BEFORE state (Left)
    surrogate.plot_surrogate(x, y, ax=ax_left, title="Before EGO Update", x_opt=x_opt)

    # 4. Evaluate the true function at x_opt and update the surrogate
    surrogate.update(x_opt, problem(x_opt.reshape(1, -1)))  
    
    print("Waiting for background thread to incorporate new optimal point...")
    time.sleep(1.0) 

    # 5. Plot the AFTER state (Right)
    surrogate.plot_surrogate(x, y, ax=ax_right, title="After EGO Update")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25) # Make room for the legends
    plt.show()