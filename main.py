import numpy as np

from surrogate import Surrogate
from forward import ForwardModel
from reverse import ReverseModel
from thruster_controller import ThrusterController

from concurrent.futures import ThreadPoolExecutor

INF = float("inf")

def _validate_in_range(val, name, lo=float("-inf"), hi=float("inf")):
    if val < lo or val > hi:
        raise ValueError(f"{name} must be between {lo} and {hi}! Got: {val}")
    return val

#  TODO
# - Logging of what commands we send, and what data we get back
# - Could save all reverse + forward samples in some per-iteration folder
# - Command thruster
# - Build + update surrogate in 1-3 D
# - Integrate reverse model
# - Integrate constraints (ideally smooth, differentiable)

def log_penalty(x, lb, ub, penalty_strength = 5e-2):
    """Evaluate a smoothly differentiable constraint penalty function that is ~zero far from the constraints and ~inf close to them"""
    # Using logarithmic barrier function: https://en.wikipedia.org/wiki/Barrier_function
    # We should save the actual predicted metric separately from the combined objective function 
    # Here's a desmos link demonstrating the functions in 1D: https://www.desmos.com/calculator/wjk4t1m2z5
    midpoint = 0.5 * (lb + ub)
    midpoint_f = -np.log(midpoint-lb) - np.log(ub - midpoint)
    penalty_per_dim = -np.log(x-lb) - np.log(ub-x) - midpoint_f
    total_penalty = penalty_strength * np.mean(penalty_per_dim)
    return total_penalty

class DiffusionController:
    def __init__(
            self,
            c0,                                           # Starting control values. TODO: also pass specification listing what each index is (or pass as dict)
            controller: ThrusterController,               # Thruster controller
            forward: ForwardModel| None = None,           # Forward model, mapping controls + state -> new state + data.
            surrogate: Surrogate | None = None,           # Surrogate model type, TODO: define API.
            metric=None,                                  # Function from data -> reals, positive definite.
            reverse: ReverseModel | None =None,           # Reverse model, maps data to several (control, state) estimates.
            num_reverse_samples=1,                        # Number of samples to draw from the reverse model fn.
            forwards_per_reverse=1,                       # How many forward model samples to draw per reverse sample
            model_trust=1.0,                              # Starting model trust parameter (default 1).
            trust_relaxation=0.5,                         # Under-relaxation parameter for updating model trust.
            control_lb = None,                            # Lower bounds for all control variables
            control_ub = None,                            # Upper bounds for all control variables
            penalty_strength = 5e-2,                      # The scale factor of the logarithmic penalty function used to avoid the bounds
        ):

        self.step = 0

        # TODO: decide on a way to decrement step_scale over time
        self.step_scale = 1.0
        self.control_point = np.array(c0),
        self.control_dim = len(c0)

        # Check length and contents of bounds
        self.penalty_strength = penalty_strength
        self.control_lb = control_lb if control_lb else [-INF for _ in range(self.control_dim)]
        self.control_lb = control_lb if control_lb else [INF for _ in range(self.control_dim)]
        assert len(control_lb) == self.control_dim, f"Control lower bound must have length {self.control_dim} to match control dimension, but got {len(control_lb)}"
        assert len(control_ub) == self.control_dim, f"Control upper bound must have length {self.control_dim} to match control dimension, but got {len(control_lb)}"
        assert np.all(self.control_ub >= self.control_lb), f"Control upper bound must be >= lower bound for all variables. Got lb: {control_lb} and ub: {control_ub}."

        # Set up the three main components of the controller/optimizer
        self.controller = controller
        self.forward = forward
        self.reverse = reverse
        self.surrogate = surrogate

        # Check metric
        if metric is None:
            raise ValueError("Metric must be specified! This should be a function of the data which returns a positive number.")

        # TODO: integrate constraints into auxilliary metric function
        # something like the following, making sure to return both values (z and z')
        # this is important because we should use z' for picking control points but z for evaluating
        # how well we predicted things for model trust checking
        # Would need to also update calls to self.metric to pass in the corresponding control point
        # -----------------------------------
        # def metric_with_constraint(c, y):
        #     z = metric(y)
        #     z_prime = log_penalty(c, self.control_lb, self.control_ub, self.constraint_strength)
        #     return z, z_prime
        #
        # self.metric = metric_with_constraint
        # -----------------------------------
        self.metric = metric

        self.num_reverse_samples = _validate_in_range(num_reverse_samples, "Num reverse samples", lo=1)
        self.forwards_per_reverse = _validate_in_range(forwards_per_reverse, "Forwards per reverse", lo=1)

        # Asynchronous executor, used to spin off forward model calls
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Previous model and surrogate predicted metrics, used to update model trust
        self.z_pred_surr = None
        self.z_pred_model = None
        self.z_pred_model_future = None

        self.model_trust = _validate_in_range(model_trust, "Model trust", 0, 1)
        self.trust_relaxation = _validate_in_range(trust_relaxation, "Trust relaxation", 0, 1)

    def __del__(self):
        # Destructor shuts down the ThreadPoolExecutor
        self.executor.shutdown(wait=True)

    def update_model_trust(self, z):
        if self.z_pred_model is None or self.z_pred_surr is None:
            # If we're in the first loop, we don't have previous predictions,
            # so we can't update the trust parameter
            if self.surrogate is None:
                # No surrogate specified, we have to trust the model
                self.model_trust = 1.0
            elif self.forward is None or self.reverse is None:
                # No model, we have to trust the surrogate
                self.model_trust = 0.0
            else:
                # Use inverse distance weighting to interpolate between surrogate and modeling
                # The distance is evaluated as the difference between predicted and observed 
                # z for a specified control action
                beta = self.trust_relaxation
                dz_surr = np.abs(z - self.z_pred_surr)
                dz_model = np.abs(z - self.z_pred_model)
                new_trust = (1.0/dz_model)**2 / (1.0/dz_model**2 + 1.0/dz_surr**2)
                self.model_trust = beta * new_trust + (1 - beta) * self.model_trust

        return self.model_trust

    def update(self, c):
        # Control thruster to the given control point
        self.control_point = c
        self.controller.control_to(c)
        y = self.controller.take_data()

        # Evaluate metric on data
        z = self.metric(y, **self.metric_args)

        # Await results of forward model from before and average the metrics
        if self.z_pred_model_future is not None:
            z_pred_model_results = self.z_pred_model_future.result()
            self.z_pred_model = 0.0
            count = 0
            for result in z_pred_model_results:
                if result is None:
                    continue
                _, yk = result
                self.z_pred_model += self.metric(yk)
                count += 1
            self.z_pred_model /= count

        # Update model trust using previous model predictions
        T = self.update_model_trust(z)

        if self.surrogate is not None:
            # Update surrogate model with new data point
            self.surrogate.update(c, z)

            # Perform local optimization on surrogate
            # to find optimal control location
            c_surr, _ = self.surrogate.optimize()
        else:
            c_surr, _ = np.zeros(self.control_dim), float("inf")

        if self.reverse is not None and self.forward is not None:
            # Get state and control estimates
            reverse_samples = self.reverse(y, c, n=self.num_reverse_samples, **self.reverse_args)

            # Propose one or more control actions 
            proposed_controls = []
            for (xk, ck) in reverse_samples:
                for _ in range(self.forwards_per_reverse):
                    # Proposed control action is a mixture of surrogate direction and random noise
                    # Balances exploration / exploitation
                    # TODO: Apply constraints here?
                    rand_direction = np.random.standard_normal(self.control_dim)
                    surr_direction = c_surr - ck
                    c_prop = ck + self.step_scale * ((1 - T) * surr_direction + rand_direction)
                    proposed_controls.append([xk, c_prop])

            # Evaluate forward model for each state estimate / control action pair
            # We do this asynchronously but immediately await the result
            # The async part is thus not really necessary, but I am leaving it in in case we 
            # might want to interleave additional work in the future.
            future = self.executor.submit(self.forward, proposed_controls)
            forward_samples = future.result()

            # Eval metrics and weight control proposals based on metrics
            # This could be in the previous loop if serial, but
            # that loop should be made parallel so I'm keeping it separate
            numerator = np.zeros(self.control_dim)
            denominator = 0.0
            for forward_sample, control_prop in zip(forward_samples, proposed_controls):
                if forward_sample is None:
                    # This occurs for simulation failures and other invalid states
                    continue
                (_, ck) = control_prop
                (xk_new, yk) = forward_sample
                zk = self.metric(yk, **self.metric_args)
                forward_sample.append(zk)

                numerator += control_prop / zk**2
                denominator += 1.0 / zk**2

            # Get final model-proposed control point
            c_model = numerator / denominator
        else:
            c_model = np.zeros(self.control_dim)
            proposed_controls = []

        # Once we have the model and surrogate-proposed controls in hand,
        # we can determine the final control action by interpolating between
        # the two based on model trust
        if self.surrogate is None:
            c_final = c_model
        elif self.forward is None or self.reverse is None:
            c_final = c_surr
        else:
            c_final = (1 - T) * c_surr + T * c_model

        # Predict surrogate output at this point
        self.z_pred_surr = self.surrogate(c_final) if self.surrogate else float("inf")

        # Predict mean model output asynchronously (so we can simultaneously command the thruster)
        final_controls = [(xk, c_final) for xk, _ in reverse_samples]
        self.z_pred_model_future = self.executor.submit(self.forward, final_controls)

        self.step += 1

        # Return final proposed control action
        return c_final