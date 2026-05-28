import numpy as np

from surrogate import Surrogate
from forward import ForwardModel
from reverse import ReverseModel
from thruster_controller import ThrusterController

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

class DiffusionController:
    def __init__(
            self,
            c0,                                           # Starting control values. TODO: also pass specification listing what each index is (or pass as dict)
            controller: ThrusterController,               # Thruster controller
            forward: ForwardModel| None = None,           # Forward model, mapping controls + state -> new state + data.
            forward_args=None,                            # Dict of extra args to pass to forward model fn.
            surrogate: Surrogate | None = None,           # Surrogate model type, TODO: define API.
            metric=None,                                  # Function from data -> reals, positive definite.
            metric_args=None,                             # Dict of extra args to pass to metric fn.
            reverse: ReverseModel | None =None,           # Reverse model, maps data to several (control, state) estimates.
            num_reverse_samples=1,                        # Number of samples to draw from the reverse model fn.
            forwards_per_reverse=1,                       # How many forward model samples to draw per reverse sample
            reverse_args=None,                            # Dict of extra args to pass to reverse model.
            model_trust=1.0,                              # Starting model trust parameter (default 1).
            trust_relaxation=0.5,                         # Under-relaxation parameter for updating model trust.
        ):

        self.step = 0
        self.step_scale = 1.0
        self.controller = controller

        self.control_point = np.array(c0),
        self.control_dim = len(c0)
        self.model_trust = _validate_in_range(model_trust, "Model trust", 0, 1)
        self.trust_relaxation = _validate_in_range(trust_relaxation, "Trust relaxation", 0, 1)

        self.forward = forward
        self.reverse = reverse
        self.surrogate = surrogate

        # Previous model and surrogate predicted metrics, used to update model trust
        self.z_pred_surr = None
        self.z_pred_model = None

        if metric is None:
            raise ValueError("Metric must be specified! This should be a function of the data which returns a positive number.")
        self.metric = metric

        self.forward_args = forward_args if forward_args else {} 
        self.reverse_args = reverse_args if reverse_args else {}
        self.metric_args = metric_args if metric_args else {}

        self.num_reverse_samples = _validate_in_range(num_reverse_samples, "Num reverse samples", lo=1)
        self.forwards_per_reverse = _validate_in_range(forwards_per_reverse, "Forwards per reverse", lo=1)

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
        # TODO: apply constraints here?
        z = self.metric(y, **self.metric_args)

        # Update model trust
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
            # TODO: set up joblib/multiprocessing infrastructure to do this in parallel
            forward_samples = []
            for (xk, ck) in proposed_controls:
                (xk_new, yk) = self.forward(xk, ck, **self.forward_args)
                forward_samples.append([xk_new, yk])

            # Eval metrics and weight control proposals based on metrics
            # This could be in the previous loop if serial, but
            # that loop should be made parallel so I'm keeping it separate
            numerator = np.zeros(self.control_dim)
            denominator = 0.0
            for forward_sample, control_prop in zip(forward_samples, proposed_controls):
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

        # TODO: apply constraints here?

        # Predict surrogate output at this point
        z_surr = self.surrogate(c_final) if self.surrogate else float("inf")

        # TODO: write async code to do this (using asyncio probably)
        # NOTE: should write helper function to spawn a bunch of forward model evaluations asynchronously
        z_model = 0.0
        if self.forward is not None:
            for xk, _ in proposed_controls:
                _, y_final = self.forward(xk, c_final, **self.forward_args)
                z_model += self.metric(y_final, **self.metric_args)
            z_model /= len(proposed_controls)

        self.z_pred_model = z_model
        self.z_pred_surr = z_surr
        self.step += 1

        return c_final