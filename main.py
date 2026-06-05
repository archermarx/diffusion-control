import os

import numpy as np
import matplotlib.pyplot as plt

from forward import ForwardModel
from reverse import ReverseModel
from thruster_controller import SimulationController
from control_loop import DiffusionController

from hall_diffusion.utils.thruster_data import invert_fft_vector

def rms_amplitude(x):
    mean = np.mean(x)
    x_centered = x - mean
    return np.sqrt(np.mean(x_centered**2))

def metric(y):
    return rms_amplitude(y["discharge_current"]["signal"])

#
config = "thrusters/h9.json"
dataset_dir = "inputs"

# Forward model
forward = ForwardModel(
    case_config=config,
    dataset_dir="h9_ref",
    num_workers=8,
    verbose=False,
    duration=2e-3,
)

# Reverse model
reverse = ReverseModel(
    config_file="reverse_model/sample_h9.toml",
    model="reverse_model/h9/checkpoint.pth.tar",
    sample_dir="reverse_model/samples",
    num_steps = 128,
    num_samples = 32,
)

# Controller
data_file = "data.json"
thruster = SimulationController(
    thruster_config=config,
    dataset_dir="inputs",
    data_file=data_file,
    dir=None,
)
if not os.path.exists(data_file):
    with open(data_file, "w") as fd:
        print("", file=data_file)

# Define initial condition
mdot_a = 11.8e-6
Vd = 400
c0 = {
    "magnetic_field_scale": 1.1,
    "anode_mass_flow_rate_kg_s": mdot_a,
    "discharge_voltage_v": Vd
}

# Collect ground truth data using forward model
B_min = 0.75
B_max = 1.25
B_vals = np.linspace(0.75 * B_min, 1.25 * B_max, 64)

out_file = "Id_bfield.npz"
if not os.path.exists(out_file):
    inputs = [(None, [b, mdot_a, Vd]) for b in B_vals]
    time = np.linspace(0, 1, 1001)
    data = forward(inputs)
    currents = [invert_fft_vector(time, f) for _, f in data]
    np.savez(out_file, current=currents)

current_data = np.load(out_file)["current"]
ampls = [rms_amplitude(c) for c in current_data]

# Initialize diffusion controller
controller = DiffusionController(
    c0=c0,
    control_vars=["magnetic_field_scale"],
    controller=thruster,
    forward=forward,
    metric=metric,
    reverse=reverse,
    surrogate=None,
    forwards_per_reverse=1,
    trust_relaxation=0.5,
    control_lb = [B_min],
    control_ub = [B_max],
    penalty_strength=5e-2,
)

out_file = "optim.json"
controller.save_to_file(out_file)
start_iter = len(controller.zs)

for i in range(2):
    print(f"iteration {start_iter + i + 1}")
    controller.step()
    controller.save_to_file(out_file)

    cs = np.array(controller.cs)[:, 0]
    alphas = np.linspace(0.25, 1.0, len(cs)) if len(cs) > 2 else [1.0]

    fig, ax = plt.subplots()
    ax.set(xlabel="Magnetic field scale", ylabel="Metric")
    ax.plot(B_vals, ampls, label="Ground truth")
    ax.axvline(B_min, color='black', linestyle='--')
    ax.axvline(B_max, color='black', linestyle='--')
    ax.plot(cs, controller.zs, color='red', zorder=9)
    ax.scatter(cs, controller.zs, color='red', alpha = alphas, zorder=10)
    ax.legend()
    fig.savefig("Id_bfield.png", dpi=200)
