import copy
import os
import shutil
import tomllib

from pathlib import Path

import numpy as np
import torch

from hall_diffusion import sample as sampling
from hall_diffusion.utils.thruster_data import ThrusterDataset

class ReverseModel:
    def __init__(
            self,
            model: str | Path,
            config_file: str | Path,
            sample_dir: str | Path,
            num_samples: int,
            num_steps: int,
            replace_samples: bool = False,
            verbose: bool = False,
        ):
        self.model = Path(model)
        self.sample_dir = Path(sample_dir)
        self.replace_samples = replace_samples
        self.verbose = verbose
        self.sample_dir_created = False
        self.num_samples = num_samples
        self.num_steps = num_steps

        # Read config file and extract some useful information
        with open(config_file, "rb") as fd:
            self.config = tomllib.load(fd)

        self.base_sim = self.config["observation"]["base_sim"]
        self.stddev = self.config["observation"]["stddev"]
        self.config.pop("observation")

        self.dataset = ThrusterDataset(self.base_sim, scalars_in_tensor=True, fourier_features=True)

    def _make_sample_dir(self):
        if not self.sample_dir_created:
            if os.path.exists(self.sample_dir) and self.replace_samples:
                shutil.rmtree(self.sample_dir)
            os.makedirs(self.sample_dir, exist_ok=True)
        self.sample_dir_created = True

    def build_config(self, data, num_samples, num_steps):
        # TODO: incorporate data + design struct for data
        config = copy.deepcopy(self.config)
        
        observation = {
            "base_sim": self.base_sim,
            "stddev": self.stddev,
            "fields": {},
        }

        for key, val in data.items():
            if key == "discharge_current":
                continue

            if isinstance(val, dict):
                mean = val["mean"]
                std = val.get("std", self.stddev)
            else:
                mean = val
                std = self.stddev

            observation["fields"][key] = {
                "x": "all",
                "y": [mean],
                "stddev": std,
                "normalized": False,
            }

        config["observation"] = observation
        config["out_dir"] = str(self.sample_dir)
        config["num_samples"] = num_samples
        config["num_steps"] = num_steps
        return config
    
    def get_scalar_params(self, x):
        if len(x.shape) == 2:
            x = x[None, ...]

        params = {}
        for k in self.dataset.params():
            c_ki = self.dataset.get_field(x, k)
            params[k] = np.mean(c_ki, axis=1)

        return params

    def __call__(self, data, num_samples=None, num_steps=None):
        num_samples = num_samples if num_samples else self.num_samples        
        num_steps = num_steps if num_steps else self.num_steps

        # Runs the diffusion model
        # Outputs samples to sample_dir
        # Loads samples, returns for use by forward model
        config = self.build_config(data, num_samples, num_steps)
        samples_allsteps = sampling.infer(self.model, config, True, True)
        samples = samples_allsteps[-1, ...]

        state_ests = self.dataset.norm.denormalize_tensor(samples)

        
        return state_ests

if __name__ == "__main__":

    controls = {
        "discharge_voltage_v": {
            "mean": 300.0,
            "std": 1.025,
        },
        "anode_mass_flow_rate_kg_s": {
            "mean": 11e-6,
            "std": 1.025,
        },
        "magnetic_field_scale": {
            "mean": 1.0,
            "std": 1.025,
        }
    }

    data = {
        "cathode_coupling_voltage_v": {
            "mean": 30.0,
            "std": 1.025,
        }
    }

    data.update(controls)

    model = ReverseModel(
        model = "reverse_model/h9/checkpoint.pth.tar",
        config_file = "reverse_model/sample_h9.toml",
        sample_dir = "reverse_samples",
        replace_samples = True,
        num_samples = 16,
        num_steps = 32,
    )


    samples = model(data)

    print(model.get_scalar_params(samples))

    param_ests = []
    # Denormalize tensors and extract controls
    for key in data:
        c_ki = model.dataset.get_field(samples, key)
        param_ests.append(np.mean(c_ki, axis=1))
    param_ests = np.array(param_ests).T

    # Check that samples obey given controls
    for (i, control) in enumerate(data):
        mean = np.mean(param_ests[:, i])
        std = np.std(param_ests[:, i])
        print(f"{control}: mean: {mean:.3e}, std: {std:.3e}")
        assert std < 0.01 * mean

    assert(len(param_ests) == model.num_samples)

    assert os.path.exists(model.sample_dir)
    assert len(os.listdir(os.path.join(model.sample_dir, "data"))) == model.num_samples

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1,1)
    
    ax.plot(model.dataset.get_field(samples.T, "nu_an"))
    ax.set(
        #ylim=(0,None),
        yscale='log',
    )

    fig.savefig("potential.png")