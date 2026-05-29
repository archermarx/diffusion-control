import copy
import os
import shutil
import tomllib

from pathlib import Path

import numpy as np

from hall_diffusion import sample as sampling
from hall_diffusion.utils.thruster_data import ThrusterDataset

class ReverseModel:
    def __init__(
            self,
            model: str | Path,
            config_file: str | Path,
            sample_dir: str | Path,
            controls: list[str],
            replace_samples: bool = False,
            verbose: bool = False,
        ):
        self.model = Path(model)
        self.sample_dir = Path(sample_dir)
        self.controls = controls
        self.replace_samples = replace_samples
        self.verbose = verbose
        self.sample_dir_created = False

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

    def build_config(self, data, controls, num_samples, num_steps):
        # TODO: incorporate data + design struct for data
        config = copy.deepcopy(self.config)
        
        observation = {
            "base_sim": self.base_sim,
            "stddev": self.stddev,
            "fields": {},
        }

        for key, val in zip(self.controls, controls):
            observation["fields"][key] = {
                "x": "all",
                "y": [val],
                "normalized": False,
            }

        config["observation"] = observation

        config["out_dir"] = str(self.sample_dir)
        config["num_samples"] = num_samples
        config["num_steps"] = num_steps
        return config

    def __call__(self, data, controls, num_samples, num_steps):
        # Runs the diffusion model
        # Outputs samples to sample_dir
        # Loads samples, returns for use by forward model
        config = self.build_config(data, controls, num_samples, num_steps)
        samples_allsteps = sampling.infer(self.model, config, True, True)
        samples = samples_allsteps[-1, ...]

        state_ests = self.dataset.norm.denormalize_tensor(samples)

        control_ests = []
        # Denormalize tensors and extract controls
        for key in self.controls:
            c_ki = self.dataset.get_field(state_ests, key)
            control_ests.append(np.mean(c_ki, axis=1))

        control_ests = np.array(control_ests).T
        
        return state_ests, control_ests

if __name__ == "__main__":

    controls = {
        "discharge_voltage_v": 300.0,
        "anode_mass_flow_rate_kg_s": 11e-6,
    }

    control_keys = list(controls.keys())
    control_vals = list(controls.values())

    model = ReverseModel(
        model = "reverse_model/h9_batch2/checkpoint.pth.tar",
        config_file = "reverse_model/sample_h9.toml",
        sample_dir = "reverse_samples",
        controls = control_keys,
        replace_samples = True,
    )

    num_samples = 16
    num_steps = 128

    samples, controls = model(None, control_vals, num_samples=num_samples, num_steps=num_steps)

    # Check that samples obey given controls
    for (i, control) in enumerate(model.controls):
        mean = np.mean(controls[:, i])
        std = np.std(controls[:, i])
        assert std < 0.01 * mean
        print(f"{control}: mean: {mean}, std: {std}")

    assert(len(controls) == num_samples)

    assert os.path.exists(model.sample_dir)
    assert len(os.listdir(os.path.join(model.sample_dir, "data"))) == num_samples
    # n, c, w = samples.shape
    # assert n == num_samples
    # assert w == 128




