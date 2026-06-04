import json
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import numpy as np

from hall_diffusion.utils.thruster_data import ThrusterDataset, invert_fft_vector

def getkey_deep(d, keystr):
    """Return the value of a key of dictionary d multiple levels deep, with each level separated by a period (.)"""
    key_seq = keystr.split(".")
    curr = d
    for k in key_seq:
        curr = curr[k]
    return curr

def setkey_deep(d, keystr, val, new_ok=False):
    """Set a key of dictionary d multiple levels deep, with each level separated by a period (.)"""
    key_seq = keystr.split(".")
    curr = d
    for k in key_seq[:-1]:
        curr = curr[k]

    if key_seq[-1] in curr or new_ok:
        curr[key_seq[-1]] = val
    else:
        raise KeyError(f"Key {keystr} not in dictionary and new_ok is false!")

class ForwardModel:
    def __init__(self,
            case_config: str | Path,
            controls: list[str],
            dataset_dir: str | Path,
            duration: float = 1e-3,
            num_workers: int =1,
            verbose: bool = False
        ):
        
        # Maps our diffusion model / control keys to specific entries in the HT.jl config dict
        self.keymap = {
            "anode_mass_flow_rate_kg_s": "config.anode_mass_flow_rate",
            "neutral_velocity_m_s": "config.neutral_velocity",
            "discharge_voltage_v": "config.discharge_voltage",
            "wall_loss_scale": "config.wall_loss_model.loss_scale",
            "cathode_coupling_voltage_v": "config.cathode_coupling_voltage",
            "magnetic_field_scale": "config.magnetic_field_scale",
        }

        self.num_workers = num_workers  # How many parallel workers/threads to employ for running simulations
        self.controls = controls        # vector of control actions
        self.duration = duration        # How long (in s) to run the forward model
        self.verbose = verbose          # Whether HT.jl will print info about simulation success/failure

        # Dataset object, useful for normalizing, denormalizing, and loading data
        self.dataset = ThrusterDataset(dataset_dir, scalars_in_tensor=True, fourier_features=True)

        # Read thruster configuration file to grab geometry, propellant, and wall material
        with open(case_config, "rb") as fd:
            cfg = json.load(fd)
            self.thruster = cfg["thruster"]
            self.wall_material = cfg["wall_material"]
            self.propellant = cfg["propellant"]

    def _base_config(self):
        L_ch = self.thruster["geometry"]["channel_length"]
        domain = (0.0, 3.2 * L_ch)
        num_cells = 128
        edges = np.linspace(domain[0], domain[1], num_cells+1)
        z_cell = 0.5 * (edges[:-1] + edges[1:])
        f_anom_base = 0.0625 * np.ones(num_cells)
        f_anom_base[z_cell < L_ch] = 0.00625

        # All values marked `placeholder` will be replaced from the simulation state
        # and/or from controls
        config = dict(
            thruster = self.thruster,
            domain = domain,
            propellant = self.propellant,
            neutral_velocity = 300.0,      # placeholder
            ncharge = 3,                   # placeholder
            anode_mass_flow_rate = 5e-6,   # placeholder
            discharge_voltage = 300.0,     # placeholder
            cathode_coupling_voltage=0.0,  # placeholder
            magnetic_field_scale=1.0,      # placeholder
            anom_model = dict(             # placeholder
                type="MultiLogBohm",
                zs=list(z_cell),
                cs=list(f_anom_base),
            ),
            wall_loss_model = dict(
                type="WallSheath",
                material = self.wall_material,
                loss_scale=1.0,           # placeholder
            ),
            ion_wall_losses = True,
            filter_circuit = dict(
                type="NoCircuit",
                elements = [],
                limit_current=100.0
            ),
        )
        
        simulation = dict(
            duration=self.duration,
            dt=1e-9,
            grid=dict(
                type="EvenGrid",
                num_cells=num_cells,
            ),
            verbose=self.verbose,
            print_errors=self.verbose,
            num_save=2001
        )

        return {
            "config": config,
            "simulation": simulation,
            "postprocess": {},
        }
    
    def _make_config(self, state, control):
        """Generate a valid HallThruster.jl config dictionary corresponding to the given state and control"""
        cfg = self._base_config()

        if state is not None:
            # Get anomalous collision frequency from tensor
            nu_anom = self.dataset.get_field(state, "nu_an", action="denormalize")
            B = self.dataset.get_field(state, "B", action="denormalize")
            wce = 1.6e-19 * B / 9.1e-31
            c_anom = nu_anom / wce
            c_anom = c_anom.squeeze(0).tolist()
            setkey_deep(cfg, "config.anom_model.cs", c_anom)

            # Extract 6 scalar params from tensor
            for (oldkey, newkey) in self.keymap.items():
                val = self.dataset.get_field(state, oldkey, action="denormalize")
                setkey_deep(cfg, newkey, val.mean().item())

        # Extract controls
        for keystr, control_val in zip(self.controls, control):
            setkey_deep(cfg, self.keymap[keystr], control_val)

        return cfg

    def __call__(self, inputs, dir=None, delete_dir=True, output_files=None):
        if dir is None:
            # Generate temporary directory to hold configs written by python (tmp_dir/inputs)
            # and outputs from julia (tmp_dir/outputs)
            tmp_dir = tempfile.mkdtemp()
        else:
            tmp_dir = dir
            if os.path.exists(tmp_dir) and delete_dir:
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)

        # Try-finally block helps ensure tmp_dir gets cleaned up
        try:
            input_dir = os.path.join(tmp_dir, "inputs")
            output_dir = os.path.join(tmp_dir, "outputs")
            output_data_dir = os.path.join(output_dir, "data")

            os.makedirs(input_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(output_data_dir, exist_ok=True)

            # Generate a UUID for each (state, control) pair so we can later find the corresponding outputs
            if output_files is None:
                ids = [uuid.uuid4() for _ in inputs]
            else:
                assert len(inputs) == len(output_files)
                ids = output_files

            for (id, (x, c)) in zip(ids, inputs):
                # Generate config corresponding to each (state, control) pair and write it to JSON
                config = self._make_config(x, c)
                tmp_file = os.path.join(input_dir, f"{id}.json")
                with open(tmp_file, "w") as fd:
                    json.dump(config, fd)

            # Invoke subprocess call to julia function
            # Julia reads input config files from `input_dir` and puts outputs, unnormalized and in .npz format, in output_dir
            # Failures and such (following the same criteria as we use to prune sims in normalize_data) will not be written
            subprocess.run([
                "julia", 
                "-t", str(self.num_workers),
                '--project="."',
                "--startup=no",
                "run_forward.jl",
                input_dir,
                output_data_dir,
            ])

            # Write metadata (including normalization info to output file)
            self.dataset.write_metadata(output_dir)

            # We read the un-normalized output data and calculate the data metric and states from it
            # We then return these as (new_state, y) pairs, or None if there was a failure
            output_dict = {}
            output_dataset = ThrusterDataset(
                output_dir,
                scalars_in_tensor=self.dataset.scalars_in_tensor,
                fourier_features=self.dataset.fourier_features,
            )

            for file, fourier, state in output_dataset:
                output_dict[file] = (state, fourier)

            outputs = []
            for id in ids:
                output_file = f"{id}.npz"
                if output_file in output_dict:
                    outputs.append(output_dict[output_file])
                else:
                    outputs.append(None)
        finally:
            # Clean up the temporary directory
            shutil.rmtree(tmp_dir)

        return outputs

if __name__ == "__main__":
    d = dict(a = dict(b = dict(c = dict(d = 10, e = 5))))
    assert(getkey_deep(d, "a.b.c.d") == 10)

    setkey_deep(d, "a.b.c.d", 2)
    assert(getkey_deep(d, "a.b.c.d") == 2)

    model = ForwardModel(
        "thrusters/h9.json",
        controls = [
            #"anode_mass_flow_rate_kg_s",
            #"discharge_voltage_v",
            "magnetic_field_scale",
        ],
        dataset_dir="inputs",
        verbose=True,
        num_workers=8,
        duration=2e-3,
    )

    field_scale = [0.75, 1.0, 1.25]
    controls = [[f] for f in field_scale]
    inds = np.random.choice(np.arange(len(model.dataset)), len(field_scale), replace=False)
    states = [state[None, ...] for _, _, state in [model.dataset[i] for i in inds]]

    state_control_pairs = list(zip(states, controls))
    outputs = model(state_control_pairs, dir="files")

    for o in outputs:
        state, fourier = o
        bfield_scale = model.dataset.get_field(state[None, ...], "magnetic_field_scale").mean().item()

        t = np.linspace(0, model.duration / 2, 1001)
        signal = invert_fft_vector(t, fourier)
        print(f"{np.mean(signal)=}")

        import matplotlib.pyplot as plt
        plt.plot(t, signal)
        plt.savefig("signal.png")
