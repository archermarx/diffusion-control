import json
import os
import shutil
import subprocess
import tempfile
import uuid
import sys

import numpy as np

#sys.path.append("/home/archermarks/projects/hall_diffusion/python")

#from utils.thruster_data import ThrusterDataset

from hall_diffusion.utils.thruster_data import ThrusterDataset

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
    def __init__(self, case_config, controls, dataset_dir, num_workers=1, verbose=False):
        # will need at least
        # 1. a ThrusterDataset (for normalizing and denormalizing as well as querying fields)
        # 2. a thruster information json (like we used for generating data)
        # 3. Some information about how controls are mapped to config keys
        # 4. Some information about data calculation
        self.num_workers = num_workers
        self.verbose = verbose
        self.controls = controls

        self.dataset = ThrusterDataset(dataset_dir, scalars_in_tensor=True, fourier_features=True)

        with open(case_config, "rb") as fd:
            cfg = json.load(fd)
            self.thruster = cfg["thruster"]
            self.wall_material = cfg["wall_material"]
            self.propellant = cfg["propellant"]

    def _base_config(self):
        L_ch = self.thruster["geometry"]["channel_length"]
        domain = (0.0, 3.2 * L_ch)
        num_cells = 64
        edges = np.linspace(domain[0], domain[1], num_cells+1)
        z_cell = 0.5 * (edges[:-1] + edges[1:])
        f_anom_base = 0.0625 * np.ones(num_cells)
        f_anom_base[z_cell < L_ch] = 0.00625

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
                loss_scale=1.0,                 # placeholder
            ),
            ion_wall_losses = True,
            filter_circuit = dict(
                type="NoCircuit",
                elements = [],
                limit_current=100.0
            ),
        )
        
        simulation = dict(
            duration=2e-3,
            dt=1e-9,
            grid=dict(
                type="EvenGrid",
                num_cells=num_cells,
            ),
            verbose=self.verbose,
            print_errors=self.verbose,
        )

        return {
            "config": config,
            "simulation": simulation,
            "postprocess": {},
        }
    
    def _make_config(self, state, control):
        # Generate a valid HallThruster.jl config dictionary corresponding to
        # the passed-in state and control
        cfg = self._base_config()

        # Extract state information (TODO: need to actually get this from tensor)
        ...

        # Extract controls
        for keystr, control_val in zip(self.controls, control):
            setkey_deep(cfg, keystr, control_val)

        return cfg

    def __call__(self, states, controls, dir=None):
        if dir is None:
            # Generate temporary directory to hold configs written by python (tmp_dir/inputs)
            # and outputs from julia (tmp_dir/outputs)
            tmp_dir = tempfile.mkdtemp()
        else:
            tmp_dir = dir
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
            ids = [uuid.uuid4() for _ in zip(states, controls)]

            for (id, x, c) in zip(ids, states, controls):
                # Generate config corresponding to each (state, control) pair and write it to JSON
                config = self._make_config(x, c)
                tmp_file = os.path.join(input_dir, f"{id}.json")
                with open(tmp_file, "w") as fd:
                    json.dump(config, fd)

            # Invoke subprocess call to julia function
            # Julia reads input config files from `input_dir` and puts outputs, unnormalized and in .npz format, in output_dir
            # Failures and such (following the same criteria as we use to prune sims in normalize_data) will not be written
            print(os.getcwd())
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
            outputs = []
            for id in ids:
                output_file = os.path.join(output_data_dir, f"{id}.npz")
                if os.path.exists(output_file):
                    contents = np.load(output_file)
                    # TODO: better to use dataset to load this
                    outputs.append((contents["params"], contents["data"], contents["fourier"], contents["perf"]))
                else:
                    outputs.append(None)
        finally:
            # Clean up the temporary directory
            shutil.rmtree(tmp_dir)
            #pass

        return outputs

if __name__ == "__main__":
    d = dict(a = dict(b = dict(c = dict(d = 10, e = 5))))
    print(f"{getkey_deep(d, "a.b.c.d")}")

    setkey_deep(d, "a.b.c.d", 2)
    print(f"{getkey_deep(d, "a.b.c.d")}")

    model = ForwardModel(
        "thrusters/h9.json",
        controls = [
            # "config.anode_mass_flow_rate",
            # "config.discharge_voltage",
            "config.magnetic_field_scale",
        ],
        dataset_dir = "inputs",
        verbose=True,
        num_workers=8,
    )

    field_scales = [[0.75], [1.0], [1.25]]
    states = [None, None, None] # TODO

    outputs = model(states, field_scales, dir="files")

    for o in outputs:
        print(o[0])

    #dataset = ThrusterDataset("files/outputs", scalars_in_tensor=True, fourier_features=True)