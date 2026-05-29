import os
import shutil

from pathlib import Path

from hall_diffusion import sample as sampling

class ReverseModel:
    def __init__(
            self,
            model: str | Path,
            config_file: str | Path,
            sample_dir: str | Path,
            replace_samples: bool = False,
            verbose : bool = False,
        ):
        self.model = Path(model)
        self.config_file = Path(config_file)
        self.sample_dir = Path(sample_dir)
        self.replace_samples = replace_samples
        self.verbose = verbose
        self.sample_dir_created = False

    def _make_sample_dir(self):
        if not self.sample_dir_created:
            if os.path.exists(self.sample_dir) and self.replace_samples:
                shutil.rmtree(self.sample_dir)
            os.makedirs(self.sample_dir, exist_ok=True)
        self.sample_dir_created = True

    def __call__(self, data, controls):
        # Runs the diffusion model
        # Outputs samples to sample_dir
        # Loads samples, returns for use by forward model
        sampling.infer(
            self.model,
            self.config_file,
            num_steps = 32,
            out_dir = self.sample_dir,
        )

if __name__ == "__main__":

    model = ReverseModel(
        model = "reverse_model/h9_batch2/checkpoint.pth.tar",
        config_file = "reverse_model/sample_h9.toml",
        sample_dir = "reverse_samples",
        replace_samples = False,
    )

    model._make_sample_dir()

    assert os.path.exists(model.sample_dir)

    model(None, None)



