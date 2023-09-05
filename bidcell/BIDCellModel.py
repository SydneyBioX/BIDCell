"""BIDCellModel class module"""
import importlib.resources
import os
from pathlib import Path
from shutil import copyfile, copytree

from .config import load_config
from .model.postprocess_predictions import postprocess_predictions
from .model.predict import fill_grid, predict
from .model.train import train
from .model.utils.utils import get_newest_id
from .processing.cell_gene_matrix import make_cell_gene_mat
from .processing.nuclei_segmentation import segment_nuclei
from .processing.nuclei_stitch_fov import stitch_nuclei
from .processing.preannotate import preannotate
from .processing.transcript_patches import generate_patches
from .processing.transcripts import generate_expression_maps


class BIDCellModel:
    """The BIDCellModel class, which provides an interface for preprocessing, training and predicting all the cell types for a datset."""

    def __init__(self, config_file: str) -> None:
        """Constructs a BIDCellModel instance using the user-supplied config file.\n
        The configuration is validated during construction.

        Args:
            config_file (str): Path to the YAML configuration file.
        """
        self.config = load_config(config_file)

    def run_pipeline(self):
        """Runs the entire BIDCell pipeline using the settings defined in the configuration.
        """
        print("### Preprocessing ###")
        print()
        self.preprocess()
        print()
        print("### Training ###")
        print()
        self.train()
        print()
        print("### Predict ###")
        print()
        self.predict()
        print()
        print("### Done ###")

    def preprocess(self) -> None:
        """Preprocess the dataset for training.
        """  
        if self.config.nuclei_fovs.stitch_nuclei_fovs:
            stitch_nuclei(self.config)
        segment_nuclei(self.config)
        generate_expression_maps(self.config)
        generate_patches(self.config)
        make_cell_gene_mat(self.config, is_cell=False)
        preannotate(self.config)

    def stitch_nuclei(self):
        """Stich separate FOV files into a single one (e.g. CosMx data).\n
        Runs inside preprocess by default, if nuclei_fovs.stitch_nuclei_fovs is True in the configuration file.
        """
        stitch_nuclei(self.config)

    def segment_nuclei(self):
        """Run the nucleus segmentation algorythm. Runs inside preprocess by default.
        """
        segment_nuclei(self.config)

    def generate_expression_maps(self):
        """Generate the expression maps. Runs inside preprocess by default.
        """        
        generate_expression_maps(self.config)

    def generate_patches(self):
        """Generate patches for training. Runs inside preprocess by default
        """
        generate_patches(self.config)

    def make_cell_gene_mat(self, is_cell: bool, timestamp: str = "last"):
        """Make a matrix containing counts for each cell.

        Args:
            is_cell (bool): If False, uses nuclei masks for creation, other wise it uses `timestamp` to chose a directory containing segmented cells outputted by BIDCell which can be in the data directory under `model_outputs`. Defaults to 'last', in which case it uses the folder with the most recent timestamp.
        """
        if timestamp == "last":
            timestamp = get_newest_id(
                os.path.join(self.config.files.data_dir, "model_outputs")
            )
        make_cell_gene_mat(self.config, is_cell, last=timestamp)

    def preannotate(self):
        preannotate(self.config)

    def train(self) -> None:
        train(self.config)

    def predict(self) -> None:
        predict(self.config)

        if self.config.experiment_dirs.dir_id == "last":
            last = get_newest_id(
                os.path.join(self.config.files.data_dir, "model_outputs")
            )
        else:
            last = self.config.experiment_dirs.dir_id
            # TODO: check if it is a valid experiement dir

        fill_grid(self.config, last)

        postprocess_predictions(self.config, last)

        make_cell_gene_mat(self.config, is_cell=True, last=last)

    @staticmethod
    def get_example_config(vendor: str) -> None:
        # TODO: Check if the vendor is valid
        params_path = (
            importlib.resources.files("bidcell") / "example_params" / f"{vendor}.yaml"
        )
        copyfile(params_path, Path().cwd() / f"{vendor}_example_config.yaml")

    @staticmethod
    def get_example_data(with_config: bool = True) -> None:
        root: Path = importlib.resources.files("bidcell")
        data_path = (
            root.parent / "data"
        )
        copytree(data_path, Path().cwd() / "example_data")
        if with_config:
            copyfile(
                root / "example_params" / "params_xenium_breast1.yaml",
                Path().cwd() / "example_data_params.yaml"
            )
