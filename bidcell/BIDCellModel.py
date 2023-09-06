"""BIDCellModel class module"""
import importlib.resources
import os
from pathlib import Path
from shutil import copyfile, copytree
from typing import Literal

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

        Parameters
        ----------
        config_file : str
            Path to the YAML configuration file.
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
        if self.config.nuclei.crop_nuclei_to_ts:
            generate_expression_maps(self.config)
            segment_nuclei(self.config)
        else:
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
        """Generate patches for training. Runs inside preprocess by default.
        """
        generate_patches(self.config)

    def make_cell_gene_mat(self, is_cell: bool, timestamp: str = "last"):
        """Make a matrix containing counts for each cell. Runs inside preprocess and predict by default.

        Parameters
        ----------
        is_cell : bool
            If False, uses nuclei masks for creation, other wise it uses `timestamp` to chose a directory containing segmented cells outputted by BIDCell.
        timestamp : str, optional
            The timestamp corrisponding to the name of a directory in the data directory under `model_outputs`, by default "last", in which case it uses the folder with the most recent timestamp.
        """
        if is_cell and timestamp == "last":
            timestamp = get_newest_id(
                os.path.join(self.config.files.data_dir, "model_outputs")
            )
        elif is_cell:
            self.__check_valid_timestamp(timestamp)
        make_cell_gene_mat(self.config, is_cell, timestamp=timestamp)

    def preannotate(self):
        """Preannotate the cells. Runs inside preprocess by default.
        """
        preannotate(self.config)

    def train(self) -> None:
        """Train the model.
        """
        train(self.config)

    def predict(self) -> None:
        """Segment and annotate the cells.
        """
        predict(self.config)

        if self.config.experiment_dirs.dir_id == "last":
            timestamp = get_newest_id(
                os.path.join(self.config.files.data_dir, "model_outputs")
            )
        else:
            timestamp = self.config.experiment_dirs.dir_id
            self.__check_valid_timestamp(timestamp)

        fill_grid(self.config, timestamp)

        postprocess_predictions(self.config, timestamp)

        make_cell_gene_mat(self.config, is_cell=True, timestamp=timestamp)

    @staticmethod
    def get_example_config(vendor: Literal["cosmx", "merscope", "stereoseq", "xenium"]) -> None:
        """Gets an example configuration for a given vendor and places it in the working directory.

        Parameters
        ----------
        vendor : Literal["cosmx", "merscope", "stereoseq", "xenium"]
            The vendor of the equiptment used to produce the dataset.
        """
        vendors = ["cosmx", "merscope", "stereoseq", "xenium"]
        if not any([vendor.lower() == x for x in vendors]):
            raise ValueError(f"Unknown vendor `{vendor}`\n\tChose one of {*vendors,}")
        params_path = (
            importlib.resources.files("bidcell") / "example_params" / f"{vendor}.yaml"
        )
        copyfile(params_path, Path().cwd() / f"{vendor}_example_config.yaml")

    @staticmethod
    def get_example_data(with_config: bool = True) -> None:
        """Gets the small example data included in the package and places it in the current working directory.

        Parameters
        ----------
        with_config : bool, optional
            Whether to get the configuration for the example data, by default True
        """
        root: Path = importlib.resources.files("bidcell")
        data_path = (
            root.parent / "data"
        )
        copytree(data_path, Path().cwd() / "example_data")
        if with_config:
            copyfile(
                root / "example_params" / "small_example.yaml",
                Path().cwd() / "params_small_example.yaml"
            )

    def __check_valid_timestamp(self, timestamp: str) -> None:
        outputs_path = Path(self.config.files.data_dir / "model_outputs")
        outputs = list(outputs_path.iterdir())
        if len(outputs) == 0:
            raise ValueError(
                f"There are no outputs yet under {str(outputs_path)}. Run BIDCell at least once with this dataset to get some."
            )
        if not any(
            [timestamp == x for x in outputs if x.is_dir()]
        ):
            valid_dirs = "\n".join(["\t" + str(x) for x in outputs])
            raise ValueError(
                f"{timestamp} is not a valid model output directory (set in configuration YAML under `experiment_dirs.dir_id`). Choose one of the following:\n{valid_dirs}"
            )
