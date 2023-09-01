"""BIDCellModel class module"""
from typing import Optional
from multiprocessing import cpu_count
import os

import yaml

from .model.postprocess_predictions import postprocess_predictions
from .model.predict import predict
from .model.train import train
from .processing.cell_gene_matrix import make_cell_gene_mat
from .processing.nuclei_segmentation import segment_nuclei
from .processing.nuclei_stitch_fov import stitch_nuclei
from .processing.preannotate import preannotate
from .processing.transcript_patches import generate_patches
from .processing.transcripts import generate_expression_maps
from .model.utils.utils import get_newest_id
from .config import Config


class AttrDict(dict):
    """Dictionary subclass whose entries can be accessed by attributes
    (as well as normally).
    """

    def __init__(self, *args, **kwargs):
        def from_nested_dict(data):
            """Construct nested AttrDicts from nested dictionaries."""
            if not isinstance(data, dict):
                return data
            else:
                return AttrDict({key: from_nested_dict(data[key]) for key in data})

        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

        for key in self.keys():
            self[key] = from_nested_dict(self[key])


class BIDCellModel:
    """The BIDCellModel class, which provides an interface for preprocessing, training and predicting all the cell types for a datset."""

    def __init__(self, config_file: str, n_processes: Optional[int] = None) -> None:
        self.config = self.__parse_config(config_file)

        if n_processes is None:
            self.n_processes = cpu_count()
        else:
            self.n_processes = n_processes

    def preprocess(self) -> None:
        if self.config.nuclei_fovs.stitch_nuclei_fovs:
            self.config.files.fp_stitched = stitch_nuclei(self.config)
        self.config["fp_rdapi"] = segment_nuclei(self.config)
        self.config["fp_maps"] = generate_expression_maps(self.config)
        generate_patches(self.config)
        make_cell_gene_mat(self.config) # TODO: set config.cgm_params.only_expr (True for nuclei)
        preannotate(self.config)
        # TODO: Which information do the end users need from the process?

    def train(self) -> None:
        train(self.config)

    def predict(self) -> None:
        predict(self.config)
        # TODO: figure out the most recent experiment. get_lastest_id()
        if self.config.postprocess.dir_id == "last":
            self.config.postprocess.dir_id = get_newest_id()
        postprocess_predictions(self.config)
        # TODO: Figure out final cell_gene_matrix call
        # config.files.dir_output_matrices
        # config.files.fp_seg
        # config.cgm_params.only_expr (False for cells)

    def __parse_config(self, config_file_path: str) -> Config:
        if not os.path.exists(config_file_path):
            FileNotFoundError(
                f"Config file at {config_file_path} could not be found. Please check if the filepath is valid."
            )

        with open(config_file_path) as config_file:
            try:
                config = yaml.safe_load(config_file)
            except Exception:
                ValueError(
                    "The inputted YAML config was invalid, try looking at the example config."
                )

        if not isinstance(config, dict):
            ValueError(
                "The inputted YAML config was invalid, try looking at the example config."
            )

        # validate the configuration schema
        config = Config(**config)

        return config

    def set_config() -> None:
        # TODO: Document all config options and allow setting single or
        #       multiple options at a time.
        raise NotImplementedError()

    def __repr__(self) -> str:
        """Returns formatted BIDCellModel as a string with
        key configuration options listed as well as information about completed
        steps.

        Returns
        -------
        str"""
        return "Not implemented yet!"


if __name__ == "__main__":
    model = BIDCellModel("params/params_xenium_breast1.yaml")
    model.preprocess()
    model.train()
    model.predict()
