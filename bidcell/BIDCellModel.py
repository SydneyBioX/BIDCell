"""BIDCellModel class module"""
from typing import Optional, Literal
from multiprocessing import cpu_count
import os

import yaml
from pydantic import BaseModel

from bidcell.model.postprocess_predictions import postprocess_predictions
from bidcell.model.predict import predict
from bidcell.model.train import train
from bidcell.processing.cell_gene_matrix import make_cell_gene_mat
from bidcell.processing.nuclei_segmentation import segment_nuclei
from bidcell.processing.nuclei_stitch_fov import stitch_nuclei
from bidcell.processing.preannotate import preannotate
from bidcell.processing.transcript_patches import generate_patches
from bidcell.processing.transcripts import generate_expression_maps
from bidcell.model.utils.utils import get_newest_id


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


class FileParams(BaseModel):
    data_dir: str
    dataset: str
    fp_dapi: str
    fp_transcripts: str
    fp_ref: str
    fp_pos_markers: str
    fp_neg_markers: str

    # Internal defaults
    # file of affine transformation - needed if cropping to align DAPI to transcripts
    fp_affine: str = "affine.csv"
    # file name of nuclei tif file
    fp_nuclei: str = "nuclei.tif"
    # file name of resized DAPI image
    fp_rdapi: str = "dapi_resized.tif"
    # txt file containing names of transcripts to filter out
    fp_transcripts_to_filter: str = "transcripts_to_filter.txt"
    # directory containing processed gene expression maps
    dir_out_maps: str = "expr_maps"
    # filtered and xy-scaled transcripts data
    fp_transcripts_processed: str = "transcripts_processed.csv"
    # txt file containing list of gene names
    fp_gene_names: str = "all_gene_names.txt"
    # directory prefix of transcript patches
    dir_patches: str = "expr_maps_input_patches_"
    # directory for nuclei expression matrices
    expr_dir: str = "cell_gene_matrices/nuclei"
    # file name of nuclei expression matrices
    fp_expr: str = "cell_expr.csv"
    # file path of nuclei annotations
    fp_nuclei_anno: str = "nuclei_cell_type.h5"


class NucleiFovParams(BaseModel):
    stitch_nuclei_fovs: bool
    dir_dapi: str | None = None
    ext_dapi: str = "tif"
    pattern_z: str = "Z###"
    pattern_f: str = "F###"
    channel_first: bool = False
    channel_dapi: int = -1
    fp_dapi_stitched: str = "dapi_preprocessed.tif"
    n_fov: int | None = None
    min_fov: int | None = None
    n_fov_h: int | None = None
    n_fov_w: int | None = None
    start_corner: Literal["ul", "ur", "bl", "br"] = "ul"
    row_major: bool = False
    z_level: int = 1
    mip: bool = False
    flip_ud: bool = False
    # TODO: Add validator for class which checks if dir_dapi is present if stitch_nuclei_fovs is True


class NucleiParams(BaseModel):
    # divide into sections if too large - maximum height to process in original resolution
    max_height: int = 24000
    # divide into sections if too large - maximum width to process in original resolution
    max_width: int = 32000
    # crop nuclei to size of transcript maps
    crop_nuclei_to_ts: bool = False
    # use CPU for Cellpose if no GPU available
    use_cpu: bool = False
    # estimated diameter of nuclei for Cellpose
    diameter: int | None = None


class TranscriptParams:
    min_qv: int = 20
    # divide into sections if too large - height of patches
    max_height: int = 3500
    # divide into sections if too large - width of patches
    max_width: int = 4000
    shift_to_origin: False
    x_col: str = "x_location"
    y_col: str = "y_location"
    gene_col: str = "feature_name"
    fp_selected_genes: str | None = None
    counts_col: str | None = None


class AffineParams(BaseModel):
    target_pix_um: float = 1.0
    base_pix_x: float
    base_pix_y: float
    base_ts_x: float
    base_ts_y: float
    global_shift_x: int = 0
    global_shift_y: int = 0


class CellGeneMatParams(BaseModel):
    fp_seg: str
    output_dir: str
    # max h+w for resized segmentation to extract expressions from
    max_sum_hw: int = 50000


class ModelParams(BaseModel):
    name: str = "custom"  # TODO: Validate this field
    patch_size: int
    elongated: list[str]


class TrainingParams(BaseModel):
    total_epochs: int = 1
    total_steps: int = 4000
    # learning rate of DL model
    learning_rate: float = 0.00001
    # adam optimiser beta1
    beta1: float = 0.9
    # adam optimiser beta2
    beta2: float = 0.999
    # adam optimiser weight decay
    weight_decay: float = 0.0001
    # optimiser
    optimizer: Literal["adam", "rmsprop"] = "adam"
    ne_weight: float = 1.0
    os_weight: float = 1.0
    cc_weight: float = 1.0
    ov_weight: float = 1.0
    pos_weight: float = 1.0
    neg_weight: float = 1.0
    # number of training steps per model save
    model_freq: int = 1000
    # number of training steps per sample save
    sample_freq: int = 100


class TestingParams(BaseModel):
    test_epoch: int = 1
    test_step: int = 4000


class PostprocessParams(BaseModel):
    # size of patches to perform morphological processing
    patch_size_mp: int = 1024
    dir_id: str | None = None


class ExperimentDirs(BaseModel):
    # directory names for each experiment under bidcell/model/experiments
    load_dir: str = "last"
    model_dir: str = "models"
    test_output_dir: str = "test_output"
    samples_dir: str = "samples"


class Config(BaseModel):
    files: FileParams
    nuclei_fovs: NucleiFovParams
    nuclei: NucleiParams
    transcripts: TranscriptParams
    affine: AffineParams
    model_params: ModelParams
    training_params: TrainingParams
    testing_params: TestingParams
    postprocess: PostprocessParams
    experiment_dirs: ExperimentDirs


class BIDCellModel:
    """The BIDCellModel class, which provides an interface for preprocessing, training and predicting all the cell types for a datset."""

    def __init__(self, config_file: str, n_processes: Optional[int] = None) -> None:
        self.config = self.__parse_config(config_file)

        if n_processes is None:
            self.n_processes = cpu_count()
        else:
            self.n_processes = n_processes

    def preprocess(self) -> None:
        if self.vendor == "CosMx":
            self.config["fp_stitched"] = stitch_nuclei(self.config)
        self.config["fp_rdapi"] = segment_nuclei(self.config)
        self.config["fp_maps"] = generate_expression_maps(self.config)
        generate_patches(self.config)
        make_cell_gene_mat(self.config)
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

    def __parse_config(self, config_file_path: str) -> Config:
        if not os.path.exists():
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
