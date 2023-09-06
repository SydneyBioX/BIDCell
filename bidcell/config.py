import os
from pathlib import Path
from typing import Literal, Annotated

import yaml
from pydantic import BaseModel, computed_field, model_validator, ConfigDict
from pydantic.functional_validators import AfterValidator


def validate_path(v: str | None) -> str:
    if v is None:
        return v

    path = Path(v)

    assert (
        path.exists()
    ), f"Invalid path {v}: Ensure you have the correct path in your config file."

    return str(path.resolve())


PathString = Annotated[str, AfterValidator(validate_path)]


class FileParams(BaseModel):
    data_dir: PathString
    fp_dapi: PathString | None = None
    fp_transcripts: PathString
    fp_ref: PathString
    fp_pos_markers: PathString
    fp_neg_markers: PathString

    # Internal defaults
    # file of affine transformation - needed if cropping to align DAPI to transcripts
    fp_affine: str = "affine.csv"
    # file name of nuclei tif file
    fp_nuclei: str = "nuclei.tif"
    # file name of resized DAPI image
    fp_rdapi: str = "dapi_resized.tif"
    # directory containing processed gene expression maps
    dir_out_maps: str = "expr_maps"
    # filtered and xy-scaled transcripts data
    fp_transcripts_processed: str = "transcripts_processed.csv"
    # txt file containing list of gene names
    fp_gene_names: str = "all_gene_names.txt"
    # directory prefix of transcript patches
    dir_patches: str = "expr_maps_input_patches_"
    # directory for cell-gene expression matrices
    dir_cgm: str = "cell_gene_matrices"
    # file name of nuclei expression matrices
    fp_expr: str = "expr_mat.csv"
    # file name of nuclei annotations
    fp_nuclei_anno: str = "nuclei_cell_type.h5"
    # file name of text file containing selected gene names, e.g. selected_genes.txt
    fp_selected_genes: str | None = None

    # Internal
    # fp_stitched: str | None = None


class NucleiFovParams(BaseModel):
    stitch_nuclei_fovs: bool
    dir_dapi: str | None = None
    ext_dapi: str = "tif"
    pattern_z: str = "Z###"
    pattern_f: str = "F###"
    channel_first: bool = False
    channel_dapi: int = -1
    n_fov: int | None = None
    min_fov: int | None = None
    n_fov_h: int | None = None
    n_fov_w: int | None = None
    start_corner: Literal["ul", "ur", "bl", "br"] = "ul"
    row_major: bool = False
    z_level: int = 1
    mip: bool = False
    flip_ud: bool = False

    @model_validator(mode="after")
    def check_dapi(self):
        if not self.stitch_nuclei_fovs:
            return self

        if self.dir_dapi is None:
            raise ValueError(
                "dir_dapi must be specified if stitch_nuclei_fovs is True."
            )

        p = Path(self.dir_dapi)

        if not p.exists():
            raise ValueError(
                "Invalid value for dir_dapi ({self.dir_dapi}): Check the config file and ensure the correct directory is specified."
            )

        if not p.is_dir():
            raise ValueError(
                "dir_dapi is not a directory: dir_dapi must point to a directory containing the FOVs to be stitched together."
            )
        return self


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


class TranscriptParams(BaseModel):
    min_qv: int = 20
    # divide into sections if too large - height of patches
    max_height: int = 3500
    # divide into sections if too large - width of patches
    max_width: int = 4000
    shift_to_origin: bool = False
    x_col: str = "x_location"
    y_col: str = "y_location"
    gene_col: str = "feature_name"
    counts_col: str | None = None
    transcripts_to_filter: list[str]


class AffineParams(BaseModel):
    target_pix_um: float = 1.0
    base_pix_x: float
    base_pix_y: float
    base_ts_x: float
    base_ts_y: float
    global_shift_x: int = 0
    global_shift_y: int = 0

    # Scaling images
    @computed_field
    @property
    def scale_pix_x(self) -> float:
        return self.base_pix_x / self.target_pix_um

    @computed_field
    @property
    def scale_pix_y(self) -> float:
        return self.base_pix_y / self.target_pix_um

    # Scaling transcript locations
    @computed_field
    @property
    def scale_ts_x(self) -> float:
        return self.base_ts_x / self.target_pix_um

    @computed_field
    @property
    def scale_ts_y(self) -> float:
        return self.base_ts_y / self.target_pix_um


class CellGeneMatParams(BaseModel):
    # max h+w for resized segmentation to extract expressions from
    max_sum_hw: int = 30000


class ModelParams(BaseModel):
    name: str = "custom"  # TODO: Validate this field
    patch_size: int
    elongated: list[str]


class TrainingParams(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
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


class ExperimentDirs(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    # directory names for each experiment
    dir_id: str = "last"
    model_dir: str = "models"
    test_output_dir: str = "test_output"
    samples_dir: str = "samples"


class Config(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    files: FileParams
    nuclei_fovs: NucleiFovParams
    nuclei: NucleiParams
    transcripts: TranscriptParams
    affine: AffineParams
    model_params: ModelParams
    training_params: TrainingParams
    testing_params: TestingParams
    cpus: int
    postprocess: PostprocessParams = PostprocessParams()
    experiment_dirs: ExperimentDirs = ExperimentDirs()
    cgm_params: CellGeneMatParams = CellGeneMatParams()


def load_config(path: str) -> Config:
    if not os.path.exists(path):
        FileNotFoundError(
            f"Config file at {path} could not be found. Please check if the filepath is valid."
        )

    with open(path) as config_file:
        try:
            config = yaml.safe_load(config_file)
        except Exception:
            raise ValueError(
                "The inputted YAML config was invalid, try looking at the example config."
            )

    if not isinstance(config, dict):
        raise ValueError(
            "The inputted YAML config was invalid, try looking at the example config."
        )

    # validate the configuration schema
    config = Config(**config)
    return config
