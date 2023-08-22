"""BIDCellModel class module"""
import os
from typing import Dict, Optional

from model.postprocess_predictions import postprocess_predicitons
from model.predict import predict
from model.train import train
from processing.cell_gene_matrix import make_cell_gene_mat
from processing.nuclei_segmentation import segment_nuclei
from processing.nuclei_stitch_fov import stitch_nuclei
from processing.preannotate import preannotate
from processing.transcript_patches import generate_patches
from processing.transcripts import generate_expression_maps

available_vendors: Dict[str, str] = {
    "cosmx": "CosMx",
    "merscope": "Merscope",
    "visium": "Visium",
}


class BIDCellModel:
    def __init__(
        self,
        vendor: str,  # TODO: infer vendor from data dir?
        raw_data_dir: str,
        outputs_dir: str,
        dapi_dir: Optional[str] = None,
        n_processes: Optional[int] = None,
    ) -> None:
        self.n_processes = n_processes

        if vendor not in available_vendors.keys():
            # TODO: list vendors
            raise ValueError("Invalid vendor: must be one of ...")  # TODO

        self.vendor = available_vendors[vendor]

        if not os.path.exists(raw_data_dir):
            raise ValueError(
                f"Invalid data directory selected: {raw_data_dir} is not a valid path."
            )
        self.raw_data_dir = raw_data_dir

        # check if output already exists
        if os.path.exists(outputs_dir):
            # TODO: Add overwrite param?
            raise ValueError("Output path already exists!")

        self.outputs_dir = outputs_dir

        if dapi_dir and not os.path.exists(dapi_dir):
            raise ValueError(
                f"Invalid DAPI directory selected: {dapi_dir} is not a valid path."
            )
        self.dapi_dir = dapi_dir

        # TODO: perhaps config should be a "hidden" var
        self.config = BIDCellModel._get_defaults(self.vendor)

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
        postprocess_predicitons(self.config)

    def _get_defaults(vendor: str) -> dict:
        # TODO: figure out defaults situation
        raise NotImplementedError()
        return dict()

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
    model = BIDCellModel()
    model.preprocess()
    model.train()
    model.predict()
