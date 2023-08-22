"""BIDCellModel class module"""
from typing import Optional, Dict
from .processing.nuclei_stitch_fov import stitch_nuclei


available_vendors: Dict[str, str] = {
    "cosmx": "CosMx",
    "merscope": "Merscope",
    "visium": "Visium"
}


class BIDCellModel:

    def __init__(self, vendor: str, n_processes: Optional[int] = None) -> None:
        self.n_processes = n_processes
        if vendor not in available_vendors.keys():
            # TODO: list vendors
            raise ValueError("Invalid vendor: must be one of ...")

        self.vendor = available_vendors[vendor]
        self.config = BIDCellModel._get_defaults(self.vendor)

    def preprocess(self) -> None:
        if self.vendor == "CosMx":
            stitch_nuclei(self.config)

    def train(self) -> None:
        pass

    def predict(self) -> None:
        pass

    def _get_defaults(vendor: str) -> dict:
        # TODO: figure out defaults situation
        if True:
            raise NotImplementedError()
        return dict()

    def set_config() -> None:
        # TODO: Document all config options and allow setting single or
        #       multiple options at a time.
        raise NotImplementedError()




if __name__ == "__main__":

    model = BIDCellModel()
    model.preprocess()
    model.train()
    model.predict()
