"""BIDCellModel class module"""


available_vendors: Dict[str, str] = {
    "cosmx": "CosMx",
    "merscope": "Merscope",
    "visium": "Visium"
}


class BIDCellModel:

    def __init__(self, n_processes: Optional[int] = None) -> None:
        self.n_processes = n_processes

    def preprocess() -> None:
        pass

    def train() -> None:
        pass

    def predict() -> None:
        pass


if __name__ == "__main__":

    model = BIDCellModel()
    model.preprocess()
    model.train()
    model.predict()
