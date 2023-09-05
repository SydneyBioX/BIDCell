from bidcell import BIDCellModel


model = BIDCellModel("params/params_xenium_breast1.yaml")
print("### Preprocessing ###")
model.preprocess()
print("### Training ###")
model.train()
print("### Training ###")
model.predict()
