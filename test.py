from bidcell import BIDCellModel


model = BIDCellModel("params/params_xenium_breast1.yaml")
print()
print("### Preprocessing ###")
print()
model.preprocess()
print()
print("### Training ###")
print()
model.train()
print()
print("### Predict ###")
print()
model.predict()
