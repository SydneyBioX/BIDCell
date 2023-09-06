from bidcell import BIDCellModel

BIDCellModel.get_example_data()

model = BIDCellModel("params_small_example.yaml")

model.run_pipeline()

# Alternatively, call individual functions

# model.preprocess()

# or call individual functions within preprocess

# # model.segment_nuclei()
# # model.generate_expression_maps()
# # model.generate_patches()
# # model.make_cell_gene_mat(is_cell=False)
# # model.preannotate()

# model.train()

# model.predict()
