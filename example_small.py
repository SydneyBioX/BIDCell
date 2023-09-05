from bidcell import BIDCellModel

model = BIDCellModel("params_small_example.yaml")

model.get_example_data()

model.run_pipeline()

# model.preprocess()

# # Alternatively, call individual functions

# # model.segment_nuclei()
# # model.generate_expression_maps()
# # model.generate_patches()
# # model.make_cell_gene_mat(is_cell=False)
# # model.preannotate()

# model.train()

# model.predict()

