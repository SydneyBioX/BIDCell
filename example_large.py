from bidcell import BIDCellModel
import argparse

def main(args):

    BIDCellModel.get_example_config(args.vendor)

    model = BIDCellModel(f"{args.vendor}_example_config.yaml")

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vendor', default='xenium', type=str, help="name of vendor")

    args = parser.parse_args()
    main(args)
