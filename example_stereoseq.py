import os 
import natsort 
import argparse
from bidcell.model.utils.utils import get_newest_id

def main(args):

    # Scaling images
    scale_pix_x = args.base_pix_x/args.target_pix_um
    scale_pix_y = args.base_pix_y/args.target_pix_um
    # Scaling transcript locations
    scale_ts_x = args.base_ts_x/args.target_pix_um
    scale_ts_y = args.base_ts_y/args.target_pix_um

    os.chdir("bidcell/processing")

    # An example where the images are cropped to the area of the detected gene expressions

    os.system(f"python transcripts.py --dataset {args.dataset} --n_processes {args.n_processes} --fp_transcripts {args.fp_transcripts} --scale_ts_x {scale_ts_x} --scale_ts_y {scale_ts_y} --max_height 4000 --max_width 5000 --global_shift_x {args.global_shift_x} --global_shift_y {args.global_shift_y} --x_col {args.x_col} --y_col {args.y_col} --gene_col {args.gene_col} --shift_to_origin --fp_selected_genes {args.fp_selected_genes} --counts {args.counts_col} --fp_nuclei none") 

    os.system(f"python nuclei_segmentation.py --dataset {args.dataset} --fp_dapi {args.fp_dapi} --scale_pix_x {scale_pix_x} --scale_pix_y {scale_pix_y} --max_height 4000 --max_width 4000 --crop_to_ts --diameter 30")

    os.system(f"python transcript_patches.py --dataset {args.dataset} --patch_size {args.patch_size}")

    os.system(f"python cell_gene_matrix.py --dataset {args.dataset} --fp_seg ../../data/{args.dataset}/nuclei.tif --output_dir cell_gene_matrices/nuclei --scale_factor_x {scale_pix_x} --scale_factor_y {scale_pix_y} --n_processes {args.n_processes} --x_col {args.x_col} --y_col {args.y_col} --gene_col {args.gene_col} --only_expr")

    os.system(f"python preannotate.py --dataset {args.dataset} --config_file ../model/configs/{args.fp_config} --fp_ref ../../data/sc_references/sc_mousebrain.csv --n_processes {args.n_processes}")

    os.chdir("../model")

    os.system(f"python train.py --config_file configs/{args.fp_config} --total_steps {args.steps+100}")

    os.system(f"python predict.py --config_file configs/{args.fp_config} --test_epoch {args.epoch} --test_step {args.steps}")

    os.system(f"python postprocess_predictions.py --epoch {args.epoch} --step {args.steps} --nucleus_fp ../../data/{args.dataset}/nuclei.tif --n_processes {args.n_processes}")

    exp_id = get_newest_id()

    os.chdir("../processing")

    os.system(f"python cell_gene_matrix.py --dataset {args.dataset} --fp_seg ../model/experiments/{exp_id}/test_output/epoch_{args.epoch}_step_{args.steps}_connected.tif --output_dir cell_gene_matrices/{args.exp_id} --scale_factor_x {scale_pix_x} --scale_factor_y {scale_pix_y} --n_processes {args.n_processes} --x_col {args.x_col} --y_col {args.y_col} --gene_col {args.gene_col}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='dataset_stereoseq_mousebrain', type=str, help="name of dataset")

    parser.add_argument('--fp_dapi', default='Mouse_brain_Adult.tif', type=str, help="name of DAPI image")

    parser.add_argument('--fp_transcripts', default='Mouse_brain_Adult_GEM_bin1.tsv.gz', type=str, help="name of transcripts file")

    parser.add_argument('--fp_selected_genes', default='selected_genes.txt', type=str, help="name of transcripts file")

    # geneID, x, y, MIDCounts
    parser.add_argument('--x_col', default='y', type=str)
    parser.add_argument('--y_col', default='x', type=str)
    parser.add_argument('--gene_col', default='geneID', type=str)
    parser.add_argument('--counts_col', default='MIDCounts', type=str)
    
    parser.add_argument('--target_pix_um', default=1.0, type=float, help="microns per pixel to perform segmentation")
    parser.add_argument('--base_pix_x', default=1.0, type=float, help="convert to microns along width by multiplying the original pixels by base_pix_x microns per pixel")
    parser.add_argument('--base_pix_y', default=1.0, type=float, help="convert to microns along height by multiplying the original pixels by base_pix_y microns per pixel")
    parser.add_argument('--base_ts_x', default=1.0, type=float, help="convert between transcript locations and target pixels along width")
    parser.add_argument('--base_ts_y', default=1.0, type=float, help="convert between transcript locations and target pixels along height")
    parser.add_argument('--global_shift_x', default=0, type=int, help="additional adjustment to align transcripts to DAPI in target pixels along image width")
    parser.add_argument('--global_shift_y', default=0, type=int, help="additional adjustment to align transcripts to DAPI in target pixels along image height")

    parser.add_argument('--fp_config', default='"config_stereoseq_mousebrain.json', type=str)

    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--steps', default=4000, type=int, help="number of training steps")
    parser.add_argument('--patch_size', default=48, type=int, help="size of input patches to segmentation model")

    parser.add_argument('--n_processes', default=16, type=int, help="number of CPUs")

    args = parser.parse_args()
    main(args)
