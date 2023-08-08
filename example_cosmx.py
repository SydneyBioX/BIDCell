import os 
import natsort 

# "fov","cell_ID","x_global_px","y_global_px","x_local_px","y_local_px","z","target","CellComp"

dataset = "dataset_cosmx_nsclc"
n_processes = 16
x_col = "x_global_px"
y_col = "y_global_px"
gene_col = "target"
epoch = 1 
steps = 4000
config = "config_cosmx_nsclc.json"

# All in microns
target_pix_um = 0.5 # microns per pixel
base_pix_x = 0.18
base_pix_y = 0.18
base_ts_x = 0.18
base_ts_y = 0.18

# Scaling images
scale_pix_x = base_pix_x/target_pix_um
scale_pix_y = base_pix_y/target_pix_um
# Scaling transcript locations
scale_ts_x = base_ts_x/target_pix_um
scale_ts_y = base_ts_y/target_pix_um


os.chdir("bidcell/processing")

os.system(f"python nuclei_stitch_fov.py --dataset {dataset} --dir_dapi Lung5_Rep1-RawMorphologyImages --pattern_z Z### --pattern_f F### --channel_first --channel_dapi -1 --n_fov 30 --min_fov 1 --n_fov_h 6 --n_fov_w 5 --row_major --z_level 1 --flip_ud")

os.system(f"python nuclei_segmentation.py --dataset {dataset} --fp_dapi dapi_preprocessed.tif --scale_pix_x {scale_pix_x} --scale_pix_y {scale_pix_y} --max_height 3648 --max_width 5472")

os.system(f"python transcripts.py --dataset {dataset} --n_processes {n_processes} --fp_transcripts Lung5_Rep1_tx_file.csv --scale_ts_x {scale_ts_x} --scale_ts_y {scale_ts_y} --max_height 4000 --max_width 5000 --global_shift_x 0 --global_shift_y 0 --x_col {x_col} --y_col {y_col} --gene_col {gene_col} --shift_to_origin")

os.system(f"python transcript_patches.py --dataset {dataset} --patch_size 64")

os.system(f"python cell_gene_matrix.py --dataset {dataset} --fp_seg ../../data/{dataset}/nuclei.tif --output_dir cell_gene_matrices/nuclei --scale_factor_x {scale_pix_x} --scale_factor_y {scale_pix_y} --n_processes {n_processes} --x_col {x_col} --y_col {y_col} --gene_col {gene_col} --only_expr")

os.system(f"python preannotate.py --dataset {dataset} --config_file ../model/configs/{config} --fp_ref ../../data/sc_references/sc_nsclc.csv --n_processes {n_processes}")

os.chdir("../model")

os.system(f"python train.py --config_file configs/{config} --total_steps {steps+100}")

os.system(f"python predict.py --config_file configs/{config} --test_epoch {epoch} --test_step {steps}")

os.system(f"python postprocess_predictions.py --epoch {epoch} --step {steps} --nucleus_fp ../../data/{dataset}/nuclei.tif --n_processes {n_processes}")

folders = next(os.walk('experiments'))[1]
folders = natsort.natsorted(folders)
folder_last = folders[-1]
exp_id = folder_last.replace('\\','/')

os.chdir("../processing")

os.system(f"python cell_gene_matrix.py --dataset {dataset} --fp_seg ../model/experiments/{exp_id}/test_output/epoch_{epoch}_step_{steps}_connected.tif --output_dir cell_gene_matrices/{exp_id} --scale_factor_x {scale_pix_x} --scale_factor_y {scale_pix_y} --n_processes {n_processes} --x_col {x_col} --y_col {y_col} --gene_col {gene_col}")