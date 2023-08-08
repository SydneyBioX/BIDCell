import os 
import natsort 

# ,barcode_id,global_x,global_y,global_z,x,y,fov,gene,transcript_id

dataset = "dataset_merscope_melanoma2"
n_processes = 16
x_col = "global_x"
y_col = "global_y"
gene_col = "gene"
epoch = 1 
steps = 4000
config = "config_merscope_melanoma2.json"

# All in microns
target_pix_um = 1.0 # microns per pixel
base_pix_x = 0.107999132774 # 1/9.259333610534667969
base_pix_y = 0.107997631125 # 1/9.259462356567382812
base_ts_x = 1.0
base_ts_y = 1.0

# Scaling images
scale_pix_x = base_pix_x/target_pix_um
scale_pix_y = base_pix_y/target_pix_um
# Scaling transcript locations
scale_ts_x = base_ts_x/target_pix_um
scale_ts_y = base_ts_y/target_pix_um


os.chdir("bidcell/processing")

os.system(f"python nuclei_segmentation.py --dataset {dataset} --fp_dapi HumanMelanomaPatient2_images_mosaic_DAPI_z0.tif --scale_pix_x {scale_pix_x} --scale_pix_y {scale_pix_y} --max_height 20000 --max_width 20000")

os.system(f"python transcripts.py --dataset {dataset} --n_processes {n_processes} --fp_transcripts HumanMelanomaPatient2_detected_transcripts.csv --scale_ts_x {scale_ts_x} --scale_ts_y {scale_ts_y} --max_height 3500 --max_width 4000 --global_shift_x 12 --global_shift_y 10 --x_col {x_col} --y_col {y_col} --gene_col {gene_col} --shift_to_origin")

os.system(f"python transcript_patches.py --dataset {dataset} --patch_size 64")

os.system(f"python cell_gene_matrix.py --dataset {dataset} --fp_seg ../../data/{dataset}/nuclei.tif --output_dir cell_gene_matrices/nuclei --scale_factor_x {scale_pix_x} --scale_factor_y {scale_pix_y} --n_processes {n_processes} --x_col {x_col} --y_col {y_col} --gene_col {gene_col} --only_expr")

os.system(f"python preannotate.py --dataset {dataset} --config_file ../model/configs/{config} --fp_ref ../../data/sc_references/sc_melanoma.csv --n_processes {n_processes}")

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