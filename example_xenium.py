import os 

dataset = "dataset_xenium_breast1"
n_processes = 16
x_col = "x_location"
y_col = "y_location"
gene_col = "feature_name"
epoch = 1 
steps = 4000
config = "config_xenium_breast1.json"
scale_x = 0.2125
scale_y = 0.2125

os.system(f"cd bidcell/processing")

os.system(f"python nuclei_segmentation.py --dataset {dataset} --fp_dapi morphology_mip.ome.tif --scale_x {scale_x} --scale_y {scale_y} --max_height 40000 --max_width 40000")

os.system(f"python cell_gene_matrix.py --dataset {dataset} --fp_seg ../../data/{dataset}/nuclei.tif --output_dir cell_gene_matrices/nuclei --scale_factor_x {scale_x} --scale_factor_y {scale_y} --n_processes {n_processes} --x_col {x_col} --y_col {y_col} --gene_col {gene_col} --only_expr")

os.system(f"python preannotate.py --dataset {dataset} --config_file ../model/configs/{config} --fp_ref ../../data/sc_references/sc_breast.csv --n_processes {n_processes}")

os.system(f"python transcripts.py --dataset {dataset} --n_processes {n_processes} --fp_transcripts transcripts.csv.gz --scale_x 1.0 --scale_y 1.0 --max_height 3500 --max_width 4000 --global_shift_x 0 --global_shift_y 0 --x_col {x_col} --y_col {y_col} --gene_col {gene_col}")

os.system(f"python transcript_patches.py --dataset {dataset} --patch_size 48")

os.system(f"cd ../model")

os.system(f"python train.py --config_file configs/{config} --total_steps {steps+100}")

os.system(f"python predict.py --config_file configs/{config} --test_epoch {epoch} --test_step {steps}")

os.system(f"python postprocess_predictions.py --epoch {epoch} --step {steps} --nucleus_fp ../../data/{dataset}/nuclei.tif --n_processes {n_processes}")

os.system(f"cd ../processing")

os.system(f"python cell_gene_matrix.py --dataset {dataset} --fp_seg ../model/experiments/2023_August_08_17_50_50/test_output/epoch_{epoch}_step_{steps}_connected.tif --output_dir cell_gene_matrices/2023_August_08_17_50_50 --scale_factor_x {scale_x} --scale_factor_y {scale_y} --n_processes {n_processes} --x_col {x_col} --y_col {y_col} --gene_col {gene_col}")