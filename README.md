# MRIGait_EmpkinS
Official repository for the Plos Computational Biology submission: Shape-Based Personalization of Musculoskeletal Models Using Smartphone Images: Validation with Gait and MRI Data

Attention:
**We cannot provide all data, such as the raw MRI data, smartphone pictures and initial avatars, due to privacy concerns. Therefore, running the "mesh_pipeline.py" and "reshape_mri.py" scripts will not work out of the box. However, we provide the SMPL fits and segmentation masks in the "results" folder in the data repository. The remaining scripts can be run with these results to reproduce the results in the paper.**

## Getting started

1. Clone this repository
2. Get our fork of SMPL-Fitting from [here](github), follow the instructions to install it.
3. Get SKEL from [here](github) and follow the instructions to install it - the dependencies are already included in our environment file.
4. Get HIT from [here](github) and follow the instructions to install it. You will need to create a new conda environment, because the dependencies are incompatible with the the main environment file, use `environment_hit.yml` to create the environment. 
5. Install environments. The main environment file is `environment_full.yml`. If you have trouble installing, try `environment_full_no_builds.yml`. We are aware that there are version conflicts regarding `numpy`, therefore, after installing the environment, you might need to downgrade `numpy` to version 1.23.5.

### Get our data
1. Download the data and the results from [here](todo) and extract it in the root of this repository. You need at least 40 GB of free disk space.
2. Extract the data. There need be two new folders at the root of the repository:
```
data/...
results/...
```
We do not provide raw MRI data, but the SMPL fits together with the segmentation masks are available in the "results" folder.

Structure of the gait data in the results folder:
```
results/P01/
    gait/
        addB-ignore_phys/ <-- S01 folder on addBiomechanics
            IK/
            ID/
            Models/
        addB-with_phys/
            S01_HIT/ <-- from addBiomechanics
            S01_SGEN/ <-- from addBiomechanics
            generic/ <-- S01_with_phys folder on addBiomechanics
    ...
```
4. To save some space, we removed the .stl files from the results/P**/MRI/final_masks folder. You can use e.g. 3D Slicer to create the .stl files from the .nrrd files in the same folder. The .stl files are not needed for creating the MRI-based models, but are needed for the mesh pipeline and the MRI reshaping scripts. If you want to run these scripts, you can create the .stl files with the following command:

### Reproduce experiments
1. Run the scripts below to reproduce the results:
```
# Cannot run: python scripts/bodyscan/mesh_pipeline.py because we do not provide the raw (identifying) data, so skip this step.
python scripts/bodyscan/hit_predictions.py
python scripts/bodyscan/all_bisp_estimations.py

python scripts/smpl_generic/create_generic_models.py
python scripts/smpl_generic/generic_models_scale_and_id.py

python scripts/gait/create_all_osim_models.py
python scripts/gait/run_id_all.py

# In reshape_mri.py, an SMPL-fit is created first. However, we do not provide the complete hull annotations (identifying data), so this step is skipped. (line 62: if False:)
python -m src.mri.reshape_mri --scan_result_path results/P01/bodyhull/smpl_fit/scan/ --mri_hull_path results/P01/MRI/final_masks --scan_name scaneca --participant P01 --hull_name_mri all --output_path results/P01/MRI/bodyhull/
python -m src.mri.reshape_mri --scan_result_path results/P02/bodyhull/smpl_fit/scan/ --mri_hull_path results/P02/MRI/final_masks --scan_name scaneca --participant P02 --hull_name_mri all --output_path results/P02/MRI/bodyhull/
python -m src.mri.reshape_mri --scan_result_path results/P03/bodyhull/smpl_fit/scan/ --mri_hull_path results/P03/MRI/final_masks --scan_name scaneca --participant P03 --hull_name_mri all --output_path results/P03/MRI/bodyhull/
python -m src.mri.reshape_mri --scan_result_path results/P04/bodyhull/smpl_fit/scan/ --mri_hull_path results/P04/MRI/final_masks --scan_name scaneca --participant P04 --hull_name_mri all --output_path results/P04/MRI/bodyhull/
python -m src.mri.reshape_mri --scan_result_path results/P05/bodyhull/smpl_fit/scan/ --mri_hull_path results/P05/MRI/final_masks --scan_name scaneca --participant P05 --hull_name_mri all --output_path results/P05/MRI/bodyhull/
python -m src.mri.reshape_mri --scan_result_path results/P06/bodyhull/smpl_fit/scan/ --mri_hull_path results/P06/MRI/final_masks --scan_name scaneca --participant P06 --hull_name_mri all --output_path results/P06/MRI/bodyhull/
python scripts/mri/mri_id.py
```
2. Reproduce the results for the paper:
Run `python scripts/mrigait_benchmark/run_mrigait_benchmark.py` to reproduce the MRIgait benchmark results. Then, run all cells in the Jupyter notebook `notebooks/paper_plots.ipynb` to reproduce the plots/other tables in the paper.

## Using this repository in production

We are developing a web-app, which means that updated pipelines are being worked on. Stay tuned!