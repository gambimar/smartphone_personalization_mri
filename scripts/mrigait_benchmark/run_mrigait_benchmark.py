import pandas as pd
import subprocess
from src.mrigait_benchmark.mrigait_benchmark import main as mrigait_benchmark_main
import os

# Find the .git root directory
def find_git_root():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while current_dir != os.path.dirname(current_dir):
        if os.path.exists(os.path.join(current_dir, '.git')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    raise FileNotFoundError("No .git directory found in the path hierarchy.")

git_root = find_git_root()

participant_info = pd.read_csv(git_root+"/data/participant_info.csv")#.iloc[[0,2,5]] # female subset
participants = participant_info['participant'].tolist()
mass_list = participant_info['mass_treadmill'].tolist()
gender_list = participant_info['sex'].tolist()
# Replace "M" and "F" with "male" and "female"
gender_list = ["male" if g == "M" else "female" for g in gender_list]

scaled_res, addB_res, sipp_res, sipp_generic_res, addB_SIPP_model_res, addB_SGEN_model_res = [], [], [], [], [], []

for p, mass, gender in zip(participants, mass_list, gender_list):
    # Run the MRIGait benchmark for each participant
    gt_model = f"{git_root}/results/{p}/MRI/osim_models/mri.osim"
    scaled_generic_model = f"{git_root}/results/{p}/gait/addB-ignore_phys/Models/match_markers_but_ignore_physics.osim"
    addB_model = f"{git_root}/results/{p}/gait/addB-with_phys/generic/Models/match_markers_and_physics.osim"
    sipp_model = f"{git_root}/results/{p}/bodyhull/bsipw_shoes/HIT_body_loose_w_skel.osim"
    sipp_generic_model = f"{git_root}/results/{p}/gait/ID_results/SMPL_generic_scale.osim"
    addB_SIPP_model = f"{git_root}/results/{p}/gait/addB-with_phys/S0{p[-1]}_HIT/Models/match_markers_and_physics.osim"
    addB_SGEN_model = f"{git_root}/results/{p}/gait/addB-with_phys/S0{p[-1]}_SGEN/Models/match_markers_and_physics.osim"

    if True: # debug
        import src.opensim.opensim_utils as opensim_utils
        gt_model = opensim_utils.get_model(gt_model)
        gt_model_df = opensim_utils.get_bsip_dataframe_with_feet_and_HAT(gt_model)
        #print(gt_model_df.head())
        

    trajectory_files = [f"{git_root}/results/{p}/gait/addB-ignore_phys/IK/Trial{trial}_filtered.mot" for trial in range(1, 7) if not (p == "P06" and trial == 5)]
    
    i = 0
    for model, res in zip(
        [scaled_generic_model, addB_model, sipp_model, sipp_generic_model, addB_SIPP_model, addB_SGEN_model],
        [scaled_res, addB_res, sipp_res, sipp_generic_res, addB_SIPP_model_res, addB_SGEN_model_res]
    ):
        rnt = False if i < 5 else False  # For the last two models, we want to return not kinetic
        res.append(mrigait_benchmark_main(
            gt_model=gt_model,
            model2=model,
            trajectory_files=trajectory_files,
            verbose=False,
            return_not_kinetic=rnt))
        i += 1
    print('===== GT model df =====', p)
    print(gt_model_df)
# Print the mean results for each model
print("Scaled Generic Model Results:")
scaled_res, addB_res, sipp_res, sipp_generic_res, addB_SIPP_model_res, addB_SGEN_model_res = \
    pd.concat(scaled_res), pd.concat(addB_res), pd.concat(sipp_res), pd.concat(sipp_generic_res), \
    pd.concat(addB_SIPP_model_res), pd.concat(addB_SGEN_model_res)
for res, name in zip([scaled_res, addB_res, sipp_res, sipp_generic_res, addB_SIPP_model_res, addB_SGEN_model_res],
                ['scaled_generic', 'addB', 'sipp', 'sipp_generic', 'addB_SIPP', 'addB_SGEN']):
    print(f'===== {name} Model Results =====')
    print(res.describe().loc[['mean', 'std']])
    print()

