""" 
    Runs the MRIGait benchmark for two given opensim models on a given set of trajectories.
"""

import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import opensim
import pandas as pd
import numpy as np
import argparse
from src.mrigait_benchmark.exportSignalsFromIK import get_signal
from src.opensim import opensim_utils


def main(gt_model, model2, trajectory_files, verbose = False, return_not_kinetic = False):
    """
    Main function to run the MRIGait benchmark.
    
    Args:
        model1 (str): Path to the first opensim model.
        model2 (str): Path to the second opensim model.
        trajectory_files (list of str): List of paths to the trajectory files.
        
    Returns:
        pd.DataFrame: Result of the MRIGait benchmark.
    """
    results = pd.DataFrame(columns=['mass', 'mass_center', 'inertia','kinetic_energy', 'rel_mass', 'rel_mass_center', 'rel_inertia', 'rel_kinetic_energy'])

    df_gt = opensim_utils.get_bsip_dataframe_with_feet_and_HAT(opensim_utils.get_model(gt_model))
    df2 = opensim_utils.get_bsip_dataframe_with_feet_and_HAT(opensim_utils.get_model(model2))
    # remove: HAT, talus_r, calcn_r, toes_r, toes_l, talus_l, calcn_l

    # The arms are not in the MRI model, so we remove them from the comparison
    df_gt = df_gt[~df_gt['name'].isin(['HAT', 'talus_r', 'calcn_r', 'toes_r', 'toes_l', 'talus_l', 'calcn_l', 'patella_r', 'patella_l', 'ulna_r', 'ulna_l', 'radius_r', 'radius_l', 'hand_r', 'hand_l', 'humerus_r', 'humerus_l'])]
    df2 = df2[~df2['name'].isin(['HAT', 'talus_r', 'calcn_r', 'toes_r', 'toes_l', 'talus_l', 'calcn_l', 'patella_r', 'patella_l', 'ulna_r', 'ulna_l', 'radius_r', 'radius_l', 'hand_r', 'hand_l', 'humerus_r', 'humerus_l'])]

    mass_e = (df_gt['mass'] - df2['mass']).mean()
    rel_mass_e = ((df_gt['mass'] - df2['mass']) / df_gt['mass']).mean()


    mass_center_e = np.linalg.norm(np.vstack(df_gt['mass_center'] - df2['mass_center']),axis=1).mean()
    rel_mass_center_e = (np.linalg.norm(np.vstack(df_gt['mass_center'] - df2['mass_center']),axis=1) / np.linalg.norm(np.vstack(df_gt['mass_center']),axis=1)).mean()

    inertia_e, rel_inertia_e = [], []
    for segment in df_gt['name']:
        inertia_gt = np.diag(df_gt[df_gt['name'] == segment]['inertia'].values[0])
        inertia_2 = np.diag(df2[df2['name'] == segment]['inertia'].values[0])
        inertia_e.append((inertia_gt - inertia_2).mean())
        rel_inertia_e.append(((inertia_gt - inertia_2) / inertia_gt).mean())

    inertia_e = np.array(inertia_e).mean()
    rel_inertia_e = np.array(rel_inertia_e).mean()

    #inertia_e = np.linalg.norm((df_gt['inertia'] - df2['inertia']),axis=0).mean()
    #rel_inertia_e = (np.linalg.norm((df_gt['inertia'] - df2['inertia']),axis=0) / np.linalg.norm(df_gt['inertia'],axis=0)).mean()

    if verbose:
        print(f"mass difference: {mass_e}")
        print(f"relative mass difference: {rel_mass_e}")
        print()
        print(f"mass_center difference: {mass_center_e}")
        print(f"relative mass_center difference: {rel_mass_center_e}")
        print()
        print(f"inertia difference: {inertia_e}")
        print(f"relative inertia difference: {rel_inertia_e}")
        print()

    if return_not_kinetic:
        results.loc[len(results)] = [np.abs(mass_e), np.abs(mass_center_e), np.abs(inertia_e), np.nan,
                                     np.abs(rel_mass_e), np.abs(rel_mass_center_e), np.abs(rel_inertia_e), np.nan]
        return results

    kin1, kin2 = [], []

    for trajectory_file in trajectory_files:
        # Get the signals for the models, according to raitors script. This will be easier in 
        print(f"Processing trajectory file: {trajectory_file}")
        # Split the trajectory file into parts of length 1000
        traj_df = pd.read_csv(trajectory_file, sep='\t', skiprows=10)
        for i in range(0, len(traj_df), 5000):
            end = i + 1000 if i + 1000 < len(traj_df) else len(traj_df)
            part = traj_df.iloc[i:end]
            # copy the first 10 rows to the top of the part
            with open(trajectory_file, 'r') as f:
                header = f.readlines()[:10]
            
            # Create a temporary file for the part
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.mot') as temp_file:
                temp_file.writelines(header)
                part.to_csv(temp_file, sep='\t', index=False, header=True)
                kin1.append(get_signal(gt_model, temp_file.name).kinetic_energy.to_numpy())
                kin2.append(get_signal(model2, temp_file.name).kinetic_energy.to_numpy())

    if len(kin1) > 0: 
        kin1 = np.concatenate(kin1)
        kin2 = np.concatenate(kin2)
        kin_diff = (kin1 - kin2).mean()
        rel_kin_diff = ((kin1 - kin2) / kin1).mean()

    if verbose:
        print(f"kinetic energy difference: {kin_diff}")
        print(f"relative kinetic energy difference: {rel_kin_diff}")

    results.loc[len(results)] = [np.abs(mass_e), np.abs(mass_center_e), np.abs(inertia_e), np.abs(kin_diff),
                                 np.abs(rel_mass_e), np.abs(rel_mass_center_e), np.abs(rel_inertia_e), np.abs(rel_kin_diff)]
    return results




def call_cmd():
    """
    Testing function to run the MRIGait benchmark. Only supplies a single trajectory file.
    """
    parser = argparse.ArgumentParser(description='Run the MRIGait benchmark.')
    parser.add_argument('--model1', type=str, required=True, help='Path to the first opensim model.')
    parser.add_argument('--model2', type=str, required=True, help='Path to the second opensim model.')
    parser.add_argument('--trajectories', type=str, required=True, help='Path to the trajectories file.')
    args = parser.parse_args()
    main(args.model1, args.model2, [args.trajectories], verbose=True)


if __name__ == '__main__':
    call_cmd()
    # I think opensim kinda likes global paths