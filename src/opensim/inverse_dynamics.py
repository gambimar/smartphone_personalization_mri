import argparse
import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
from src.opensim import opensim_utils
import opensim


def main():
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Run inverse dynamics based on IK and ID data')
    parser.add_argument('--ik_path', type=str, help='Path to the inverse kinematics data')
    parser.add_argument('--concat_ik', type=bool, default=False,
                        help='Addbiomachanics data is in multiple files, concatenate them. This prohibits running ID')
    parser.add_argument('--mot_path', type=str, default='')
    parser.add_argument('--output_file', type=str, default='')
    parser.add_argument('--osim_file', type=str, default='')
    parser.add_argument('--lowpass_f', type=float, default=6.0, help='Lowpass filter frequency')
    parser.add_argument('--lowpass_order', type=int, default=2, help='Lowpass filter order')
    parser.add_argument('--id_setup_file', type=str, default='results/prelim/id_setup.xml', help='Path to the ID setup file')
    parser.add_argument('--external_forces_file', type=str, default='results/prelim/external_forces_settings.xml', help='Path to the external forces file')
    args = parser.parse_args()

    if args.concat_ik:
        assert args.ik_path is not None, 'If you want to concatenate the IK data, you need to provide the path to the data'
        assert not args.ik_path.endswith(".mot"), "The path to the IK data should be a folder when concatenating"
        opensim_utils.concat_addbiomechanics_data(args.ik_path)
        print(f"Trials concatenated and saved to {args.ik_path}. Call this function with a specific trial to run ID")
        return
    else:
        assert args.ik_path.endswith(".mot"), "The path to the IK data should be a .mot file"

    assert all([args.ik_path, args.mot_path, args.output_file, args.osim_file]), 'You need to provide all the paths'

    # Filter both IK and ID data if given
    if args.lowpass_f > 0:
        ik_path, ik_df = opensim_utils.filter_data(args.ik_path, args.lowpass_f, args.lowpass_order, return_df = True, mode='ik')
        grf_path = opensim_utils.filter_data(args.mot_path, args.lowpass_f, args.lowpass_order, mode='grf', clean=True)
    else:
        ik_path = args.ik_path
        grf_path = args.mot_path

    # opensim ID tool
    model = opensim.Model(args.osim_file)
    model.initSystem()
    id_tool = opensim.InverseDynamicsTool(args.id_setup_file)
    id_tool.setModel(model)

    id_tool.setCoordinatesFileName(glob_path(ik_path))
    id_tool.setExternalLoadsFileName(set_grf_path(args.external_forces_file,glob_path(grf_path)))
    id_tool.set_results_directory(glob_path(os.path.join(*(args.output_file.split('/')[:-1]))))
    id_tool.setOutputGenForceFileName(glob_path(args.output_file.split('/')[-1]))
    id_tool.setLowpassCutoffFrequency(-1) # Do not use OpenSims filter
    # Find start and end times from the ik_file - for later
    id_tool.setStartTime(ik_df['time'].iloc[0])
    id_tool.setEndTime(ik_df['time'].iloc[-1])
    print("----- Running ID -----")
    print(id_tool.getOutputGenForceFileName())
    print(id_tool.getStartTime())
    print(id_tool.getEndTime())
    print(id_tool.getCoordinatesFileName())
    print(id_tool.getLowpassCutoffFrequency())
    id_tool.run()

def glob_path(path):
    return os.path.join(os.getcwd(), path)

def set_grf_path(external_forces_file, grf_path):
    with open(external_forces_file, 'r') as f:
        with open(external_forces_file.replace('.xml', '_edited.xml'), 'w') as f2:
            for line in f:
                if 'datafile' in line:
                    f2.write(f'\t\t<datafile>{glob_path(grf_path)}</datafile>\n')
                else:
                    f2.write(line)
    return glob_path(external_forces_file.replace('.xml', '_edited.xml'))

if __name__ == '__main__':
    main()