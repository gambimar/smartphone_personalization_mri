import subprocess
import os 

def run_all(mode, ffilt=None):
    if ffilt is None:
        ffilt = 15
        extra_str = ''
    else:
        extra_str = f"_f{ffilt}"
    for participant in ['P01', 'P02', 'P03', 'P04', 'P05', 'P06']:
        if not mode['addB']:
            ik_path = f'results/{participant}/gait/addB-ignore_phys/IK/'
        elif mode['addB']:
            if mode['SBP']:
                ik_path = f'results/{participant}/gait/addB-with_phys/S0{participant[-1]}_{mode["pipeline"]}/IK/'
            else:
                ik_path = f'results/{participant}/gait/addB-with_phys/generic/IK/'

        # Concat IK data
        c0 = "python src/opensim/inverse_dynamics.py"
        c1 = f"--ik_path={ik_path}"
        c2 = f"--concat_ik=True"
        os.system(f"{c0} {c1} {c2}")#, shell=True)
        for trial in range(1,7):
            if (participant == 'P06') and (trial == 5):
                continue
            if (participant != 'P03') or (trial != 2):
                pass

            output_path = f'results/{participant}/gait/ID_results/Trial_{trial}{extra_str}'
            if mode['addB']:
                output_path += f"-addB"
            if mode['SBP']:
                output_path += f"-SBP-{mode['pipeline']}_{mode['scan']}"
            output_path += f".sto"

            mot_path = f'data/{participant}/gait/Trial{trial}.mot'

            if not mode['addB'] and not mode['SBP']:
                osim_path = f'results/{participant}/gait/addB-ignore_phys/Models/match_markers_but_ignore_physics.osim'
            elif mode['addB'] and not mode['SBP']:
                osim_path = f'results/{participant}/gait/addB-with_phys/generic/Models/match_markers_and_physics.osim'
            elif not mode['addB'] and mode['SBP']:
                pipeline, scan = mode['pipeline'], mode['scan']
                if mode == 'SMPL':
                    osim_path = f'results/{participant}/bodyhull/bsipw_shoes/{pipeline}_{scan}_loose.osim'
                else:
                    osim_path = f'results/{participant}/bodyhull/bsipw_shoes/{pipeline}_{scan}_loose_w_skel.osim'
            else: 
                osim_path = f'results/{participant}/gait/addB-with_phys/S0{participant[-1]}_{mode["pipeline"]}/Models/match_markers_and_physics.osim'
            print(osim_path)
            #osim_file = f'../data/{participant}/gait/Trial{trial}.osim'

            c1 = f"--ik_path={ik_path}/Trial{trial}.mot"
            c3 = f"--mot_path={mot_path}"
            c4 = f"--output_file={output_path}"
            c5 = f"--osim_file={osim_path}"
            c6 = f"--lowpass_f={ffilt}"

            os.system(f"{c0} {c1} {c3} {c4} {c5} {c6}")

mode0 = {
    'addB': True,
    'SBP': True,
    'pipeline': 'HIT',
    'scan': 'body',
}
mode1 = {
    'addB': True,
    'SBP': False,
    'pipeline': None,
    'scan': None,
}
mode2 = {
    'addB': False,
    'SBP': True,
    'pipeline': 'HIT',
    'scan': 'body',
}
mode3 = {
    'addB': False,
    'SBP': False,
    'pipeline': None,
    'scan': None,
}
mode4 = {
    'addB': True,
    'SBP': True,
    'pipeline': 'SGEN',
    'scan': None,
}

mode_SMPL = {
    'addB': False,
    'SBP': True,
    'pipeline': 'SMPL',
    'scan': 'body',
}
mode_IH = {
    'addB': False,
    'SBP': True,
    'pipeline': 'InsideHumans',
    'scan': 'skin',
}



if __name__ == "__main__":
    run_all(mode0)
    run_all(mode1)
    run_all(mode2)
    run_all(mode3)
    run_all(mode4)

    # Sensitivity analyses
    # Modelling
    run_all(mode_SMPL)
    #run_all(mode_IH) # InsideHumans is not yet available in the public repository

    # Filter tests
    for ffilt in [6,9,12,15,20,30]:

        pass
        # Compare SIPP to scaled
        run_all(mode2, ffilt=ffilt)
        run_all(mode3, ffilt=ffilt)
    
    
""" # Arguments to the ID script
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
"""