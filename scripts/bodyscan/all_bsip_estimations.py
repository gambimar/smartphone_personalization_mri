if __name__ == "__main__":
    import os
    import pandas as pd
    # Loop over all participants and options
    for participant in ['P01', 'P02', 'P03', 'P04', 'P05', 'P06']:
        part_info = pd.read_csv(f'data/participant_info.csv')
        bodymass = part_info[part_info['participant'] == participant].mass.values[0]
        bodymass_w_shoes = part_info[part_info['participant'] == participant].mass_treadmill.values[0]
        gender = part_info[part_info['participant'] == participant].sex.values[0]
        gender = "female" if gender == "F" else "male"
        for pipeline in ['HIT', 'SMPL', 'InsideHumans']:
            for hull in ['loose', 'tight']:
                if hull == 'tight':
                    continue
                for model in ['body', 'face', 'skin', 'scan']:
                    if pipeline == 'InsideHumans' and model != 'skin':
                        continue
                    for use_skel in [True, False]:
                        skel_c = '--use_skel=True' if use_skel else ''
                        for bm, name in zip([bodymass, bodymass_w_shoes], ['', 'w_shoes']):
                            command = f"python src/bodyscan/estimate_bsip.py --hull {hull} --pipeline {pipeline} --input_path results/{participant}/bodyhull --model {model} --output_path results/{participant}/bodyhull/bsip{name} --mass {bm} {skel_c} --gender={gender}"
                            # run the command
                            print(command)
                            os.system(command)