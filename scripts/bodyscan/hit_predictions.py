import subprocess
import os
import torch
import pandas as pd
import sys

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Choose the working directory as the root of the HIT repository, which is two levels up from the current file
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'HIT')))
sys.path.append(os.getcwd()) # Im confused

participant_info = pd.read_csv('../data/participant_info.csv')
commandlist = []
for participant in ['P01', 'P02', 'P03', 'P04', 'P05', 'P06']:
    continue
    for scan in ['body','face','scan','skin']: # No HIT inference for inner objs
        scan_names = {
            'body': 'bodyscan',
            'face': 'facescan',
            'scan': 'scaneca',
            'skin': 'skin_layer'
        }
        input_path = f"../results/{participant}/bodyhull/smpl_fit/{scan}/loose/{scan_names[scan]}_ignored_segments.pkl"
        output_path = f"../results/{participant}/bodyhull/hit_prediction/{scan}/"
        gender = participant_info[participant_info['participant'] == participant].sex.values[0]
        gender = 'male' if gender == 'M' else 'female'
        command = f"python demos/infer_smpl.py --exp_name=hit_{gender} --target_body {input_path} --device {device} --to_infer smpl_file --out_folder {output_path}"
        commandlist.append(command)
    
for generic in ['generic_male', 'generic_female', 'generic_neutral']:
    input_path = f"../results/{generic}/bodyhull/smpl_fit/generic/generic_model.pkl"
    output_path = f"../results/{generic}/bodyhull/hit_prediction/generic/"
    os.makedirs(output_path, exist_ok=True)
    gender = generic.split('_')[-1]
    command = f"python demos/infer_smpl.py --exp_name=hit_{gender} --target_body {input_path} --device {device} --to_infer smpl_file --out_folder {output_path}"
    commandlist.append(command)

if __name__ == '__main__':
    for command in commandlist:
        print(command)
        subprocess.run(command, shell=True)
