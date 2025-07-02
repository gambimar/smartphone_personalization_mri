import numpy as np
import pandas as pd
import os
import subprocess


participant_info = pd.read_csv("data/participant_info.csv")
participants = participant_info['participant'].tolist()
mass_list = participant_info['mass_treadmill'].tolist()
gender_list = participant_info['sex'].tolist()
# Replace "M" and "F" with "male" and "female"
gender_list = ["male" if g == "M" else "female" for g in gender_list]

for p, mass, _ in zip(participants, mass_list, gender_list):
    # === Run Inverse Dynamics ===
    name = "MRI"
    model = "generic"

    for trial in range(1,7):
        if (p == 'P01' and trial > 2) or (p == 'P06' and trial == 5):
            continue
        c0 = "python src/opensim/inverse_dynamics.py"
        c1 = f"--ik_path=results/{p}/gait/addB-ignore_phys/IK/Trial{trial}.mot"
        c2 = f"--osim_file="+os.path.abspath(f"results/{p}/MRI/osim_models/mri.osim")
        c3 = f"--output_file=results/{p}/gait/ID_results/Trial{trial}_{name}.sto"
        c4 = f"--mot_path=data/{p}/gait/Trial{trial}.mot --lowpass_f=15"

        if trial == 1:
            subprocess.run(f"{c0} {c1.replace(f'Trial{trial}.mot','')} --concat_ik=True", shell=True)
        subprocess.run(f"{c0} {c1} {c2} {c3} {c4} > /dev/null 2>&1", shell=True)

        # Load the result file 
    