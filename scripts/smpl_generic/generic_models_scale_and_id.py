### Create SMPL-generic models
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

# I guess this is accurate, but hard to find in opensim
joint_to_body_mapping = {
    "pelvis": "hip_r",
    "femur_r": "walker_knee_r",
    "tibia_r": "ankle_r",
    "patella_r": "patellofemoral_r",
    "talus_r": "subtalar_r",
    "calcn_r": "mtp_r",
    "toes_r": "mtp_r",
    "femur_l": "hip_l",
    "tibia_l": "walker_knee_l",
    "patella_l": "patellofemoral_l",
    "talus_l": "ankle_l",
    "calcn_l": "subtalar_l",
    "toes_l": "mtp_l",
    "torso": "acromial_r",
    "humerus_r": "elbow_r",
    "ulna_r": "radioulnar_r",
    "radius_r": "radius_hand_r",
    "hand_r": "radius_hand_r",
    "humerus_l": "elbow_l",
    "ulna_l": "radioulnar_l",
    "radius_l": "radius_hand_l",
    "hand_l": "radius_hand_l"
}

import src.opensim.opensim_utils as opensim_utils   
import opensim
for p, mass, gender in zip(participants,mass_list,gender_list):
    #load the generic model
    generic_smpl_model = f"results/generic_{gender}/opensim/smpl_generic_{gender}.osim"
    addB_result_model = f"results/{p}/gait/addB-ignore_phys/Models/match_markers_but_ignore_physics.osim"

    #load the generic model
    generic_model = opensim_utils.get_model(generic_smpl_model)
    addB_result_model = opensim_utils.get_model(addB_result_model)

    bodys_smpl = generic_model.getBodySet()
    bodys_res = addB_result_model.getBodySet()

    joints_smpl = generic_model.getJointSet()
    joints_res = addB_result_model.getJointSet()
    
    joint_scale_factors = {}
    for i in range(joints_smpl.getSize()):
        # Skip the pelvis joint, because it is not scaled
        if joints_smpl.get(i).getName() == "ground_pelvis":
            continue
        joint_s = joints_smpl.get(i)
        joint_res = joints_res.get(i)
        # Scale location in parent frame
        translation = joint_s.get_frames(0).get_translation()
        t_s = np.linalg.norm(np.array([translation.get(0), translation.get(1), translation.get(2)]))

        t_r = joint_res.get_frames(0).get_translation()
        t_r = np.linalg.norm(np.array([t_r.get(0), t_r.get(1), t_r.get(2)]))

        joint_scale_factors[joint_s.getName()] = t_r/t_s
        #print(f"Joint {joint_s.getName()} mass scale factor: {joint_scale_factors[joint_s.getName()]}")

    # Check the complete mass
    mass_s = []
    mass_r = []
    for i in range(bodys_smpl.getSize()):
        body_s = bodys_smpl.get(i)
        body_res = bodys_res.get(i)
        mass_s.append(body_s.getMass()*joint_scale_factors[joint_to_body_mapping[body_s.getName()]])
        mass_r.append(body_res.getMass())

    mass_s = np.array(mass_s)
    mass_r = np.array(mass_r)
    # Enforce that the target mass adds up
    extra_scale = np.sum(mass_r)/np.sum(mass_s)
    for key in joint_scale_factors.keys():
        joint_scale_factors[key] *= extra_scale

    for i in range(bodys_smpl.getSize()):
        body_s = bodys_smpl.get(i)
        body_res = bodys_res.get(i)
        
        scale_factor = joint_scale_factors[joint_to_body_mapping[body_s.getName()]]
        # Scale mass # pow 3
        mass = body_s.getMass() * scale_factor
        body_res.setMass(mass)
        # Scale inertia # pow 5, because mass is scaled by pow 3 and radius by pow 2
        inertia = opensim_utils.to_numpy3_3(body_s.get_inertia()) * scale_factor**2

        body_res.setInertia(opensim.Inertia(
            inertia[0,0], inertia[1,1], inertia[2,2],
            inertia[0,1], inertia[0,2], inertia[1,2]
        ))

        # Scale com # pow 1
        com = opensim_utils.to_numpy3(body_s.getMassCenter()) * scale_factor / extra_scale # extra_scale is essentially extra mass, so we need to divide by it to get the correct com
        body_res.setMassCenter(opensim.Vec3(com[0], com[1], com[2]))
    
    save_path = f"results/{p}/gait/ID_results/SMPL_generic_scale.osim"
    opensim_utils.save_opensim_model(addB_result_model, save_path)#

    # === Run Inverse Dynamics ===
    name = "SIPP"
    model = "generic"

    for trial in range(1,7):
        if (p == 'P01' and trial > 2) or (p == 'P06' and trial == 5):
            continue
        if p != 'P05': continue
        if trial != 3: continue
        c0 = "python src/opensim/inverse_dynamics.py"
        c1 = f"--ik_path=results/{p}/gait/addB-ignore_phys/IK/Trial{trial}.mot"
        c2 = f"--osim_file="+os.path.abspath(f"results/{p}/gait/ID_results/SMPL_generic_scale.osim")
        c3 = f"--output_file=results/{p}/gait/ID_results/Trial{trial}_{name}_{model}.sto"
        c4 = f"--mot_path=data/{p}/gait/Trial{trial}.mot --lowpass_f=15"

        if trial == 1:
            subprocess.run(f"{c0} {c1.replace(f'Trial{trial}.mot','')} --concat_ik=True", shell=True)
        subprocess.run(f"{c0} {c1} {c2} {c3} {c4}", shell=True)# > /dev/null 2>&1", shell=True)

        # Load the result file 
    