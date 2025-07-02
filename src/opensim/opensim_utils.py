"""
This file contains utility functions for OpenSim models. As OpenSim is incompatible with numpy, we need to convert between the two.
"""
import opensim as osim
import numpy as np
import pandas as pd
import os

def get_model(model_file):
    mod = osim.Model(model_file)
    state = mod.initSystem()
    return mod

def get_joint_translation(model, body_name):
    parent_joint_name = get_parent_joint(model, body_name)
    for joint in range(model.getJointSet().getSize()):
        jour = model.getJointSet().get(joint)
        if jour.getName() == parent_joint_name:    
            translation = jour.get_frames(0).get_translation()
    return np.array([translation.get(0), translation.get(1), translation.get(2)])

def to_numpy3(vec3): # For com vectors
    return np.array([vec3.get(0), vec3.get(1), vec3.get(2)])

def to_numpy3_3(mat3): # for inertia matrices
    return np.array([[mat3.get(0), mat3.get(3), mat3.get(4)], 
                     [mat3.get(3), mat3.get(1), mat3.get(5)], 
                     [mat3.get(4), mat3.get(5), mat3.get(2)]])

def get_bsip_dataframe(model):
    body_set = model.getBodySet()
    nSeg = body_set.getSize()
    df = pd.DataFrame(columns=['name', 'mass', 'mass_center', 'inertia'])
    for i in range(nSeg):
        body = body_set.get(i)
        df.loc[i] = [body.getName(), body.getMass(), to_numpy3(body.getMassCenter()), to_numpy3_3(body.get_inertia())]
    return df

def get_joint_dataframe(model):
    joint_set = model.getJointSet()
    nJoints = joint_set.getSize()
    df = pd.DataFrame(columns=['name', 'parent', 'child', 'location', 'orientation'])
    for i in range(nJoints):
        joint = joint_set.get(i)
        df.loc[i] = [joint.getName(), joint.getParentFrame().getName(), joint.getChildFrame().getName(), to_numpy3(joint.get_location_in_parent()), to_numpy3(joint.get_orientation_in_parent())]
    return df

def get_parent_joint(model, body_name):
    joint_set = model.getJointSet()
    for i in range(joint_set.getSize()):
        joint = joint_set.get(i)
        if joint.getChildFrame().getName() == body_name+'_offset':
            return joint.getName()
    return None

def get_bsip_dataframe_with_feet_and_HAT(model, foot = ['calcn','talus','toes'], HAT = ['pelvis','torso']):
    bodies_df = get_bsip_dataframe(model)
    #joint_df = get_joint_dataframe(model)
    for side in ['_l', '_r']:
        # Add translations 
        foot_mass = 0
        for f in foot:
            foot_mass += bodies_df.loc[bodies_df['name'] == f+side, 'mass'].values[0]
        foot_mass_center = np.zeros(3)
        trans = np.zeros(3)
        for f in foot:
            if foot != 'calcn': # calcn is the parent segment for the foot
                trans += get_joint_translation(model, f+side)
            foot_mass_center += bodies_df.loc[bodies_df['name'] == f+side, 'mass'].values[0]*(bodies_df.loc[bodies_df['name'] == f+side, 'mass_center'].values[0]+trans)
        foot_mass_center /= foot_mass
        foot_inertia = np.zeros((3,3))
        trans = np.zeros(3)
        for f in foot:
            if foot != 'calcn':
                trans += get_joint_translation(model, f+side)
            foot_inertia += bodies_df.loc[bodies_df['name'] == f+side, 'inertia'].values[0] + \
                    bodies_df.loc[bodies_df['name'] == f+side, 'mass'].values[0] * \
                    np.outer(foot_mass_center+trans, foot_mass_center+trans)
        bodies_df.loc[bodies_df.shape[0]] = ['foot'+side, foot_mass, foot_mass_center, foot_inertia]
    HAT_mass = 0
    for h in HAT:
        HAT_mass += bodies_df.loc[bodies_df['name'] == h, 'mass'].values[0]
    HAT_mass_center = np.zeros(3)
    trans = np.zeros(3)
    for h in HAT: # pelvis is the parent segment for the torso
        if h != 'pelvis':
            trans += get_joint_translation(model, h)
        HAT_mass_center += bodies_df.loc[bodies_df['name'] == h, 'mass'].values[0]*(bodies_df.loc[bodies_df['name'] == h, 'mass_center'].values[0]+trans)
    HAT_mass_center /= HAT_mass
    HAT_inertia = np.zeros((3,3))
    trans = np.zeros(3)
    for h in HAT:
        if h != 'pelvis':
            trans += get_joint_translation(model, h)
        HAT_inertia += bodies_df.loc[bodies_df['name'] == h, 'inertia'].values[0] + \
                bodies_df.loc[bodies_df['name'] == h, 'mass'].values[0] * \
                np.outer(HAT_mass_center+trans, HAT_mass_center+trans)
    bodies_df.loc[bodies_df.shape[0]] = ['HAT', HAT_mass, HAT_mass_center, HAT_inertia]
    return bodies_df

def set_opensim_bsip_parameters(model, bodies_df):
    ''' 
        Set the mass, mass center and inertia parameters of the bodies in the OpenSim model from a pandas dataframe.
    '''
    # Filter out the HAT and foot segments if they appear in the dataframe
    bodies_df = bodies_df[~bodies_df['name'].isin(['HAT', 'foot_l', 'foot_r'])]
    for idx, body in bodies_df.iterrows():
        seg = model.getBodySet().get(body['name'])
        seg.setMass(float(body['mass']))
        seg.setMassCenter(osim.Vec3(body['mass_center'][0], body['mass_center'][1], body['mass_center'][2]))
        seg.setInertia(osim.Inertia(body['inertia'][0,0], body['inertia'][1,1], body['inertia'][2,2], body['inertia'][0,1], body['inertia'][0,2], body['inertia'][1,2]))
    return model

def save_opensim_model(model, file_name):
    model.printToXML(file_name)


def concat_addbiomechanics_data(ik_data_path):
    # Find all mot files in the folder
    mot_files = sorted([f for f in os.listdir(ik_data_path) if f.endswith('.mot')])
    # Check unique trials
    trials = set([f.split('_')[0] for f in mot_files])
    for t in trials:
        # Skip trials without _, as they are not produced by addbiomechanics
        if t.find('.mot') > 0:
            continue
        # Find all files for the trial
        trial_files = [f for f in mot_files if f.startswith(t) and '_segment' in f]
        # Sort the trial files by the segment number as int
        trial_files = sorted(trial_files, key=lambda x: int(x.split('_')[-2].split('_')[0]))

        # Load all trials
        df = pd.concat([pd.read_csv(os.path.join(ik_data_path, f), sep='\t', skiprows=10) for f in trial_files])
        nRows = df.shape[0]
        # Create a new file, with the same name as the set
        with open(os.path.join(ik_data_path, f"{t}.mot"), 'w', newline='\n') as f:
            with open(os.path.join(ik_data_path, trial_files[0]), 'r') as f0:
                # Write the first two lines from the original file
                for i in range(2):
                    f.write(f0.readline())
                f0.readline() # Skip the third line
                f.write(f"nRows={nRows}\n")
                for i in range(7):
                    f.write(f0.readline())
            df.to_csv(f, sep='\t', index=False)



def filter_data(data_path, lowpass_f, lowpass_order, mode='ik', return_df=False, clean=False):
    from scipy.signal import butter, sosfiltfilt
    if mode == 'ik':
        skiprows = 10
    elif mode == 'grf':
        skiprows = 5
    else:
        raise ValueError('Filter mode should be either marker or grf')
    

    data = pd.read_csv(data_path, sep='\t', skiprows=skiprows)
    if clean:
        # Clamp the center of pressure values to min/max values of 1.5m (bigger than the treadmill dimensions)
        # and the torques to -100Nm to 100N*m
        # This is done to avoid unrealistic values in the data, which would then get filtered and cause more issues
        for col in data.columns:
            if col[-2] == 'p':
                data[col] = data[col].clip(-1.5, 1.5)
            if col.endswith('_torque_y'):
                # Clamp the torque values to -100Nm to 100N*m
                data[col] = data[col].clip(-100, 100)

    # Lowpass filter the data
    def butter_lowpass_filter(data, cutoff, fs, order):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
        y = sosfiltfilt(sos, data)
        return y

    for col in data.columns:
        if col in ['time', 'Time', 'Frame#']:
            continue # Do not filter time, looking at you opensim
        data[col] = butter_lowpass_filter(data[col], lowpass_f, 100, lowpass_order)
        data[col] = data[col].apply(lambda x: round(x, 8))

    # Testwise: Set the mean of the force plate data to 0 - it is constantly biased somehow...
    if mode == 'grf':
        for col in data.columns:
            if col in ['ground_force_vx', 'ground_force_vz', '1_ground_force_vx', '1_ground_force_vz']:
                print('Mean of', col, 'is', data[col].mean())

    with open(data_path.replace('.mot', '_filtered.mot'), 'w', newline='\n') as f:
        with open(data_path, 'r') as f0:
            for i in range(skiprows+1):
                f.write(f0.readline())
        data_arr = data.to_numpy()
        # Somehow, thats the weird thing: the data is not written correctly if we use pd.to_csv
        # But literally 10 lines above this it works fine. I have no idea why
        np.savetxt(f, data_arr, delimiter='\t', newline='\n', fmt='%.8f')
        f.flush()

    if return_df:
        return data_path.replace('.mot', '_filtered.mot'), data
    return data_path.replace('.mot', '_filtered.mot')

def hull_bsip_to_opensim(hull_bsip, model,
                         arms_from_hull=False,
                         extra_shoe_weight=0,
                         keep_bodies=False):

    # Check if lumbar_body is in the hull_bsip.names and replace it with "torso"
    if 'lumbar_body' in hull_bsip['segment'].values:
        hull_bsip.loc[hull_bsip['segment'] == 'lumbar_body', 'segment'] = 'torso'

    hull_bsip.rename(columns={'segment': 'name'}, inplace=True)

    # Define arms segments
    if not arms_from_hull:
        arm_segments = ['humerus_r', 'humerus_l', 'radius_r', 'radius_l', 'ulna_r', 'ulna_l', 'hand_r', 'hand_l']
    else: 
        arm_segments = []
    if type(keep_bodies) == list: # We simply keep the bodies that are desired (usually the toes in MRI, they are out of volume)
        arm_segments += keep_bodies

    model_bsip = get_bsip_dataframe(model)
    bodymass = hull_bsip['mass'].sum()

    if not arms_from_hull or keep_bodies:
        # Use the original BSIP of the model for the arms
        # Remove the arms from the hull_bsip
        hull_bsip = hull_bsip[~hull_bsip['name'].isin(arm_segments)]
        # Add the arms from the model
        hull_bsip = pd.concat([hull_bsip, model_bsip[model_bsip['name'].isin(arm_segments)]])

        # check the total mass and adjust mass and inertia parameters
        bodymass_new = hull_bsip['mass'].sum()
        hull_bsip['mass'] = hull_bsip['mass'] / bodymass_new * bodymass
        hull_bsip['inertia'] = hull_bsip['inertia'].apply(lambda x: x / bodymass_new * bodymass)
    else: 
        # Rotate the arms to the correct orientation, discrepancy between skel and opensim - T-pose or handing arms defaults
        from scipy.spatial.transform import Rotation as R
        for arm in arm_segments:
            if arm in hull_bsip['name'].values:
                if arm.endswith('_r'):
                    rotate = [0,0,-np.pi/2]
                else:
                    rotate = [0,0,np.pi/2]
                r =  R.from_euler('xyz', rotate, degrees=False)
                hull_bsip.loc[hull_bsip['name'] == arm, 'mass_center'] = hull_bsip.loc[hull_bsip['name'] == arm, 'mass_center'] @ r.as_matrix()
                hull_bsip.loc[hull_bsip['name'] == arm, 'inertia'] = hull_bsip.loc[hull_bsip['name'] == arm, 'inertia'].apply(lambda x: r.as_matrix() @ x @ r.as_matrix().T)

    if extra_shoe_weight > 0:
        # Add the weight of the shoes to the feet
        feet = ['calcn_r', 'calcn_l', 'talus_r', 'talus_l', 'toes_r', 'toes_l']

        starting_foot_mass = hull_bsip[hull_bsip['name'].isin(feet)]['mass'].sum()
        scale_factor = (extra_shoe_weight + starting_foot_mass) / starting_foot_mass

        hull_bsip.loc[hull_bsip['name'].isin(feet), 'mass'] *= scale_factor
        hull_bsip.loc[hull_bsip['name'].isin(feet), 'inertia'] = hull_bsip.loc[hull_bsip['name'].isin(feet), 'inertia'].apply(lambda x: x * scale_factor)

    # Set all undefined bisp values to (almost) 0 - patella (not segmented), radius (same body as ulna), talus (same body as calcn)
    missing_bsip = model_bsip[~model_bsip['name'].isin(hull_bsip['name'])]
    missing_bsip.loc[:,'mass'] = 1e-5
    missing_bsip.loc[:,'mass_center'] = missing_bsip['mass_center'].apply(lambda x: np.array([1e-5, 1e-5, 1e-5]))
    missing_bsip.loc[:,'inertia'] = missing_bsip['inertia'].apply(lambda x: np.eye(3) * 1e-5)
    hull_bsip = pd.concat([hull_bsip, missing_bsip])

    return hull_bsip, set_opensim_bsip_parameters(model, hull_bsip)





