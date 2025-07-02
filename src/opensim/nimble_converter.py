import nimblephysics as nimble
import numpy as np
import os
import pandas as pd

# We use nimble to convert from c3d to trc and mot
# Then the convertion is the same 

def c3d_to_opensim(c3d_path, output_name):
    if output_name.endswith('.trc'):
        output_name = output_name.replace('.trc', '')
    elif output_name.endswith('.mot'):
        output_name = output_name.replace('.mot', '')
    assert c3d_path.endswith('.c3d'), 'The input file should be a .c3d file'

    # Create a global version of the filepath
    if not c3d_path.startswith('/'):
        c3d_path = os.path.join(os.getcwd(), c3d_path)

    trial = trial = nimble.biomechanics.C3DLoader.loadC3D(c3d_path)
    # Convert Markers
    c3d_to_trc(trial, f"{output_name}.trc")
    # Convert Forces
    c3d_to_mot(trial, f"{output_name}.mot")
    
def c3d_to_trc(trial, output_name):
    assert output_name.endswith('.trc'), 'The output file should be a .trc file'

    # Get a array that is (n_frames, coords)
    # Write header 
    full_array, column_names = nimble_dict_to_arr(trial)
    full_array[:,2::3] *= -1 # Invert X and Z
    full_array[:,4::3] *= -1
    with open(output_name, 'w', newline="\n") as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{output_name}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{trial.framesPerSecond}\t{trial.framesPerSecond}\t{full_array.shape[0]}\t{full_array.shape[1]//3}\tm\t{trial.framesPerSecond}\t{0}\t{full_array.shape[0]}\n")
        f.write("Frame#\tTime\t")

        for marker_name in column_names:
            f.write(f"{marker_name}\t\t\t")
        f.write("\n")
        f.write("\t\t")
        for idx, _ in enumerate(trial.markers):
            f.write(f"X{idx+1}\tY{idx+1}\tZ{idx+1}\t")
        f.write("\n")
        f.flush()
        # Write data. 
        fmt = ['%.6f'] * full_array.shape[1]
        fmt[0] = '%.0f'
        fmt[1] = '%.3f'
        np.savetxt(f, full_array, delimiter='\t', fmt=fmt, newline='\n')

def nimble_dict_to_arr(trial):
    df =  pd.DataFrame(trial.markerTimesteps)
    # Find which number comes after 1 & 2
    #max_len = max(df.dropna().applymap(lambda x: len(x) if isinstance(x, np.ndarray) else 0).values.flatten())    
    df_filled = df.applymap(lambda x: x if isinstance(x, np.ndarray) else np.full(3, 0)) # nan to [0,0,0]

    # Stack using list comprehension - cause pandas is salty object handling
    rows_stacked = [np.stack(row) for row in df_filled.values]
    array_3d = np.stack(rows_stacked)

    # Reshape to 2D
    n_frames, n_markers, n_coords = array_3d.shape
    full_array = array_3d.reshape(n_frames, n_markers*n_coords)
    return np.hstack([np.arange(n_frames)[:, np.newaxis]+1,np.array(trial.timestamps)[:, np.newaxis], full_array]), df_filled.columns
    
def c3d_to_mot(trial, output_name):
    assert output_name.endswith('.mot'), 'The output file should be a .mot file'

    # Convert Analogs
    n_forces = len(trial.forcePlates)
    forces = np.hstack([np.hstack([np.array(fp.forces), np.array(fp.centersOfPressure), np.array(fp.moments)]) for fp in trial.forcePlates])
    # Rearrange the forces - First forces, then CoP, then other forces, moments last
    col_arr = np.array([*np.arange(6), *np.arange(9,15), *np.arange(6,9), *np.arange(15,18)])
    forces = forces[:,col_arr]
    # Invert X and Z
    forces[:,2::3] *= -1
    forces[:,0::3] *= -1
    # Write header
    with open(output_name, 'w', newline="\n") as f:
        #f.write(f"{output_name}\n")
        f.write("version=1\n")
        f.write("nRows=%d\n" % len(trial.timestamps))
        f.write(f"nColumns={forces.shape[1]+1}\n")
        f.write("inDegrees=yes\n")
        f.write("endheader\n")
        f.write("time\t")
        # Write CoP
        f.write("ground_force_vx\tground_force_vy\tground_force_vz\t")
        f.write("ground_force_px\tground_force_py\tground_force_pz\t")
        f.write("1_ground_force_vx\t1_ground_force_vy\t1_ground_force_vz\t")
        f.write("1_ground_force_px\t1_ground_force_py\t1_ground_force_pz\t")
        f.write("ground_torque_x\tground_torque_y\tground_torque_z\t")
        f.write("1_ground_torque_x\t1_ground_torque_y\t1_ground_torque_z\n")
        #f.write("ground_force_vx\tground_force_vy\tground_force_vz\tground_force_mx\tground_force_my\tground_force_mz\t")
        #f.write("ground_force_px\tground_force_py\tground_force_pz\t")
        #f.write("ground_force_1_vx\tground_force_1_vy\tground_force_1_vz\tground_force_1_mx\tground_force_1_my\tground_force_1_mz\t")#
        #f.write("ground_force_1_px\tground_force_1_py\tground_force_1_pz\n")
        # Write data
        data = np.hstack((np.array(trial.timestamps)[:,np.newaxis],forces))
        fmt = ['%.6f'] * data.shape[1]
        fmt[0] = '%.2f'
        np.savetxt(f, data, delimiter='\t', fmt=fmt, newline='\n')

if __name__ == '__main__':
    import os 
    # Iterate over all c3d files in data/P...
    for participant in range(1, 7):
        # Find all c3d files for this participant
        c3d_files = [f for f in os.listdir(f'data/P0{participant}/gait') if f.endswith('.c3d')]
        for c3d_file in c3d_files:
            c3d_to_opensim(f'data/P0{participant}/gait/{c3d_file}', f'data/P0{participant}/gait/{c3d_file[:-4]}')
            print('Converted Participant ', participant, ' File ', c3d_file)
