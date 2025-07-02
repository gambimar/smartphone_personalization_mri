from pyomeca import Markers, Analogs
import numpy as np
import opensim as osim

def c3d_to_opensim(c3d_path, output_name):
    if output_name.endswith('.trc'):
        output_name = output_name.replace('.trc', '')
    elif output_name.endswith('.mot'):
        output_name = output_name.replace('.mot', '')
    assert c3d_path.endswith('.c3d'), 'The input file should be a .c3d file'

    # Convert Markers
    c3d_to_trc(c3d_path, f"{output_name}.trc")
    # Convert Forces
    c3d_to_mot(c3d_path, f"{output_name}.mot")
    
def c3d_to_trc(c3d_path, output_name):
    assert c3d_path.endswith('.c3d'), 'The input file should be a .c3d file'
    assert output_name.endswith('.trc'), 'The output file should be a .trc file'

    # Convert Markers
    markers = Markers.from_c3d(c3d_path)
    # Get a array that is (n_frames, coords)
    marker_data = markers.data[:3].reshape(markers.shape[1]*3, -1).T
    frame_idx = np.arange(marker_data.shape[0])
    time = markers.coords['time'].data
    full_array = np.hstack([frame_idx[:, np.newaxis], time[:, np.newaxis], marker_data*1e-3]).copy()

    # Write header 
    with open(output_name, 'w') as f:
        f.write(f"PathFileType\t4\t(X/Y/Z)\t{output_name}\n")
        f.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
        f.write(f"{markers.rate}\t{markers.rate}\t{markers.shape[2]}\t{markers.shape[1]}\tm\t{markers.rate}\t{markers.attrs['first_frame']}\t{markers.shape[1]}\n")
        f.write("Frame#\tTime\t")
        for marker_name in markers.coords['channel'].data:
            f.write(f"{marker_name}\t\t\t")
        f.write("\n")
        f.write("\t\t")
        for idx, _ in enumerate(markers.coords['channel'].data):
            f.write(f"X{idx+1}\tY{idx+1}\tZ{idx+1}\t")
        f.write("\n")
        f.flush()
        # Write data. 
        np.savetxt(f, full_array, delimiter='\t', fmt='%s')
    
def c3d_to_mot(c3d_path, output_name):
    assert c3d_path.endswith('.c3d'), 'The input file should be a .c3d file'
    assert output_name.endswith('.mot'), 'The output file should be a .mot file'

    # Convert Analogs
    analogs = Analogs.from_c3d(c3d_path)
    # Get a array that is (n_frames, coords)
    analog_data = analogs.data.T

    # CoP 
    cop = np.zeros((analog_data.shape[0], 6))

    # Write header
    with open(output_name, 'w') as f:
        f.write(f"GRF\nversion=1\nnRows={analog_data.shape[0]}\nnColumns=19\ninDegrees=yes\nendheader\n")
        f.write("time\t")
        namings = {
            "Force.Fx1": "ground_force_vx",
            "Force.Fy1": "ground_force_vy",
            "Force.Fz1": "ground_force_vz",
            "Moment.Mx1": "ground_torque_x",
            "Moment.My1": "ground_torque_y",
            "Moment.Mz1": "ground_torque_z",
            "Force.Fx2": "1_ground_force_vx", 
            "Force.Fy2": "1_ground_force_vy",
            "Force.Fz2": "1_ground_force_vz",
            "Moment.Mx2": "1_ground_torque_x",
            "Moment.My2": "1_ground_torque_y",
            "Moment.Mz2": "1_ground_torque_z"
            }
        idx = np.zeros(len(namings))
        for i, (key, value) in enumerate(namings.items()):
            key_idx = np.where(analogs.coords['channel'].data==key)[0][0]
            idx[i] = key_idx
            f.write(f"{value}\t")
        
        # Write CoP
        f.write("ground_force_px\tground_force_py\tground_force_pz\t1_ground_force_px\t1_ground_force_py\t1_ground_force_pz\n")
        f.flush()
        # Write data
        data = np.hstack([analogs.coords['time'].data[:, None], analog_data[:, idx.astype(int)], cop])
        np.savetxt(f, data, delimiter='\t', fmt='%s')

if __name__ == '__main__':
    import os 
    from tqdm import tqdm
    # Iterate over all c3d files in data/P...
    for participant in tqdm(range(1, 6)):
        # Find all c3d files for this participant
        c3d_files = [f for f in os.listdir(f'data/P{participant}') if f.endswith('.c3d')]
        for c3d_file in c3d_files:
            c3d_to_opensim(f'data/P{participant}/{c3d_file}', f'data/P{participant}/{c3d_file[:-4]}')
