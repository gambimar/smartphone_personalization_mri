import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm



def load_nifti(file_path, get_affine=False):
    nifti = nib.load(file_path)
    if get_affine:
        return nifti.affine
    return np.flip(np.transpose(np.array(nifti.dataobj),(2,1,0)),axis=1)

def save_nifti(file_path, data, affine=np.diag([0.78125,2, 0.78125,1])):
    new_nifti = nib.Nifti1Image(np.transpose(np.flip(data, axis=1),(2,1,0)), affine)
    nib.save(new_nifti, file_path)

def concat_nifti_arrays(arrays, overlap=128, blend=False, blend_size = 0, blend_mode = "linear", axis = 0, normalize = False, normalize_treshold = 200, last_scan_offset = 400):
    """
    Combine a list of arrays into a single nifti file.
    """
    len_total = sum([arr.shape[axis] for arr in arrays]) - (len(arrays)-1)*overlap
    shape = list(arrays[0].shape)
    shape[axis] = len_total
    new_array = np.zeros(shape)
    n_arr = len(arrays)
    if normalize:
        first_value = np.mean(arrays[0][arrays[0]>normalize_treshold])
    for i, arr in enumerate(arrays):
        if normalize:
            curr_value = np.mean(arr[arr>normalize_treshold])
            arr = arr * first_value / curr_value
        if i == 0:
            new_array[-arr.shape[0]+overlap//2:,...] = arr[overlap//2:,...]
        elif i == len(arrays)-1:
            new_overlap = 640 - last_scan_offset * 640 // 500
            # position_difference is half the difference between the last scan and the second to last scan
            new_array[new_overlap-overlap:(arr.shape[0]+new_overlap//2)-overlap,...] = arr[:arr.shape[0]-new_overlap//2,...]
        else:
            new_array[(n_arr-i-1)*(arr.shape[0]-overlap)+overlap//2:(n_arr-i)*(arr.shape[0]-overlap)+overlap//2,...] = arr[overlap//2:-(overlap//2),...]
    
    assert blend == False, "Blend not implemented yet, may not be a good idea"
    return new_array

def find_deformation_characteristics(participants=[1,2,3,4,5,6], transitions=[[1,2]], power=2, aggfunc = np.median, fit_range = [4,124], overlap=128, threshold = 200):
    """
    Find the deformation characteristics of the gradient of the deformation field. We assume that the deformation field is a polynomial, and we find the squared polynomial coefficients by comparing point correspondences between two scans.
    Returns the deformation polynomial
    """
    central_y = 80
    central_z = 320
    assert np.sum(fit_range) == overlap, "Fit range must sum to overlap"

    # Find point correspondences
    all_diffs = []
    for participant in participants:
        for scan1, scan2 in transitions:
            corresponding_points = {"upper_scan_1":[], "upper_scan_2":[], "lower_scan_1":[], "lower_scan_2":[]}
            data1 = load_nifti(get_data_path(participant, scan1))
            data2 = load_nifti(get_data_path(participant, scan2))
            # Find corresponding points
            for y in range(fit_range[0], fit_range[1]):
                current_points = {"upper_scan_1":None, "upper_scan_2":None, "lower_scan_1":None, "lower_scan_2":None}
                for k in range(data1.shape[1]):
                    if data1[y,k,central_z] > threshold:
                        current_points["upper_scan_1"] = (y,k)
                        if current_points["lower_scan_1"] is None:
                            current_points["lower_scan_1"] = (y,k)
                    if data2[data1.shape[0]-overlap+y,k,central_z] > threshold:
                        current_points["upper_scan_2"] = (y,k)
                        if current_points["lower_scan_2"] is None:
                            current_points["lower_scan_2"] = (y,k)
                for key in current_points:
                    corresponding_points[key].append(current_points[key])
        # Aggregate the deformation field
        all_diffs.append((np.array(corresponding_points["upper_scan_1"])[:,1]-central_y) / (np.array(corresponding_points["upper_scan_2"])[:,1]-central_y))
        all_diffs.append((np.array(corresponding_points["lower_scan_1"])[:,1]-central_y) / (np.array(corresponding_points["lower_scan_2"])[:,1]-central_y))
    agg_array = aggfunc(np.array([*all_diffs]), axis=0)**0.5 # The true deformation field is the halfway point between the no and double deformation fields
    # polynomial fit
    poly = np.polyfit(range(fit_range[0],fit_range[1],1), agg_array, power)
    poly_y = np.poly1d(poly)(range(-300,overlap,1))

    start_idx = np.argmin(poly_y)
    return poly_y[start_idx:]-poly_y[start_idx]+1, {"fit_range":fit_range, "overlap":overlap, "threshold":threshold, "central_y":central_y, "central_z":central_z}

def apply_deformation_characteristics(deformation_characteristics, participants=[1,2,3,4,5,6], scans=[1,2,3,4,5]):
    """
    Apply the deformation characteristics to the deformation field of the participants.
    """
    poly, params = deformation_characteristics
    deform_len = len(poly)
    y_mid = params["central_y"]
    all_scans = {}
    print('')
    for participant in tqdm(participants):
        scan_list = []
        for scan in scans:
            if (participant==6) and (scan==5): continue
            data = load_nifti(get_data_path(participant, scan))
            new_data = np.zeros(data.shape)
            new_data[deform_len:-deform_len,...] = data[deform_len:-deform_len,...]
            for i in range(deform_len):
                for z in range(640):
                    p_in = np.arange(y_mid)
                    p_out = np.arange(y_mid)*poly[i]

                    new_data[-deform_len+i,:y_mid,z][::-1] = np.interp(p_in, p_out, data[-deform_len+i,:y_mid,z][::-1])
                    new_data[-deform_len+i,y_mid:,z] = np.interp(p_in, p_out, data[-deform_len+i,y_mid:,z])
                    new_data[deform_len-i,:y_mid,z][::-1] = np.interp(p_in, p_out, data[deform_len-i,:y_mid,z][::-1])
                    new_data[deform_len-i,y_mid:,z] = np.interp(p_in, p_out, data[deform_len-i,y_mid:,z])
            scan_list.append(new_data)
        all_scans[participant] = scan_list
    return all_scans

def correct_participant_6_movements(concat_array):
    cfg = {"bound_layer":576, "r_ver":8, "r_hor":1, "l_ver":8, "l_hor":8, "midpoint":320}
    # Correct left foot
    concat_array[:cfg["bound_layer"],
                 :-cfg["l_ver"],
                 cfg["l_hor"]:cfg["midpoint"]] = concat_array[:cfg["bound_layer"],
                                                                cfg["l_ver"]:,
                                                                :cfg["midpoint"]-cfg["l_hor"]]
    # Correct right foot
    concat_array[:cfg["bound_layer"],
                 :-cfg["r_ver"],
                 cfg["midpoint"]+cfg["r_hor"]:] = concat_array[:cfg["bound_layer"],
                                                                cfg["r_ver"]:,
                                                                cfg["midpoint"]:-cfg["r_hor"]]
    return concat_array

get_data_path = lambda participant, scan: f"data/P0{participant}/MRI/nifti/{scan}_fl3d_t1_setandgo_cor.nii"
get_default_result_path = lambda participant: f"results/P0{participant}/MRI/nifti/"