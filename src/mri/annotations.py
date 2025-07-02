import nibabel
import nrrd
import os
import numpy as np
from tqdm import tqdm

tissue_types = {
    'prostate': "organ",
    'gluteus_maximus_left': "muscle",
    'iliopsoas_left': "muscle",
    'gluteus_medius_right': "muscle",
    'autochthon_right': "muscle",
    'iliac_vena_right': "water",
    'scapula_right': "bone",
    'humerus_right': "bone",
    'hip_right': "bone",
    'portal_vein_and_splenic_vein': "water",
    'aorta': "water",
    'femur_right': "bone",
    'femur_left': "bone",
    'stomach': "organ",
    'gallbladder': "organ",
    'duodenum': "organ",
    'clavicula_left': "bone",
    'inferior_vena_cava': "water",
    'gluteus_minimus_left': "muscle",
    'clavicula_right': "bone",
    'small_bowel': "organ",
    'iliac_artery_left': "water",
    'colon': "organ",
    'kidney_left': "organ",
    'lung_right': "lung",
    'iliac_vena_left': "water",
    'sacrum': "bone",
    'spleen': "organ",
    'autochthon_left': "muscle",
    'esophagus': "organ",
    'heart': "organ",
    'gluteus_minimus_right': "muscle",
    'adrenal_gland_left': "organ",
    'adrenal_gland_right': "organ",
    'iliac_artery_right': "water",
    'scapula_left': "bone",
    'pancreas': "organ",
    'hip_left': "bone",
    'liver': "organ",
    'intervertebral_discs': "bone",
    'iliopsoas_right': "muscle",
    'lung_left': "lung",
    'humerus_left': "bone",
    'spinal_cord': "organ",
    'gluteus_maximus_right': "muscle",
    'vertebrae': "bone",
    'brain': "brain",
    'urinary_bladder': "organ",
    'gluteus_medius_left': "muscle",
    'kidney_right': "organ",
    'vertebrae_C5': "bone",
    'vertebrae_T1': "bone",
    'vertebrae_T12': "bone",
    'vertebrae_L4': "bone",
    'vertebrae_C7': "bone",
    'vertebrae_T10': "bone",
    'vertebrae_T3': "bone",
    'vertebrae_T7': "bone",
    'vertebrae_C3': "bone",
    'vertebrae_T9': "bone",
    'vertebrae_T5': "bone",
    'vertebrae_C1': "bone",
    'vertebrae_L2': "bone",
    'sacrum': "bone",
    'vertebrae_C4': "bone",
    'vertebrae_T11': "bone",
    'vertebrae_T2': "bone",
    'vertebrae_C6': "bone",
    'vertebrae_L5': "bone",
    'vertebrae_C2': "bone",
    'vertebrae_T6': "bone",
    'vertebrae_L1': "bone",
    'vertebrae_L3': "bone",
    'vertebrae_T8': "bone",
    'vertebrae_T4': "bone",
    'tibia': "bone",
    'fibula': "bone",
    'ulna': "bone",
    'patella': "bone",
    'tarsal': "bone",
    'metatarsal': "bone",
    'phalanges_feet': "bone",
    'radius': "bone",
    'subcutaneous_fat': "fat",
    'torso_fat': "fat",
    'skeletal_muscle': "muscle",
    'body_trunc': "all",
    'body_extremities': "all",
    'teres_major': "muscle",
    'thigh_medial_compartment_right': "muscle",
    'triceps_brachii': "muscle",
    'serratus_anterior': "muscle",
    'thigh_posterior_compartment_right': "muscle",
    'subscapularis': "muscle",
    'sartorius_left': "muscle",
    'thigh_medial_compartment_left': "muscle",
    'thigh_posterior_compartment_left': "muscle",
    'infraspinatus': "muscle",
    'trapezius': "muscle",
    'coracobrachial': "muscle",
    'quadriceps_femoris_left': "muscle",
    'deltoid': "muscle",
    'pectoralis_minor': "muscle",
    'sartorius_right': "muscle",
    'supraspinatus': "muscle",
    'quadriceps_femoris_right': "muscle"
}

def tissue_types_from_ts(total_segmentator_results_path):
    '''
        Get the labels from total_segmentator and assign them a tissue class based on the tissue_types dictionary.
    '''
    # Search for all files in the total_segmentator_results_path - only in 2nd level
    nifti_files = []
    nifti_names = []
    for root, _, _ in os.walk(total_segmentator_results_path):
        for r2, _, f2 in os.walk(root):
            for file in f2:
                if file.endswith(".nii.gz"):
                    nifti_files.append(os.path.join(r2, file))
                    nifti_names.append(file.split(".")[0])

    # We use a mask for every tissue type to check for overlap
    # Therefore, get all unique entries from the tissue_types dictionary
    unique_tissue_types = list(set(tissue_types.values()))

    # load the first nifti file to get the shape
    nifti = nibabel.load(nifti_files[0])
    data = nifti.get_fdata()
    shape = data.shape
    tissue_masks = {tissue_type: np.zeros(shape, dtype=np.bool_) for tissue_type in unique_tissue_types}

    print('Loading nifti data and assigning labels')
    for i in tqdm(range(len(nifti_files))): # Not beautiful but tqdm malfunction so i changed that
        nifti = nibabel.load(nifti_files[i])
        tissue_type = tissue_types[nifti_names[i]]
        tissue_masks[tissue_type] |= (np.array(nifti.dataobj)).view(bool)

    print("Checking for overlap")
    already_checked = []
    tissue_hierarchy = {
        "bone": 0,
        "brain": 1,
        "lung": 2,
        "organ": 3,
        "muscle": 4,
        "water": 5,
        "fat": 6,
        "all": 7
    }
    for tissue_type in unique_tissue_types:
        for other_tissue_type in unique_tissue_types:
            if tissue_type != other_tissue_type:
                if (other_tissue_type, tissue_type) in already_checked:
                    continue
                overlap = tissue_masks[tissue_type] & tissue_masks[other_tissue_type]
                # Skip the next part for now, it might remove correct tissue
                #if np.sum(overlap) > 0: # Remove the overlap from the tissue type with the higher hierarchy
                #    if tissue_hierarchy[tissue_type] > tissue_hierarchy[other_tissue_type]:
                #        tissue_masks[tissue_type] &= ~overlap
                #    else:
                #        tissue_masks[other_tissue_type] &= ~overlap
                print(f"Overlap between {tissue_type} and {other_tissue_type}: {np.sum(overlap)}")
                already_checked.append((tissue_type, other_tissue_type))

    print("Create a single mask for all tissues")
    # nrrd header, copy all affine information from the first nifti file
    spacing = list(np.diag(nifti.affine)[:3])
    header = {'kinds': ['domain', 'domain', 'domain'], 'units': ['mm', 'mm', 'mm'], 
              'spacings': spacing, 'encoding': 'gzip'}
    custom_field_type_map = {}

    all_tissue_mask = np.zeros(shape, dtype=np.uint8)
    for idx, tissue_type in enumerate(unique_tissue_types):
        all_tissue_mask[tissue_masks[tissue_type]] = idx+1
        header[tissue_type] = idx+1
        custom_field_type_map[tissue_type] = "int"
    
    print("Save the nifti file, this takes ca. 30s mins")
    np.save(total_segmentator_results_path+'/all_tissue_mask.npy', all_tissue_mask)
    # Also save it as a numpy file because that is easier to load and waay quicker
    os.makedirs(os.path.join(total_segmentator_results_path, '../combined_masks'), exist_ok=True)
    for tissue_type in unique_tissue_types:
        # Save the masks as nifti files
        nifti = nibabel.Nifti1Image(tissue_masks[tissue_type].astype(np.uint8), nifti.affine)
        nibabel.save(nifti, os.path.join(total_segmentator_results_path, f'../combined_masks/{tissue_type}.nii.gz'))
    print("Done")
    