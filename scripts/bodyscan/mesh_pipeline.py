import json
import torch
import pandas as pd
import numpy as np

body_scan_name = "final_fit_invtrans_bodyscan.obj"
face_scan_name = "final_fit_invtrans_facescan.obj"
skin_layer_name = "skin_layer.obj"
musc_layer_name = "musc_layer.obj"
skel_layer_name = "skel_layer.obj"
scnc_layer_name = "scaneca.ply"

if __name__ == "__main__":
    # Add the parent-parent directory to the path
    import os
    import sys
    curr_file_path = os.path.abspath(__file__)
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(curr_file_path)))
    sys.path.append(parent_dir)
    os.chdir(parent_dir)
    from src.bodyscan import mesh_utils

    # Load external files
    landmarks = pd.read_csv(f"data/resources/landmarks.csv")
    participant_info = pd.read_csv("data/participant_info.csv")
    ignore_segments = json.load(open("data/resources/ignore_segments.json"))
    ignore_segments_scaneca = ['head', 'rightHandIndex1', 'leftHandIndex1']
    vertex_segmentation = json.load(open("SMPL-Fitting/smpl_vert_segmentation.json"))

    # Main loop
    for participant in range(1, 7):
        pid = f"P{str(participant).zfill(2)}"
        # create the result directory
        os.makedirs(f"results/{pid}/bodyhull/smpl_fit", exist_ok=True)
        os.makedirs(f"results/{pid}/bodyhull/transformed", exist_ok=True)

        data_path = f"data/{pid}/bodyhull"
        result_path = f"results/{pid}/bodyhull"

        skin_layer = mesh_utils.load_trimesh_from_file(f"{data_path}/{skin_layer_name}")

        if True:
            for scan_name in [body_scan_name, face_scan_name]:
                ### Scan alignment
                # Load all meshes for the participant
                scan = mesh_utils.load_trimesh_from_file(f"{data_path}/{scan_name}")
    
                # Transform the scanned meshes to the correct scale - taken from the skin layer
                scan = mesh_utils.register_scanned_meshes(skin_layer, scan)
    
                # Save the transformed meshes
                mesh_utils.save_trimesh_to_obj(scan, f"{result_path}/transformed/{scan_name[-12:]}")

                # Load landmarks for this model
                landmark_positions = {row["name"]: list(scan.vertices[row["id_avatar_scanning"]]) for _, row in landmarks.iterrows()}
                # Dump the landmarks to json
                mesh_utils.save_landmarks_to_json(landmark_positions,
                                                  f"{result_path}/transformed/{scan_name[-12:-4]}_landmarks.json")
            # Scaneca - scale to correct size
            scaneca_scale_factor = 1e-3
            scaneca = mesh_utils.load_trimesh_from_file(f"{data_path}/{scnc_layer_name}")
            scaneca.vertices = mesh_utils.rotate_scaneca_coordinate_system(scaneca.vertices * scaneca_scale_factor)
            mesh_utils.save_trimesh_to_obj(scaneca, f"{result_path}/transformed/{scnc_layer_name.replace('.ply','.obj')}")
            scaneca_landmarks = mesh_utils.get_approximate_landmarks_scaneca(participant_info.loc[participant_info['participant'] == pid, 'height'].values[0])
            mesh_utils.save_landmarks_to_json(scaneca_landmarks, f"{result_path}/transformed/{scnc_layer_name[:-4]}_landmarks.json")
            ### 3-step SMPL fitting
            for inside_humans_layer in [skel_layer_name, musc_layer_name, skin_layer_name]:
                # Load the inside humans layer
                inside_humans_mesh = mesh_utils.load_trimesh_from_file(f"{data_path}/{inside_humans_layer}")
                # Find landmarks on the inside humans layer
                landmark_positions = {row["name"]: list(inside_humans_mesh.vertices[row["id_inside_humans"]]) for _, row in landmarks.iterrows()}
                # Dump the landmarks to json
                mesh_utils.save_landmarks_to_json(landmark_positions,
                                                  f"{result_path}/transformed/{inside_humans_layer[:-4]}_landmarks.json")
        if True: # SMPL fitting
            sys.path.append("SMPL-Fitting")
            import fit_body_model
            import fit_vertices
            from utils import (load_config, process_body_model_fit_verts,
                                process_dataset_name, save_configs, set_seed,
                               initialize_fit_verts_loss_weights,load_loss_weights_config,
                               initialize_fit_bm_loss_weights)
            for mesh_name in [skel_layer_name, musc_layer_name, skin_layer_name, body_scan_name, face_scan_name, scnc_layer_name]:
                if mesh_name.startswith("final_fit_invtrans"):
                    mesh_name = mesh_name[-12:]
                    landmarks_path = f"{result_path}/transformed/{mesh_name[:-4]}_landmarks.json"
                    mesh_path = f"{result_path}/transformed/{mesh_name}"
                elif mesh_name.startswith("scaneca"):
                    mesh_name = mesh_name.replace(".ply",".obj")
                    landmarks_path = f"{result_path}/transformed/{mesh_name[:-4]}_landmarks.json"
                    mesh_path = f"{result_path}/transformed/{mesh_name}"
                else:
                    landmarks_path = f"{result_path}/transformed/{mesh_name[:-4]}_landmarks.json"
                    mesh_path = f"{data_path}/{mesh_name}"
                # SMPL Fitting
                print(pid,': 1. Fit; fitting the full model', mesh_name)
                cfg = load_config("SMPL-Fitting/configs/config.yaml")
                cfg_tight_fit = cfg['fit_vertices_optimization']

                ## 1. Fit - Full model
                cfg['fit_body_model_optimization']['use_landmarks'] = landmarks.name.to_list()
                cfg['fit_body_model_optimization']['ignore_segments'] = [] if not mesh_name.startswith("scaneca") else \
                                                                        ["rightHandIndex1", "leftHandIndex1"] # Scaneca - hands are closed
                cfg['fit_body_model_optimization']['volume_target'] = False
                sex = participant_info[participant_info["participant"] == pid].sex.values[0]
                height = participant_info[participant_info["participant"] == pid].height.values[0]
                scan_info = {"participant_id": pid,
                             "scan_name": mesh_name[:4],
                             "fit_type": "initial"}
                if mesh_name.startswith("scaneca"):
                    cfg["fit_body_model_optimization"]["loss_weight_option"] = "scaneca"
                cfg = mesh_utils.process_cfg_body_model(cfg, scan_info, mesh_path, landmarks_path)
                cfg['body_model_gender'] = "FEMALE" if sex == "F" else "MALE" #
                init_coords = torch.tensor([0, 0.9, 0] if not mesh_name.startswith("scaneca") else [-0.45, 0.9, 0.31])
                cfg['init_params'] = {'scale':torch.tensor(height/1.7), 'trans':init_coords} # ca. ~1.7 is the average SMPL height, use as initial guess
                fit_body_model.fit_body_model_onto_scan(cfg)
                res_path = f"results/{pid}/bodyhull/smpl_fit/{scan_info['scan_name']}/{scan_info['fit_type']}/{mesh_name[:-4]}.npz"
                smpl_result = np.load(res_path, allow_pickle=True)
                mesh_utils.smpl_result_to_obj(smpl_result, res_path.replace(".npz", ".obj"), has_faces=False)

                for ssm in ["smpl", "skel"]:
                    ## 2. Fit - Define ROI's
                    print(pid,': 2. Fit; ignoring segments that usually have artifacts for the', ssm, 'model')
                    # Create a version of the mesh where the ignored segments are removed
                    curr_scan = mesh_utils.load_trimesh_from_file(mesh_path)
                    cfg['ignore_segments'] = ignore_segments if not mesh_name.startswith("scaneca") else ignore_segments_scaneca
                    new_res_path = mesh_utils.create_mesh_without_ignored_segments(curr_scan, smpl_result,
                                                                    cfg['ignore_segments'], vertex_segmentation, res_path)
                    # Update the config file
                    cfg['scan_path'] = new_res_path
                    scan_info["fit_type"] = "loose" if ssm == "smpl" else "loose_skel"
                    cfg['body_model'] = ssm
                    if ssm == "skel":
                        cfg['reference_smpl'] = f"{cfg['save_path']}/{mesh_name[:-4]}_ignored_segments.obj"#
                        cfg['loss_weights'] = initialize_fit_bm_loss_weights(
                                                        load_loss_weights_config(which_strategy="fit_bm_loss_weight_strategy",
                                                        which_option="skel",
                                                        path="SMPL-Fitting/configs/loss_weight_configs.yaml"))
                    cfg['save_path'] = mesh_utils.create_results_dir(scan_info)
                    # Fit the model again
                    fit_body_model.fit_body_model_onto_scan(cfg)
                    smpl_result = np.load(f"{cfg['save_path']}/{mesh_name.replace('.obj','_ignored_segments.npz')}", allow_pickle=True)
                    mesh_utils.save_np_res_as_pkl(f"{cfg['save_path']}/{mesh_name.replace('.obj','_ignored_segments.npz')}")
                    mesh_utils.smpl_result_to_obj(smpl_result, new_res_path.replace(".npz", ".obj").replace("initial","loose"), has_faces=False)
                    save_configs(cfg)

                for ssm in ["smpl"]: #, "skel"]: # Tight SKEL fit is identical to SMPL
                    ## 3. Fit - Tight wrap
                    print(pid,': 3. Fit; tight fit without')
                    scan_info["fit_type"] = "tight" if ssm == "smpl" else "tight_skel"
                    cfg_loss_weights = load_loss_weights_config(
                        "fit_verts_loss_weight_strategy",
                        "default_option",
                        path="SMPL-Fitting/configs/loss_weight_configs.yaml")
                    cfg_loss_weights = initialize_fit_verts_loss_weights(cfg_loss_weights)
                    cfg.update(cfg_tight_fit)
                    cfg["loss_weights"]=cfg_loss_weights
                    cfg['start_from_previous_results'] = cfg['save_path']
                    cfg['start_from_body_model'] = None
                    cfg['save_path'] = mesh_utils.create_results_dir(scan_info)
                    cfg = process_body_model_fit_verts(cfg)
                    set_seed(cfg["seed"])
                    save_configs(cfg)
                    # Fit the model to vertices
                    fit_vertices.fit_vertices_onto_scan(cfg)
                    smpl_result = np.load(f"{cfg['save_path']}/{mesh_name.replace('.obj', '_ignored_segments.npz')}",
                                          allow_pickle=True)
                    mesh_utils.smpl_result_to_obj(smpl_result, f"{cfg['save_path']}/{mesh_name.replace('.obj', '_ignored_segments.obj')}", has_faces=False)
## TODO: Check all the results (6*5=30) fits
