## Functions to estimate the bsips from the body scan data
# Number of final models to be used for the bsip estimation:
# 1. HIT models
#   1.1 Original Pose (Body)
#   1.2 Reposed to T (Face)
#   1.3 Reposed to wide stance T (Skin)
#   1.4 Scaneca scan - Scaneca Pose (Scan)
# 2. SMPL shape-only model
#   2.1 Original Pose (Body)
#   2.2 Reposed to T (Face)
#   2.3 Reposed to wide stance T (Skin)
#   2.4 Scaneca scan - Scaneca Pose (Scan)
# 3. InsideHumans Model
#   3.1 Reposed to wide stance T (Skin) - fat layer only
#   3.2 Using SKEL's skeleton
# 4. Meta models (not in this file - we could weighted average multiple models)
### These models x2 - on the loose and on the tight hull

import argparse
import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
import numpy as np
from src.bodyscan import mesh_utils, bsip_utils, segment, orientations
import trimesh
from SKEL.skel import kin_skel


def main():
    # Parse input arguments
    parser = argparse.ArgumentParser(description='Estimate the BSIPs from body scan data')
    # Loose or tight hull
    parser.add_argument('--hull', type=str, default='loose', choices=['loose', 'tight'],
                        help='Whether to use the loose or tight hull for BSIP estimation')
    # Pipeline - HIT, SMPL, InsideHumans
    parser.add_argument('--pipeline', type=str, default='SMPL', choices=['HIT', 'SMPL', 'InsideHumans', 'MRI'],
                        help='Which pipeline to use for BSIP estimation')
    # Input path to the body scan data
    parser.add_argument('--input_path', type=str, default='results/P06/bodyhull',
                        help='Path to the body scan data')
    # Which model to use for the BSIP estimation
    parser.add_argument('--model', type=str, default='skin', choices=['body', 'face', 'skin', 'scan','generic', 'MRI'],
                        help='Which model to use for the BSIP estimation')
    parser.add_argument('--mri_hull_path', type=str, default='results/P01/MRI/final_masks/deformed')
    # Output path to save the estimated BSIPs as a pandas dataframe
    parser.add_argument('--output_path', type=str, default='results/P01/bodyhull/bsip',
                        help='Path to save the estimated BSIPs')
    # Exact body mass
    parser.add_argument('--mass', type=float, default=70.0,
                        help='Exact body mass of the participant')
    # gender
    parser.add_argument('--gender', type=str, default='female', choices=['male','female'])
    # Use the SKEL skeleton for any pipeline
    parser.add_argument('--use_skel', type=bool, default=False,
                        help='Whether to include the SKEL skeleton for the BSIP estimation')
    # Debug flag
    parser.add_argument('--debug', type=bool, default=False,
                        help='Whether to print debug information')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    """
        What happens in this function
        1. Load the body models
            1.1 SMPL && SKEL fit
            1.2 if tight: also load tight
            1.3 if HIT/Inside Humans - load additional data into dictionary
        2. Joint regression
            2.1 Use skel model
            2.2 if tight, sanity check whether the joints are close to non-tight
        3. If tight && HIT
            3.1 Reshaping needs to be done
        4. BSIP estimation
            4.1 Estimate cut planes
            4.2 Estimate BSIPs
    """
    if args.pipeline != 'MRI':
        scan_names = {
            'body': 'bodyscan',
            'face': 'facescan',
            'scan': 'scaneca',
            'skin': 'skin_layer',
            'generic': 'generic',
        }
        smpl_fit = np.load(os.path.join(args.input_path, 'smpl_fit', args.model, 'loose', f'{scan_names[args.model]}_ignored_segments.npz'), allow_pickle=True)
        #skel_fit = np.load(os.path.join(args.input_path, 'skel_fit', args.model, 'loose_skel', f'{scan_names[args.model]}_ignored_segments.npz'), allow_pickle=True)
        skel_result = mesh_utils.get_skel_output(os.path.join(args.input_path, 'smpl_fit', args.model, 'loose_skel', f'{scan_names[args.model]}_ignored_segments.npz'), args.gender)

        if args.hull == 'tight':
            smpl_fit_tight = np.load(os.path.join(args.input_path, 'smpl_fit', args.model, 'tight', f'{scan_names[args.model]}_ignored_segments.npz'), allow_pickle=True)

        if args.pipeline in ['SMPL', "HIT"]:
            meshes_of_interest = {
                "all": [mesh_utils.load_trimesh_from_file(os.path.join(args.input_path, 'smpl_fit', args.model, 'loose', f'{scan_names[args.model]}_ignored_segments.obj'))],
            }
            if args.pipeline == 'HIT':
                mesh_utils.adjust_hit_scaling(os.path.join(args.input_path, "hit_prediction", args.model, f"hit_{args.gender}_best"), smpl_fit)
                meshes_of_interest['AT'] = [mesh_utils.load_trimesh_from_file(
                    os.path.join(args.input_path, "hit_prediction", args.model, f"hit_{args.gender}_best", "AT_mesh.obj"))]
                meshes_of_interest['BT'] = [mesh_utils.load_trimesh_from_file(
                    os.path.join(args.input_path, "hit_prediction", args.model, f"hit_{args.gender}_best", "BT_mesh.obj"))]
                meshes_of_interest['LT'] = [mesh_utils.load_trimesh_from_file(
                    os.path.join(args.input_path, "hit_prediction", args.model, f"hit_{args.gender}_best", "LT_mesh.obj"))]
        elif args.pipeline == 'InsideHumans':
            meshes_of_interest = {
                "skin": [mesh_utils.load_trimesh_from_file(os.path.join(args.input_path, 'smpl_fit', 'skin', 'loose', 'skin_layer_ignored_segments.obj'))],
                "musc": [mesh_utils.load_trimesh_from_file(os.path.join(args.input_path, 'smpl_fit', 'musc', 'loose', 'musc_layer_ignored_segments.obj'))],
            }
        else:
            raise ValueError(f"Pipeline {args.pipeline} does not exist. Choices: ['SMPL', 'HIT', 'InsideHumans']")

        if args.use_skel:
            meshes_of_interest['skel'] = [trimesh.Trimesh(vertices=skel_result['skel_verts'], faces=skel_result['skel_faces'])]

        # === Deform meshes to the tight hull (if needed) ===
        if args.hull == 'tight':
            if args.pipeline == "InsideHumans":
                meshes_of_interest = { # Tight fits are already known
                    "skin": [mesh_utils.load_trimesh_from_file(os.path.join(args.input_path, 'smpl_fit', 'skin', 'tight', 'skin_layer_ignored_segments.obj'))],
                    "musc": [mesh_utils.load_trimesh_from_file(os.path.join(args.input_path, 'smpl_fit', 'musc', 'tight', 'musc_layer_ignored_segments.obj'))],
                }
            if args.pipeline in ['SMPL', 'HIT']:
                meshes_of_interest['all'] = [trimesh.Trimesh(vertices=smpl_fit_tight['vertices'], faces=skel_result['skin_faces'])]
                if args.pipeline == 'HIT':
                    raise NotImplementedError("Reshaping for HIT models is not implemented yet")
                if args.use_skel:
                    raise NotImplementedError("Reshaping with SKEL is not implemented yet")

    else:
        meshes_of_interest = {
            "all": [mesh_utils.load_trimesh_from_file(os.path.join(args.mri_hull_path, "all.obj"))],
            "cortical_bone": [mesh_utils.load_trimesh_from_file(os.path.join(args.mri_hull_path, "cortical_bone.obj"))],
            "trabecular_bone": [mesh_utils.load_trimesh_from_file(os.path.join(args.mri_hull_path, "trabecular_bone.obj"))],
            "fat": [mesh_utils.load_trimesh_from_file(os.path.join(args.mri_hull_path, "fat.obj"))],
            "muscle": [mesh_utils.load_trimesh_from_file(os.path.join(args.mri_hull_path, "muscle.obj"))],
            "lung": [mesh_utils.load_trimesh_from_file(os.path.join(args.mri_hull_path, "lung.obj"))],
            #"other": [mesh_utils.load_trimesh_from_file(os.path.join(args.mri_hull_path, "other.obj"))],
        }
        skel_result = mesh_utils.get_skel_output(os.path.join(args.mri_hull_path, '..', '..', 'bodyhull', 'loose_skel', f'all.npz'), args.gender)

    # === HIT Error correction, flipping faces by heuristics in case blender segmentation flips them  ===
    if args.pipeline == "HIT":
        # I have 0 trust in HIT's marching cubes, so we bisect everything with the "all" mesh
        import src.bodyscan.slice_mesh_blender as smb
        for mesh_name, mesh in meshes_of_interest.items():
            if mesh_name == "all":
                continue
            print(f"Cutting {mesh_name} with the all mesh")
            # Cut the mesh with the all mesh
            cut_mesh = smb.slice_mesh_with_blender(mesh[0], meshes_of_interest["all"][0], "INTERSECT")
            if cut_mesh:
                meshes_of_interest[mesh_name] = [cut_mesh]
            else:
                raise ValueError(f"Could not cut {mesh_name} with the all mesh")
            if args.debug:
                # print volume loss
                original_volume = np.sum([part.volume for part in mesh])
                new_volume = np.sum([part.volume for part in meshes_of_interest[mesh_name]])
                print(f"Volume loss for {mesh_name}: {original_volume*1000:.2f} -> {new_volume*1000:.2f} ({(original_volume - new_volume) / original_volume * 100:.2f}%)")


    # === Cut meshes ===
    result_meshes = {}
    for mesh_name, mesh in meshes_of_interest.items():
        allow_convex_hull = 'default'
        result_meshes[mesh_name] = {}
        all_bodies = segment.all_bodies if args.pipeline != 'MRI' else segment.mri_bodies
        for body in all_bodies:
            ##if body != "femur_r": continue
            if args.pipeline == 'MRI': print(f"Cutting {mesh_name} for {body}")
            try_fix = False if mesh_name == "skel" else True # Skel has broken faces, so we don't want to fix them
            if args.pipeline == "MRI":
                allow_convex_hull = 'default' if mesh_name == "all" else 'no'
                result_meshes[mesh_name][body] = segment.cut_segment(body, mesh, skel_result['joints'], try_fix=try_fix, allow_convex_hull=allow_convex_hull, split_first=False, mesh_name = mesh_name)
            else:
                result_meshes[mesh_name][body] = segment.cut_segment(body, mesh, skel_result['joints'], try_fix=try_fix)
            
    # === HIT: Fix windings: all interior meshes are flipped ===
    if args.pipeline == "HIT":
        for mesh_name, mesh in result_meshes.items():
            if mesh_name not in ['all', 'BT', 'skel']:
                for body in all_bodies:
                    if body not in mesh:
                        continue
                    if type(mesh[body]) != trimesh.Trimesh:
                        print(f"Mesh {mesh_name} for {body} is not a trimesh, but {type(mesh[body])}")
                        continue
                    n_bodies = 2 if body == "lumbar_body" else 1 # Safety cuts results in 1 valid bodies
                    meshes = result_meshes[mesh_name][body].split(only_watertight=True)
                    # Sort meshes by volume
                    meshes = sorted(meshes, key=lambda x: x.volume, reverse=True)
                    for i in range(len(meshes)):
                        if i >= n_bodies:
                            meshes[i].faces = meshes[i].faces[:, ::-1]
                    result_meshes[mesh_name][body] = trimesh.util.concatenate(meshes)
                        


    # === Debug report ===
    if args.debug:
        print(args.debug)
        print("Meshes of result:", result_meshes.keys(), "MOI:", meshes_of_interest.keys())
        for mesh_name, mesh in result_meshes.items():
            print(f"Mesh name: {mesh_name}")
            original_volume = np.sum([part.volume for part in meshes_of_interest[mesh_name]])
            new_volume = 0
            for body, cut_mesh in mesh.items():
                if cut_mesh:
                    new_volume += cut_mesh.volume
            print(
                f"Volume: (new | old | diff) [L]: {new_volume * 1e3:.2f} | {original_volume * 1e3:.2f} | {(new_volume - original_volume) * 1e3:.2f}")
            # Save all body parts as .obj in results/tmp
            os.makedirs('results/tmp', exist_ok=True)
            for body, cut_mesh in mesh.items():
                if not cut_mesh:
                    continue
                scene = trimesh.Scene(cut_mesh)
                try:
                    scene.export(f"results/tmp/{mesh_name}_{body}.obj")
                except:
                    print(f"Could not save {mesh_name}_{body}.obj")

    # Get orientations per body part
    body_orientations = orientations.get_bone_orientation(skel_result['pose'])

    # Estimate the BSIPs
    bsips = bsip_utils.estimate_bsip(result_meshes, body_orientations, skel_result['joints'], args)

    # Adjust for bodymass
    total_mass = np.sum(bsips['mass'])
    print(f"Total mass: {total_mass}, Adjusting to {args.mass}")
    bsips['mass'] = bsips['mass'] / total_mass * args.mass
    bsips['inertia'] = bsips['inertia'].apply(lambda x: x / total_mass * args.mass)

    # Save the BSIPs as a pandas dataframe
    bsips.to_pickle(os.path.join(args.output_path, f'{args.pipeline}_{args.model}_{args.hull}{"_w_skel" if args.use_skel else ""}.pkl'))
    print(f"BSIPs estimated and saved to ",{os.path.join(args.output_path, f'{args.pipeline}_{args.model}_{args.hull}{"_w_skel" if args.use_skel else ""}.pkl')})



"""
    Some in between project management ketchup
    Gait:
        - process all data in addbiomechanics
        - do ID for all participants and models (scaled, addB, our models)
        (- process new initial guess in addbiomechanics) (if results are okish)
        (- do ID for that data again -- addB is not exactly solving the same thing as opensim)
        (- Compare with OpenSim RRA, NMSM builder)
    MRI:
        - correct labels, export meshes
        - body model fit loose and tight, reshapeing
        - NMS builder for a more exact skeleton?
        - ID with loose + tight, addB and NMS models
    Bodyhull:
        - Get bsip (this file)
"""

if __name__ == '__main__':
    main()