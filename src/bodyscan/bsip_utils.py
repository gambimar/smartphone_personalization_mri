import pandas as pd
from zstandard.backend_cffi import new_nonzero

from src.bodyscan import segment
from SKEL.skel import kin_skel
from scipy.spatial.transform import Rotation as R
import trimesh.transformations as tf
import trimesh
import numpy as np

# Densities from https://itis.swiss/virtual-population/tissue-properties/database/density/
default_densities = {
    "HIT":
        {
            "all_torso": 394, # Lung tissue only
            "all": 1050, # All other tissues
            "BT": 1178, # Bone tissue only - the bones aren't very exact
            "AT": 911, # Adipose tissue only
            "LT": 1090, # Lean tissue only
            "LT_torso": 1050,
            "skel": 1178, # Bone tissue only
        },
    "SMPL":
        {
            "all": 1050,
            "skel": 1178,
        },
    "InsideHumans":
        {
            "skin": 911,
            "musc": 1090,
            "skel": 1178,
        },
    "SKEL":
        {
            "pelvis": 1178,
        },
    "MRI":
        {
            "all": 1050, # All others should be 
            "trabecular_bone": 1178,
            "cortical_bone": 1178,
            "fat": 911,
            "muscle": 1090,
            "lung": 394,
            "other": 1050,
        }
}


def estimate_bsip(result_meshes, body_orientations, joints, args):
    """
    Estimate the BSIP of the body segments based on the mesh and the orientations
    :param result_meshes: Dictionary of mesh segments
    :param body_orientations: Dictionary of body orientations
    :param args: Arguments
    :return: Dictionary of BSIPs
    """
    if args.pipeline == "MRI":
        all_bodies = segment.mri_bodies
    else:
        all_bodies = segment.all_bodies
    result = pd.DataFrame(columns=['segment', 'mass', "mass_center", 'inertia'])
    for body in all_bodies:
        mesh_sip = {}
        for mesh_name, mesh in result_meshes.items():
            if body not in mesh:
                raise ValueError(f"Body {body} is not in the mesh {mesh}")
            if args.pipeline == "HIT" and mesh_name == "BT" and args.use_skel:
                continue # Skip this mesh, because we add it to the lean tissue
            elif args.pipeline == "HIT" and mesh_name == "LT" and args.use_skel:
                combined = trimesh.util.concatenate(mesh[body], result_meshes["BT"][body])
                mesh[body] = combined if type(combined) != list else False
            orientation = body_orientations[body]
            joint_position = joints.squeeze()[kin_skel.skel_joints_name.index(body)]
            mass, com = estimate_segment_bsip(mesh[body], orientation, joint_position)
            mesh_sip[mesh_name] = {
                'mesh': mesh[body],
                'mass': mass,
                "mass_center": com,
            }

        # Remove overlapping meshes
        if args.pipeline == "HIT" and not args.use_skel:
            mesh_sip = get_inertia_and_update_main_body_mass(mesh_sip, "all", ["AT", "BT", "LT"])
        elif args.pipeline == "HIT" and args.use_skel:
            if mesh_sip["LT"]["mesh"]:
                mesh_sip = get_inertia_and_update_main_body_mass(mesh_sip, "LT", ["skel"]) # Remove SKEL from the lean/bony tissue
            mesh_sip = get_inertia_and_update_main_body_mass(mesh_sip, "all", ["AT", "LT", "skel"]) # Remove the lean tissue from the all
        elif args.pipeline == "SMPL" and args.use_skel:
            mesh_sip = get_inertia_and_update_main_body_mass(mesh_sip, "all", ["skel"])
        elif args.pipeline == "InsideHumans" and args.use_skel:
            mesh_sip = get_inertia_and_update_main_body_mass(mesh_sip, "musc", ["skel"])
            mesh_sip = get_inertia_and_update_main_body_mass(mesh_sip, "skin", ["musc", "skel"])
        elif args.pipeline == "InsideHumans" and not args.use_skel:
            mesh_sip = get_inertia_and_update_main_body_mass(mesh_sip, "skin", ["musc"])
        elif args.pipeline == "MRI":
            mesh_sip = get_inertia_and_update_main_body_mass(mesh_sip, "all", ["trabecular_bone", "cortical_bone", "fat", "muscle", "lung"])
        else: # Only the SMPL model, do nothing
            mesh_sip = get_inertia_and_update_main_body_mass(mesh_sip, "all", [])

        # === Factor in the density ===
        for mesh_name, mesh in mesh_sip.items():
            if args.pipeline == "HIT" and mesh_name == "torso":
                mesh['mass'] *= default_densities[args.pipeline][f"{mesh_name}_{body}"]
                mesh['inertia'] *= default_densities[args.pipeline][f"{mesh_name}_{body}"]
            else:
                mesh['mass'] *= default_densities[args.pipeline][mesh_name]
                mesh['inertia'] *= default_densities[args.pipeline][mesh_name]

        # === Sum up the results ===
        mass = sum([mesh['mass'] for mesh in mesh_sip.values()])
        com = sum([mesh["mass_center"] * mesh['mass'] for mesh in mesh_sip.values()]) / mass
        inertia = sum([mesh['inertia'] for mesh in mesh_sip.values()])

        # Rotate the inertia to the correct orientation - fixed, scipy.spatial.transform.Rotation.apply would only work for vectors
        inertia = orientation.T @ inertia @ orientation

        result.loc[len(result)] = [body, mass, com, inertia]
    return result


# SMPL results
#Estimated BSIP for calcn_r: mass=0.58, com=[0.01107923 0.00351048 0.08848897], inertia=[[ 1.54056035e-03  1.89716987e-04  6.24708723e-05]
# [-2.85399573e-04  1.59827246e-03  2.63101048e-04]
# [ 1.97623917e-04  1.19995109e-04  3.76568355e-04]]
#Estimated BSIP for tibia_r: mass=2.94, com=[-0.01132348 -0.15808383 -0.05412182], inertia=[[ 0.03419011  0.00036642 -0.00286787]
# [-0.00584912  0.00436177 -0.00213702]
# [ 0.00248532 -0.00265359  0.03441071]]
#Estimated BSIP for femur_r: mass=9.91, com=[-0.02159039 -0.15136532 -0.02333337], inertia=[[ 0.15755671  0.01902932 -0.00930377]
# [-0.01231569  0.04020639  0.00996473]
# [ 0.01084949 -0.00563327  0.15412098]]
#Estimated BSIP for pelvis: mass=15.57, com=[-0.00239888  0.01906724 -0.07285991], inertia=[[ 0.15745727  0.00170896 -0.0056328 ]
# [-0.00413927  0.1587745  -0.01858318]
# [ 0.0027135  -0.01794385  0.16723627]]



def estimate_segment_bsip(mesh, orientation, joint_position):
    """
    Estimate the BSIP of a single segment
    :param mesh: Mesh of the segment
    :param orientation: Orientation of the segment
    :param joint_position: Position of the joint
    :return: Densityless: Mass, CoM and Inertia of the segment
    """
    if not mesh:
        return np.array([0]), np.zeros(3)
    mass = mesh.volume
    # Rotate the com and inertia to the correct orientation

    transform = R.from_matrix(orientation)
    com = transform.inv().apply(mesh.center_mass - joint_position)

    mesh2 = mesh.copy()
    mesh2.vertices -= mesh.center_mass

    return mass, com

def get_inertia_and_update_main_body_mass(mesh_sip, target_mesh, substraction_meshes):
    """
    Calculate the inertia of the mesh
    :param mesh_sip: Dictionary of meshes
    :return: Dictionary of meshes with inertia
    """
    complete_com = mesh_sip[target_mesh]["mass_center"].copy()

    # mass_centerpute inertia wrt to the complete CoM for all submeshes
    submesh_inertia = []
    for mesh_name, mesh in mesh_sip.items():
        if mesh_name not in substraction_meshes:
            continue
        T = tf.translation_matrix(mesh["mass_center"] - complete_com)
        if not mesh["mesh"]:
            mesh['inertia'] = np.zeros((3, 3))
        else:
            mesh["inertia"] = trimesh.inertia.transform_inertia(T, mesh["mesh"].moment_inertia,
                                                                mass=mesh["mass"], parallel_axis=True)
        submesh_inertia.append(mesh["inertia"])
    
    mesh_sip[target_mesh]["inertia"] = mesh_sip[target_mesh]['mesh'].moment_inertia - sum(submesh_inertia)
    new_mass = mesh_sip[target_mesh]["mass"] - sum([mesh_sip[k]["mass"] for k in substraction_meshes])
    mesh_sip[target_mesh]["mass_center"] = (mesh_sip[target_mesh]["mass"] * complete_com - sum([mesh_sip[k]["mass"] * mesh_sip[k]["mass_center"] for k in substraction_meshes])) / new_mass
    mesh_sip[target_mesh]["mass"] = new_mass

    return mesh_sip
