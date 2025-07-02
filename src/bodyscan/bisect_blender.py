import bpy
import bmesh
import tempfile
import trimesh
import os

def bisect_and_repair_mesh(tri_mesh: trimesh.Trimesh, plane_normal, plane_origin, blender_object=None) -> trimesh.Trimesh:
    if blender_object is None:
        # Save input trimesh to a temporary file (e.g., STL format)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_in:
            tri_mesh.export(tmp_in.name)
            tmp_in_path = tmp_in.name

        # Clear existing mesh data in Blender
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)

        # Import the mesh into Blender
        bpy.ops.import_mesh.stl(filepath=tmp_in_path)
        obj = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = obj

    else: 
        # Use the provided Blender object
        obj = blender_object
        bpy.context.view_layer.objects.active = obj

    # Enter edit mode and get bmesh
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action='SELECT')
    
    mesh = bmesh.from_edit_mesh(obj.data)
    mesh.faces.ensure_lookup_table()
    mesh.verts.ensure_lookup_table()

    # Perform bisect
    geom = mesh.verts[:] + mesh.edges[:] + mesh.faces[:]
    bmesh.ops.bisect_plane(
        mesh,
        geom=geom,
        plane_co=plane_origin,
        plane_no=plane_normal,
        use_snap_center=False,
        clear_inner=False,
        clear_outer=True,
    )

    # Cap the cut
    bmesh.ops.holes_fill(mesh, edges=mesh.edges, sides=0)
    bmesh.update_edit_mesh(obj.data)

    # Exit edit mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Export the result
    with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as tmp_out:
        tmp_out_path = tmp_out.name
        bpy.ops.export_mesh.stl(filepath=tmp_out_path, use_selection=True)

    # Load back into Trimesh
    repaired_mesh = trimesh.load(tmp_out_path)
    if type(repaired_mesh) == trimesh.Scene:
        repaired_mesh = repaired_mesh.dump()
        repaired_mesh = trimesh.util.concatenate(repaired_mesh)

    # Cleanup temp files
    if blender_object is None:
        # Only remove the temp files if we created them
        os.remove(tmp_in_path)
    os.remove(tmp_out_path)

    # Return the blender object and the repaired mesh
    return repaired_mesh, obj
