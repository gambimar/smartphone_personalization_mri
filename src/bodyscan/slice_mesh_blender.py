import trimesh
import tempfile
import subprocess
import os

def slice_mesh_with_blender(target_mesh: trimesh.Trimesh, cutter_mesh: trimesh.Trimesh, operation='DIFFERENCE') -> trimesh.Trimesh:
    assert operation in {'DIFFERENCE', 'INTERSECT', 'UNION'}, "Invalid operation type"

    with tempfile.TemporaryDirectory() as tmpdir:
        # File paths
        target_path = os.path.join(tmpdir, "target.stl")
        cutter_path = os.path.join(tmpdir, "cutter.stl")
        output_path = os.path.join(tmpdir, "result.stl")

        # Export both meshes
        target_mesh.export(target_path)
        cutter_mesh.export(cutter_path)

        # Create Blender script
        import bpy

        # Clean up
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()
        
        # Import target
        bpy.ops.object.select_all(action='DESELECT')
        before = set(bpy.data.objects)
        bpy.ops.import_mesh.stl(filepath=target_path)
        after = set(bpy.data.objects)
        target_obj = (after - before).pop()
        target_obj.name = "Target"

        # Import cutter
        bpy.ops.object.select_all(action='DESELECT')
        before = set(bpy.data.objects)
        bpy.ops.import_mesh.stl(filepath=cutter_path)
        after = set(bpy.data.objects)
        cutter_obj = (after - before).pop()
        cutter_obj.name = "Cutter"

        # Apply transforms
        for obj in [target_obj, cutter_obj]:
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # Boolean operation
        bpy.ops.object.select_all(action='DESELECT')
        target_obj.select_set(True)
        bpy.context.view_layer.objects.active = target_obj
        bool_mod = target_obj.modifiers.new(name="BooleanCut", type='BOOLEAN')
        bool_mod.object = cutter_obj
        bool_mod.operation = operation
        bpy.ops.object.modifier_apply(modifier=bool_mod.name)

        # Export result
        bpy.ops.object.select_all(action='DESELECT')
        target_obj.select_set(True)
        bpy.ops.export_mesh.stl(filepath=output_path, use_selection=True)
        return trimesh.load(output_path)