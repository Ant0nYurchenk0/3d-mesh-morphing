"""
TRELLIS Mesh Diagnostic Tool
==============================

Run this on your TRELLIS-generated meshes BEFORE morphing to understand 
exactly what's wrong and how severe the issues are.

Usage:
    python diagnose_mesh.py your_mesh.glb
"""

import trimesh
import numpy as np
import sys


def diagnose(mesh_path: str):
    print(f"\n{'='*60}")
    print(f"MESH DIAGNOSTIC: {mesh_path}")
    print(f"{'='*60}\n")
    
    mesh = trimesh.load(mesh_path, force='mesh')
    
    # --- Basic stats ---
    print(f"Vertices:  {len(mesh.vertices):,}")
    print(f"Faces:     {len(mesh.faces):,}")
    print(f"Edges:     {len(mesh.edges):,}")
    
    # --- OT feasibility ---
    n = len(mesh.vertices)
    ot_matrix_gb = (n * n * 8) / 1e9
    print(f"\nOT cost matrix for two meshes this size: {ot_matrix_gb:.1f} GB")
    if ot_matrix_gb > 4:
        print(f"  ⚠ PROBLEM: Too large for dense OT. Need decimation to ~10K vertices.")
    elif ot_matrix_gb > 0.5:
        print(f"  ⚠ WARNING: Large. Will be slow. Consider decimation.")
    else:
        print(f"  ✓ OK for dense OT.")
    
    # --- Manifold check ---
    print(f"\nWatertight: {mesh.is_watertight}")
    print(f"Is volume:  {mesh.is_volume}")
    
    # --- Connected components ---
    components = mesh.split(only_watertight=False)
    print(f"\nConnected components: {len(components)}")
    if len(components) > 1:
        sizes = sorted([len(c.faces) for c in components], reverse=True)
        print(f"  ⚠ PROBLEM: Multiple components. Sizes: {sizes[:5]}{'...' if len(sizes)>5 else ''}")
        print(f"  Largest has {sizes[0]} faces ({sizes[0]/len(mesh.faces)*100:.1f}% of total)")
        print(f"  → The small components are floating debris that will corrupt MCF")
    
    # --- Degenerate faces ---
    areas = mesh.area_faces
    zero_area = np.sum(areas < 1e-10)
    tiny_area = np.sum(areas < 1e-6)
    print(f"\nZero-area faces (<1e-10):  {zero_area}")
    print(f"Tiny-area faces (<1e-6):   {tiny_area}")
    if zero_area > 0:
        print(f"  ⚠ PROBLEM: Degenerate triangles will cause NaN in cotangent weights")
    
    # --- Duplicate faces ---
    unique_faces = len(mesh.unique_faces())
    dups = len(mesh.faces) - unique_faces
    if dups > 0:
        print(f"\nDuplicate faces: {dups}")
        print(f"  ⚠ PROBLEM: Duplicate faces corrupt topology")
    
    # --- Non-manifold edges ---
    # An edge shared by >2 faces is non-manifold
    from collections import Counter
    edge_counts = Counter(map(tuple, np.sort(mesh.edges, axis=1).tolist()))
    non_manifold_edges = sum(1 for c in edge_counts.values() if c > 2)
    boundary_edges = sum(1 for c in edge_counts.values() if c == 1)
    print(f"\nBoundary edges (holes):     {boundary_edges}")
    print(f"Non-manifold edges (>2 faces): {non_manifold_edges}")
    if non_manifold_edges > 0:
        print(f"  ⚠ PROBLEM: Non-manifold edges break cotangent Laplacian and MCF")
    if boundary_edges > 0:
        print(f"  ⚠ PROBLEM: Holes mean the mesh is not closed — MCF to sphere will fail")
    
    # --- Vertex distribution ---
    centroid = mesh.centroid
    distances = np.linalg.norm(mesh.vertices - centroid, axis=1)
    print(f"\nVertex distance from centroid:")
    print(f"  Min:    {distances.min():.4f}")
    print(f"  Max:    {distances.max():.4f}")
    print(f"  Mean:   {distances.mean():.4f}")
    print(f"  Std:    {distances.std():.4f}")
    
    # Check for interior vertices (very close to centroid, likely inside the mesh)
    interior_thresh = distances.mean() * 0.3
    interior_count = np.sum(distances < interior_thresh)
    if interior_count > 0:
        print(f"\n  ⚠ PROBLEM: {interior_count} vertices suspiciously close to centroid")
        print(f"    (distance < {interior_thresh:.4f})")
        print(f"    These are likely interior vertices from voxel extraction.")
        print(f"    They will distort MCF and create phantom OT correspondences.")
    
    # --- Euler characteristic ---
    V, E, F = len(mesh.vertices), len(mesh.edges), len(mesh.faces)
    euler = V - E + F
    print(f"\nEuler characteristic (V-E+F): {euler}")
    print(f"  Expected for sphere (genus 0): 2")
    if euler != 2:
        print(f"  ⚠ PROBLEM: Genus is not 0. The mesh has {'holes' if euler < 2 else 'anomalies'}.")
        print(f"    The convex hull morphing approach explicitly requires genus 0.")
    
    # --- Normal consistency ---
    if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
        # Check for flipped normals by seeing if normals point outward
        face_centers = mesh.triangles_center
        centroid_to_face = face_centers - centroid
        dots = np.sum(mesh.face_normals * centroid_to_face, axis=1)
        flipped = np.sum(dots < 0)
        print(f"\nInward-facing normals: {flipped}/{len(mesh.faces)} ({flipped/len(mesh.faces)*100:.1f}%)")
        if flipped > len(mesh.faces) * 0.1:
            print(f"  ⚠ PROBLEM: Many normals face inward — inconsistent winding")
    
    # --- Summary ---
    problems = []
    if len(components) > 1:
        problems.append("multiple components (floating debris)")
    if not mesh.is_watertight:
        problems.append("not watertight (has holes)")
    if non_manifold_edges > 0:
        problems.append(f"{non_manifold_edges} non-manifold edges")
    if zero_area > 0:
        problems.append(f"{zero_area} degenerate faces")
    if interior_count > 0:
        problems.append(f"{interior_count} interior vertices")
    if euler != 2:
        problems.append(f"euler={euler} (not genus-0)")
    if n > 15000:
        problems.append(f"too many vertices ({n:,}) for dense OT")
    
    print(f"\n{'='*60}")
    if problems:
        print(f"FOUND {len(problems)} ISSUE(S):")
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
        print(f"\n→ Run preprocess_and_morph.py to fix these before morphing.")
    else:
        print("MESH LOOKS CLEAN — should work with morphing pipeline.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_mesh.py <mesh_file.glb>")
        sys.exit(1)
    
    for path in sys.argv[1:]:
        diagnose(path)
