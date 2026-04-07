"""
LangGraph pipeline nodes for the 3D mesh morphing pipeline.

Each node receives the full MorphingState, performs its stage of work,
and returns a partial state update dict.

Node execution order (see graph.py):

  Input: image
    setup → enhance_image (1) → image_to_base_mesh (2) → morph_image (4)
                                                              ↓
  Input: mesh                                          target_mesh (5)
    setup → render_mesh (3) ──────────────────────────────►  ↓
                                                        repair_mesh (6)
                                                              ↓
                                                       morph_meshes (7) → END

Node skip behaviour:
  - enhance_image_node:      if state.skip_enhance, copies input_image as base_image_path
  - image_to_base_mesh_node: if state.skip_base_mesh, sets base_mesh_path = None
  - render_mesh_node:        always runs when input_mesh is provided
  - repair_mesh_node:        multi-stage repair (PyMeshLab → ManifoldPlus → Instant Meshes)
  - morph_meshes_node:       placeholder (logs; no output yet)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .state import MorphingState

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Shared mesh utilities (used by nodes 7a and 7b)
# ------------------------------------------------------------------

def _normalise(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Centre and scale mesh to fit in [-1, 1]³."""
    m = m.copy()
    m.vertices -= m.bounds.mean(axis=0)
    m.vertices /= np.max(m.extents) / 2.0
    return m


def _pca_orient(m: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate mesh to align principal axes with coordinate axes (Y-up, sign-disambiguated).

    Principal axes are sorted by descending variance and assigned as:
      - Largest variance  → Y axis (up / height)
      - 2nd largest       → X axis (width)
      - Smallest          → Z axis (depth)

    Y-up is the canonical camera convention used by PyTorch3D's default renderer and
    most 3D viewers.  Aligning both base and target meshes to this frame means that
    silhouette renders from the same azimuth always see semantically equivalent views,
    preventing the optimiser from wasting steps correcting orientation.

    Sign disambiguation: for each axis, flip so the side with greater spread
    (higher 90th percentile magnitude) points in the positive direction.
    """
    pts = m.vertices
    cov = np.cov(pts.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)  # ascending order

    # Descending sort: idx[0]=largest variance, idx[1]=2nd, idx[2]=smallest
    idx = np.argsort(eigenvalues)[::-1]

    # Map to Y-up convention: X=2nd-largest, Y=largest, Z=smallest
    xyz_order = [idx[1], idx[0], idx[2]]
    R = eigenvectors[:, xyz_order].T  # rows = new basis vectors in original space

    rotated = (R @ pts.T).T

    # Sign disambiguation: flip each axis so the more-spread-out half is positive
    for i in range(3):
        col = rotated[:, i]
        if np.percentile(col, 90) < abs(np.percentile(col, 10)):
            rotated[:, i] = -col
            R[i] = -R[i]

    # Ensure proper rotation (det = +1). A det = -1 matrix is a reflection, which
    # reverses face winding and makes all normals point inward (hollow appearance).
    if np.linalg.det(R) < 0:
        rotated[:, 2] = -rotated[:, 2]

    result = m.copy()
    result.vertices = rotated
    return result


# ------------------------------------------------------------------
# Node 0: setup
# ------------------------------------------------------------------

def setup_node(state: MorphingState) -> dict:
    """
    Create a timestamped Session directory and load the morphing prompt.

    Sets: session, prompt
    """
    from shared.session import Session

    # session_base is always set by run_morphing.py to morphing_pipeline/sessions/
    session_base: Path = state["session_base"]
    session = Session(base_dir=session_base)
    log.info("[setup] session: %s", session.dir)

    prompt_file = Path(state["prompt_file"])
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    log.info("[setup] prompt loaded (%d chars)", len(prompt))

    return {"session": session, "prompt": prompt}


# ------------------------------------------------------------------
# Node 1: enhance_image  (skippable)
# ------------------------------------------------------------------

def enhance_image_node(state: MorphingState) -> dict:
    """
    Node 1 — identity regeneration via GPT (skippable).

    When skip_enhance=True, passes the raw input_image through without
    calling the OpenAI API.  When skip_enhance=False, sends the image to
    GPT with a prompt asking it to reproduce it faithfully.

    Sets: base_image_path
    """
    import os
    from src.image_generator import GPTImageGenerator

    input_image = state["input_image"]
    session = state["session"]

    if state.get("skip_enhance", False):
        log.info("[enhance_image] skipped — using raw input image")
        return {"base_image_path": input_image}

    cfg = state["cfg"]
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Copy .env.example → .env and add your OpenAI API key."
        )

    generator = GPTImageGenerator(
        api_key=api_key,
        model=cfg.image_generation.model,
        size=cfg.image_generation.size,
    )
    output_path = session.dir / "enhanced_image.png"
    try:
        generator.enhance(Path(input_image), output_path)
        log.info("[enhance_image] saved -> %s", output_path)
        return {"base_image_path": str(output_path)}
    except Exception as exc:
        log.error("[enhance_image] GPT call failed: %s — falling back to raw input", exc)
        return {"base_image_path": input_image}


# ------------------------------------------------------------------
# Node 2: image_to_base_mesh  (skippable)
# ------------------------------------------------------------------

def image_to_base_mesh_node(state: MorphingState) -> dict:
    """
    Node 2 — generate a base 3D mesh from the (possibly enhanced) image (skippable).

    When skip_base_mesh=True, skips the API call.  The base mesh is optional
    for the final morphing step — if absent, only the target mesh is generated.

    Sets: base_mesh_path
    """
    from shared.models import get_model_client

    if state.get("skip_base_mesh", False):
        log.info("[image_to_base_mesh] skipped")
        return {"base_mesh_path": None}

    cfg = state["cfg"]
    model_name = state["model_name"]
    model_cfg = cfg.models.get(model_name)
    if model_cfg is None:
        raise ValueError(
            f"Model '{model_name}' not found in config. "
            f"Available: {list(cfg.models.keys())}"
        )

    session = state["session"]
    base_image = state.get("base_image_path")
    if not base_image:
        log.warning("[image_to_base_mesh] no base image — skipping")
        return {"base_mesh_path": None}

    log.info("[image_to_base_mesh] model=%s image=%s", model_name, base_image)
    try:
        client = get_model_client(model_name, model_cfg)
        raw_path = client.reconstruct(Path(base_image))
        dest = session.dir / f"base_mesh{raw_path.suffix}"
        import shutil
        shutil.copy2(raw_path, dest)
        log.info("[image_to_base_mesh] saved -> %s", dest)
        return {"base_mesh_path": str(dest)}
    except Exception as exc:
        log.error("[image_to_base_mesh] failed: %s", exc)
        return {"base_mesh_path": None}


# ------------------------------------------------------------------
# Node 3: render_mesh  (runs only when input is a mesh)
# ------------------------------------------------------------------

def render_mesh_node(state: MorphingState) -> dict:
    """
    Node 3 — render input mesh to a 2D image (runs when input_mesh is provided).

    Produces the base_image_path that feeds into morph_image_node (node 4).
    Also sets base_mesh_path to the input_mesh so it can be used in node 7.

    Sets: base_image_path, base_mesh_path, render_image_path
    """
    import trimesh
    from shared.renderer import Renderer
    from PIL import Image

    cfg = state["cfg"]
    input_mesh = state["input_mesh"]
    session = state["session"]

    log.info("[render_mesh] rendering: %s", input_mesh)
    try:
        mesh = trimesh.load(str(input_mesh), force="mesh", process=False)
        renderer = Renderer(cfg.render)
        img = renderer.render(mesh)

        render_path = session.dir / "render.png"
        Image.fromarray(img).save(render_path)
        log.info("[render_mesh] saved render -> %s", render_path)

        return {
            "render_image_path": str(render_path),
            "base_image_path": str(render_path),
            "base_mesh_path": input_mesh,
        }
    except Exception as exc:
        log.error("[render_mesh] failed: %s", exc)
        raise


# ------------------------------------------------------------------
# Node 4: morph_image
# ------------------------------------------------------------------

def morph_image_node(state: MorphingState) -> dict:
    """
    Node 4 — generate morphed image from base image using the user's prompt.

    Calls the OpenAI images.edit() API with the base_image_path and prompt.

    Sets: morphed_image_path
    """
    import os
    from src.image_generator import GPTImageGenerator

    cfg = state["cfg"]
    session = state["session"]
    base_image = state.get("base_image_path")
    prompt = state["prompt"]

    if not base_image:
        raise RuntimeError("[morph_image] no base_image_path in state")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY environment variable is not set. "
            "Copy .env.example → .env and add your OpenAI API key."
        )

    generator = GPTImageGenerator(
        api_key=api_key,
        model=cfg.image_generation.model,
        size=cfg.image_generation.size,
    )
    output_path = session.dir / "morphed_image.png"
    log.info("[morph_image] generating morphed image from: %s", base_image)
    generator.morph(Path(base_image), prompt, output_path)
    log.info("[morph_image] saved -> %s", output_path)

    return {"morphed_image_path": str(output_path)}


# ------------------------------------------------------------------
# Node 5: target_mesh
# ------------------------------------------------------------------

def target_mesh_node(state: MorphingState) -> dict:
    """
    Node 5 — reconstruct 3D target mesh from the morphed image.

    Uses the same ImageTo3DClient infrastructure as benchmark_pipeline.

    Sets: target_mesh_path
    """
    from shared.models import get_model_client

    cfg = state["cfg"]
    model_name = state["model_name"]
    model_cfg = cfg.models.get(model_name)
    if model_cfg is None:
        raise ValueError(
            f"Model '{model_name}' not found in config. "
            f"Available: {list(cfg.models.keys())}"
        )

    session = state["session"]
    morphed_image = state.get("morphed_image_path")
    if not morphed_image:
        raise RuntimeError("[target_mesh] no morphed_image_path in state")

    log.info("[target_mesh] model=%s image=%s", model_name, morphed_image)
    client = get_model_client(model_name, model_cfg)
    raw_path = client.reconstruct(Path(morphed_image))

    dest = session.dir / f"target_mesh{raw_path.suffix}"
    import shutil
    shutil.copy2(raw_path, dest)
    log.info("[target_mesh] saved -> %s", dest)

    return {"target_mesh_path": str(dest)}


# ------------------------------------------------------------------
# Node 6: repair_mesh
# ------------------------------------------------------------------

def repair_mesh_node(state: MorphingState) -> dict:
    """
    Node 6 — multi-stage mesh repair.

    Stage 1 (always): PyMeshLab — remove fragments, fix non-manifold,
        close holes, remove degenerate faces, decimate, re-orient normals.
    Stage 2 (if cleanup_cost > 0 after stage 1): ManifoldPlus CLI.
    Stage 3 (if remesh=True): Instant Meshes CLI.

    Sets: repaired_mesh_path, repair_report
    """
    import json
    from src.mesh_repairer import MeshRepairer

    target_mesh = state.get("target_mesh_path")
    if not target_mesh:
        raise RuntimeError("[repair_mesh] no target_mesh_path in state")

    cfg = state["cfg"]
    session = state["session"]
    output_path = session.dir / "repaired_mesh.glb"

    repairer = MeshRepairer(cfg.repair)
    report = repairer.repair(
        input_path=Path(target_mesh),
        output_path=output_path,
        session_dir=session.dir,
        remesh=state.get("remesh", False),
    )

    report_path = session.dir / "repair_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    log.info(
        "[repair_mesh] cost %d → %d  stages=%s  faces=%s",
        report["cost_before"], report["cost_final"],
        report["stages_run"], report["face_count_final"],
    )
    return {"repaired_mesh_path": str(output_path), "repair_report": report}


# ------------------------------------------------------------------
# Node 7a: morph_meshes_sdf
# ------------------------------------------------------------------

def morph_meshes_sdf_node(state: MorphingState) -> dict:
    """
    Node 7 (sdf) — morph base mesh → target mesh via SDF interpolation.

    Algorithm:
      1. Load both meshes and normalise each to [-1, 1]³.
      2. Evaluate the signed distance field (SDF) at every point of a shared
         grid_resolution³ voxel grid using trimesh proximity + face-normal sign.
      3. Linearly interpolate SDF_A and SDF_B at n_frames steps (t = 0 → 1).
      4. Extract an isosurface at level=0 for each frame via marching cubes
         (scikit-image, included with trimesh[easy]).
      5. Save each frame as a GLB file in session_dir/transition/.

    Requires: base_mesh_path AND (repaired_mesh_path or target_mesh_path) in state.
    Sets: transition_path
    """
    import json

    import trimesh.proximity
    from skimage.measure import marching_cubes

    base_mesh_path = state.get("base_mesh_path")
    target_mesh_path = state.get("repaired_mesh_path") or state.get("target_mesh_path")

    if not base_mesh_path or not target_mesh_path:
        raise RuntimeError(
            "[morph_meshes_sdf] need both base_mesh_path and "
            "repaired_mesh_path (or target_mesh_path) in state"
        )

    cfg = state["cfg"]
    session = state["session"]
    grid_res: int = cfg.morph.grid_resolution
    n_frames: int = cfg.morph.n_frames

    log.info("[morph_meshes_sdf] base=%s", base_mesh_path)
    log.info("[morph_meshes_sdf] target=%s", target_mesh_path)
    log.info("[morph_meshes_sdf] grid=%d³  frames=%d", grid_res, n_frames)

    # ── Load, normalise, PCA-align ───────────────────────────────────
    mesh_a = trimesh.load(str(base_mesh_path), force="mesh", process=True)
    mesh_b = trimesh.load(str(target_mesh_path), force="mesh", process=True)

    mesh_a = _pca_orient(_normalise(mesh_a))
    mesh_b = _pca_orient(_normalise(mesh_b))

    # ── Decimate for SDF ────────────────────────────────────────────
    # The SDF is sampled on a grid_res³ grid so mesh detail beyond ~5k faces
    # is invisible. Decimating large meshes here keeps SDF computation fast
    # regardless of the reconstruction model's output resolution.
    _SDF_MAX_FACES = 5_000

    def _decimate(m: trimesh.Trimesh) -> trimesh.Trimesh:
        if len(m.faces) > _SDF_MAX_FACES:
            m = m.simplify_quadric_decimation(face_count=_SDF_MAX_FACES)
            log.debug("[morph_meshes_sdf] decimated to %d faces for SDF", len(m.faces))
        return m

    mesh_a_sdf = _decimate(mesh_a)
    mesh_b_sdf = _decimate(mesh_b)
    log.info(
        "[morph_meshes_sdf] SDF meshes — A: %d faces  B: %d faces",
        len(mesh_a_sdf.faces), len(mesh_b_sdf.faces),
    )

    # ── Build voxel grid ────────────────────────────────────────────
    # Pad slightly beyond ±1 so the surface is captured even at the boundary.
    pad = 1.2
    lin = np.linspace(-pad, pad, grid_res)
    xx, yy, zz = np.meshgrid(lin, lin, lin, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])  # (N, 3)

    # ── Compute SDFs ────────────────────────────────────────────────
    def _sdf(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
        """Signed distance: negative inside mesh, positive outside.

        Sign is determined by the dot product of (query - closest) with the
        closest face normal — robust for non-watertight meshes.
        """
        closest, dists, tri_ids = trimesh.proximity.closest_point(mesh, points)
        face_normals = mesh.face_normals[tri_ids]
        direction = points - closest
        dot = np.einsum("ij,ij->i", direction, face_normals)
        sign = np.where(dot >= 0, 1.0, -1.0)
        return sign * dists

    log.info("[morph_meshes_sdf] computing SDF A (%d pts)...", len(pts))
    sdf_a = _sdf(mesh_a_sdf, pts).reshape(grid_res, grid_res, grid_res)

    log.info("[morph_meshes_sdf] computing SDF B...")
    sdf_b = _sdf(mesh_b_sdf, pts).reshape(grid_res, grid_res, grid_res)

    # ── Extract frames ──────────────────────────────────────────────
    out_dir = session.dir / "transition"
    out_dir.mkdir(exist_ok=True)

    spacing = (2.0 * pad) / (grid_res - 1)
    origin = np.array([-pad, -pad, -pad])
    ts = np.linspace(0.0, 1.0, n_frames)
    frame_paths: list[str] = []

    log.info("[morph_meshes_sdf] extracting %d frames...", n_frames)
    for i, t in enumerate(ts):
        sdf_t = (1.0 - t) * sdf_a + t * sdf_b
        try:
            verts, faces, *_ = marching_cubes(sdf_t, level=0.0, spacing=(spacing,) * 3)
            verts += origin  # shift from grid-index space to world space
            m = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            fname = out_dir / f"frame_{i:04d}.glb"
            m.export(str(fname))
            frame_paths.append(str(fname))
        except Exception as exc:
            log.warning("[morph_meshes_sdf] frame %03d (t=%.2f) skipped: %s", i, t, exc)

    info = {
        "method": "sdf",
        "grid_resolution": grid_res,
        "n_frames": n_frames,
        "frames_saved": len(frame_paths),
    }
    (out_dir / "morph_info.json").write_text(json.dumps(info, indent=2))

    log.info(
        "[morph_meshes_sdf] done — %d/%d frames saved to %s",
        len(frame_paths), n_frames, out_dir,
    )
    return {"transition_path": str(out_dir), "exit_code": 0}


# ------------------------------------------------------------------
# Node 7b-1: diff_optimize — Phase 1: gradient-based deformation
# ------------------------------------------------------------------

def diff_optimize_node(state: MorphingState) -> dict:
    """
    Phase 1 of the differential rendering morph.

    Loads base + target meshes, normalises and PCA-aligns both to a Y-up
    canonical frame, adaptively subdivides the base mesh, then optimises
    per-vertex offsets using Chamfer + silhouette losses with coarse-to-fine
    scheduling.

    Saves to session dir for inspection:
      diff_base_mesh.glb     — subdivided, aligned base mesh
      diff_target_mesh.glb   — aligned target mesh
      diff_deformed_raw.glb  — base mesh + raw optimised offsets
      diff_offsets.npy       — raw per-vertex offsets (float32, shape [N, 3])
    """
    import json
    import torch
    from pytorch3d.loss import (
        chamfer_distance, mesh_edge_loss,
        mesh_laplacian_smoothing, mesh_normal_consistency,
    )
    from pytorch3d.ops import sample_points_from_meshes
    from pytorch3d.renderer import (
        BlendParams, FoVPerspectiveCameras, MeshRasterizer,
        MeshRenderer, RasterizationSettings, SoftSilhouetteShader,
        look_at_view_transform,
    )
    from pytorch3d.structures import Meshes

    base_mesh_path = state.get("base_mesh_path")
    target_mesh_path = (
        state.get("repaired_mesh_path") or state.get("target_mesh_path")
    )
    if not base_mesh_path or not target_mesh_path:
        raise RuntimeError("[diff_optimize] need both base_mesh_path and target_mesh_path")

    cfg = state["cfg"]
    session = state["session"]
    dc = cfg.diff_rend
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("[diff_optimize] device=%s  steps=%d", device, dc.n_steps)

    # ── Load, normalise, align ───────────────────────────────────────
    mesh_a = _pca_orient(_normalise(
        trimesh.load(str(base_mesh_path), force="mesh", process=True)
    ))
    mesh_b = _pca_orient(_normalise(
        trimesh.load(str(target_mesh_path), force="mesh", process=True)
    ))

    # Adaptive subdivision — need enough vertices for expressive deformation
    while len(mesh_a.vertices) < 2000:
        mesh_a = mesh_a.subdivide()

    log.info(
        "[diff_optimize] base: %d verts %d faces | target: %d verts %d faces",
        len(mesh_a.vertices), len(mesh_a.faces),
        len(mesh_b.vertices), len(mesh_b.faces),
    )

    # ── Save aligned meshes for downstream nodes + inspection ────────
    base_mesh_out = session.dir / "diff_base_mesh.glb"
    target_mesh_out = session.dir / "diff_target_mesh.glb"
    mesh_a.export(str(base_mesh_out))
    mesh_b.export(str(target_mesh_out))
    log.info("[diff_optimize] saved diff_base_mesh.glb and diff_target_mesh.glb")

    def _to_pt3d(m):
        v = torch.tensor(m.vertices, dtype=torch.float32, device=device)
        f = torch.tensor(m.faces, dtype=torch.int64, device=device)
        return Meshes(verts=[v], faces=[f])

    src_mesh = _to_pt3d(mesh_a)
    tgt_mesh = _to_pt3d(mesh_b)

    # Pre-sample dense target point cloud once to reduce per-step noise
    with torch.no_grad():
        tgt_points_dense = sample_points_from_meshes(tgt_mesh, 50_000).detach()

    # ── Optimizer ────────────────────────────────────────────────────
    deform_verts = torch.zeros_like(src_mesh.verts_packed(), requires_grad=True)
    optimizer = torch.optim.SGD(
        [deform_verts], lr=dc.lr * 5, momentum=0.9, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=dc.n_steps, eta_min=1e-5
    )

    # ── Silhouette renderer ──────────────────────────────────────────
    sigma = 5e-3  # ~1.8px blur at 256px — wide enough for large deformations
    raster_settings = RasterizationSettings(
        image_size=dc.image_size,
        blur_radius=float(np.log(1.0 / sigma - 1.0) * sigma),
        faces_per_pixel=50,
    )
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=sigma, gamma=1e-4)),
    )

    # ── Camera rig — 3 elevation rings ───────────────────────────────
    elevs = [-15.0, 5.0, 30.0]
    n_per_ring = max(2, dc.n_views // len(elevs))
    cameras_RT = []
    for elev in elevs:
        for azim in np.linspace(0, 360, n_per_ring, endpoint=False):
            R, T = look_at_view_transform(dist=2.7, elev=float(elev), azim=float(azim))
            cameras_RT.append((R.to(device), T.to(device)))

    target_sils = []
    with torch.no_grad():
        for R, T in cameras_RT:
            cam = FoVPerspectiveCameras(device=device, R=R, T=T)
            sil = silhouette_renderer(tgt_mesh, cameras=cam)[..., 3]
            target_sils.append(sil.detach())

    # ── Training loop ────────────────────────────────────────────────
    log.info("[diff_optimize] Phase 1: optimizing deformation (%d steps)", dc.n_steps)

    for step in range(dc.n_steps):
        progress = step / max(1, dc.n_steps - 1)
        deformed = src_mesh.offset_verts(deform_verts)

        n_pts = int(1000 + progress * 9000)
        src_pts = sample_points_from_meshes(deformed, n_pts)
        idx = torch.randperm(tgt_points_dense.shape[1])[:n_pts]
        tgt_pts = tgt_points_dense[:, idx, :]
        loss_chamfer, _ = chamfer_distance(src_pts, tgt_pts)

        reg_decay = max(0.1, 1.0 - progress * 0.8)
        loss_edge      = mesh_edge_loss(deformed)
        loss_laplacian = mesh_laplacian_smoothing(deformed, method="uniform")
        loss_normal    = mesh_normal_consistency(deformed)

        n_sil_views = min(6, len(cameras_RT))
        view_idx = np.random.choice(len(cameras_RT), n_sil_views, replace=False)
        loss_sil = torch.zeros(1, device=device)
        for i in view_idx:
            R, T = cameras_RT[i]
            cam = FoVPerspectiveCameras(device=device, R=R, T=T)
            pred_sil = silhouette_renderer(deformed, cameras=cam)[..., 3]
            loss_sil += ((pred_sil - target_sils[i]) ** 2).mean()
        loss_sil /= n_sil_views

        loss = (
            dc.w_chamfer    * loss_chamfer
            + dc.w_silhouette * loss_sil
            + dc.w_laplacian  * loss_laplacian * reg_decay
            + dc.w_normal     * loss_normal    * reg_decay
            + dc.w_edge       * loss_edge      * reg_decay
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([deform_verts], max_norm=0.5)
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            log.info(
                "[diff_optimize] step %4d/%d  loss=%.5f"
                "  chamfer=%.5f  sil=%.5f  lap=%.5f  reg_decay=%.2f",
                step, dc.n_steps, loss.item(),
                loss_chamfer.item(), loss_sil.item(),
                loss_laplacian.item(), reg_decay,
            )

    # ── Save raw offsets and deformed mesh for inspection ────────────
    final_offsets = deform_verts.detach().clone()
    offsets_np = final_offsets.cpu().numpy()

    offsets_out = session.dir / "diff_offsets.npy"
    np.save(str(offsets_out), offsets_np)

    v_raw = (src_mesh.verts_packed() + final_offsets).cpu().numpy()
    f_raw = src_mesh.faces_packed().cpu().numpy()
    raw_mesh = trimesh.Trimesh(vertices=v_raw, faces=f_raw, process=False)
    raw_mesh.fix_normals()
    deformed_raw_out = session.dir / "diff_deformed_raw.glb"
    raw_mesh.export(str(deformed_raw_out))

    log.info(
        "[diff_optimize] Phase 1 complete. Offset magnitude: mean=%.4f max=%.4f. "
        "Saved diff_deformed_raw.glb and diff_offsets.npy",
        final_offsets.norm(dim=1).mean().item(),
        final_offsets.norm(dim=1).max().item(),
    )

    return {
        "diff_base_mesh_path":   str(base_mesh_out),
        "diff_target_mesh_path": str(target_mesh_out),
        "diff_offsets_path":     str(offsets_out),
        "diff_deformed_mesh_path": str(deformed_raw_out),
    }


# ------------------------------------------------------------------
# Node 7b-2: diff_refine — Phase 1.5: smooth + project offsets
# ------------------------------------------------------------------

def diff_refine_node(state: MorphingState) -> dict:
    """
    Phase 1.5 of the differential rendering morph.

    Takes the raw per-vertex offsets from diff_optimize_node and cleans them
    up in three steps:
      1. Laplacian smooth the offset field (kills high-freq noise / ripples)
      2. Project each deformed vertex onto the closest point on the target surface
      3. Blend: use projected positions for nearby vertices, smoothed positions
         where the topology gap is large (avoids tears from forced projection)

    Saves to session dir for inspection:
      diff_deformed_refined.glb  — deformed mesh after refinement
      diff_offsets_refined.npy   — refined per-vertex offsets
    """
    from collections import defaultdict
    import torch
    from scipy.sparse import lil_matrix, eye as speye

    base_mesh_path   = state.get("diff_base_mesh_path")
    target_mesh_path = state.get("diff_target_mesh_path")
    offsets_path     = state.get("diff_offsets_path")
    if not base_mesh_path or not target_mesh_path or not offsets_path:
        raise RuntimeError(
            "[diff_refine] missing diff_base_mesh_path / diff_target_mesh_path / diff_offsets_path"
        )

    session = state["session"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("[diff_refine] Phase 1.5: refining offsets")

    mesh_a  = trimesh.load(str(base_mesh_path),   force="mesh", process=False)
    mesh_b  = trimesh.load(str(target_mesh_path), force="mesh", process=False)
    offsets_np = np.load(str(offsets_path))        # (N, 3) float32

    n_verts   = offsets_np.shape[0]
    faces_np  = mesh_a.faces

    # ── Step 1: Sparse Laplacian smoothing of the offset field ──────
    # Build adjacency from face list, then construct the blend matrix
    # ((1-λ)I + λL) and apply it n_iters times.
    adjacency = defaultdict(set)
    for f in faces_np:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[f[i]].add(f[j])

    n_smooth_iters = 15
    smooth_lambda  = 0.5

    L = lil_matrix((n_verts, n_verts), dtype=np.float32)
    for v_idx in range(n_verts):
        neighbors = list(adjacency[v_idx])
        n_nb = len(neighbors)
        if n_nb == 0:
            L[v_idx, v_idx] = 1.0
            continue
        for nb in neighbors:
            L[v_idx, nb] = 1.0 / n_nb
        L[v_idx, v_idx] = 0.0

    L = L.tocsr()
    I = speye(n_verts, format="csr")
    blend_matrix = (1 - smooth_lambda) * I + smooth_lambda * L

    smoothed_np = offsets_np.copy()
    for _ in range(n_smooth_iters):
        smoothed_np = blend_matrix @ smoothed_np

    log.info("[diff_refine] Laplacian smoothing done (%d iters)", n_smooth_iters)

    # ── Step 2: Project deformed vertices onto target surface ────────
    base_verts_np  = mesh_a.vertices.astype(np.float32)
    deformed_smoothed_np = base_verts_np + smoothed_np

    closest_points, distances, _ = trimesh.proximity.closest_point(
        mesh_b, deformed_smoothed_np
    )
    log.info(
        "[diff_refine] projection: median dist=%.5f  max dist=%.5f",
        float(np.median(distances)), float(distances.max()),
    )

    # ── Step 3: Selective blend ──────────────────────────────────────
    # Vertices within 2× median distance trust the projected position.
    # Vertices farther away (topology mismatch) keep the smoothed position.
    median_dist     = float(np.median(distances))
    blend_threshold = median_dist * 2.0

    blend_weight = np.clip(
        1.0 - (distances / (blend_threshold + 1e-8)), 0.0, 1.0
    )[:, None]  # (N, 1)

    refined_positions = (
        blend_weight       * closest_points
        + (1 - blend_weight) * deformed_smoothed_np
    )
    refined_offsets_np = (refined_positions - base_verts_np).astype(np.float32)

    n_projected = int((blend_weight.squeeze() > 0.5).sum())
    log.info(
        "[diff_refine] %d/%d verts snapped to target, %d kept from smoothing",
        n_projected, n_verts, n_verts - n_projected,
    )

    # ── Save refined mesh and offsets for inspection + Phase 2 ──────
    refined_offsets_out = session.dir / "diff_offsets_refined.npy"
    np.save(str(refined_offsets_out), refined_offsets_np)

    refined_mesh = trimesh.Trimesh(
        vertices=refined_positions.astype(np.float32),
        faces=faces_np,
        process=False,
    )
    refined_mesh.fix_normals()
    refined_mesh_out = session.dir / "diff_deformed_refined.glb"
    refined_mesh.export(str(refined_mesh_out))

    log.info(
        "[diff_refine] Phase 1.5 complete. Saved diff_deformed_refined.glb "
        "and diff_offsets_refined.npy"
    )

    return {
        "diff_offsets_path":    str(refined_offsets_out),
        "diff_refined_mesh_path": str(refined_mesh_out),
    }


# ------------------------------------------------------------------
# Node 7b-3: diff_interpolate — Phase 2: generate animation frames
# ------------------------------------------------------------------

def diff_interpolate_node(state: MorphingState) -> dict:
    """
    Phase 2 of the differential rendering morph.

    Loads the (refined) per-vertex offsets and interpolates from zero to the
    full offset using a Hermite smoothstep curve, exporting one GLB per frame.
    The interpolation is done over the offsets (not positions) so frame 0 is
    always the unmodified base mesh and frame N-1 is the fully deformed target.
    """
    import json
    import torch

    base_mesh_path = state.get("diff_base_mesh_path")
    offsets_path   = state.get("diff_offsets_path")
    if not base_mesh_path or not offsets_path:
        raise RuntimeError(
            "[diff_interpolate] missing diff_base_mesh_path / diff_offsets_path"
        )

    cfg     = state["cfg"]
    session = state["session"]
    dc      = cfg.diff_rend
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mesh_a     = trimesh.load(str(base_mesh_path), force="mesh", process=False)
    offsets_np = np.load(str(offsets_path))

    verts_t   = torch.tensor(mesh_a.vertices, dtype=torch.float32, device=device)
    faces_np  = mesh_a.faces
    offsets_t = torch.tensor(offsets_np, dtype=torch.float32, device=device)

    out_dir = session.dir / "transition"
    out_dir.mkdir(exist_ok=True)
    frame_paths: list[str] = []

    log.info("[diff_interpolate] Phase 2: generating %d morph frames", dc.n_frames)

    for i in range(dc.n_frames):
        t = i / max(1, dc.n_frames - 1)
        # Hermite smoothstep: ease-in / ease-out
        t_smooth = t * t * (3.0 - 2.0 * t)

        with torch.no_grad():
            deformed_verts = (verts_t + offsets_t * t_smooth).cpu().numpy()

        frame = trimesh.Trimesh(
            vertices=deformed_verts, faces=faces_np, process=False
        )
        frame.fix_normals()
        fname = out_dir / f"frame_{i:04d}.glb"
        frame.export(str(fname))
        frame_paths.append(str(fname))

    info = {
        "method": "differential_three_phase",
        "n_frames": len(frame_paths),
        "device": str(device),
        "offsets_source": str(offsets_path),
    }
    (out_dir / "morph_info.json").write_text(json.dumps(info, indent=2))
    log.info("[diff_interpolate] done — %d frames in %s", len(frame_paths), out_dir)

    return {"transition_path": str(out_dir), "exit_code": 0}