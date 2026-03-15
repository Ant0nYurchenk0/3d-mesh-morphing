"""
Multi-stage mesh repair pipeline for morphing preparation.

Repair stages
─────────────
Stage 1 — PyMeshLab (always runs)
  Removes floating fragments, fixes non-manifold geometry, closes holes,
  removes degenerate faces, decimates to target face count, re-orients normals.

Stage 2 — ManifoldPlus (runs if cleanup_cost > 0 after stage 1)
  External CLI tool that re-wraps the surface into a watertight manifold.
  More aggressive than PyMeshLab; handles severe non-manifold geometry.
  Requires: `manifoldplus` binary on PATH or configured path.
  GitHub: https://github.com/hjwdzh/ManifoldPlus

Stage 3 — Instant Meshes (optional, runs only if remesh=True)
  External CLI tool that produces clean uniform triangle topology.
  Useful for vertex-level morphing where topology regularity matters.
  Requires: `InstantMeshes` binary on PATH or configured path.
  GitHub: https://github.com/wjakob/instant-meshes

After each stage that runs, a quality check (cleanup_cost) is logged so the
operator can see whether repair actually improved the mesh.

Public API:
    MeshRepairer(cfg: RepairConfig).repair(input_path, output_path, session_dir, remesh) -> dict
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import trimesh

from shared.mesh_quality import check_mesh_file, cleanup_cost

log = logging.getLogger(__name__)


class MeshRepairer:
    """
    Orchestrates the multi-stage mesh repair pipeline.

    Parameters
    ----------
    cfg : RepairConfig
        Typed repair settings from MorphingConfig.repair.
    """

    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def repair(
        self,
        input_path: Path,
        output_path: Path,
        session_dir: Path,
        remesh: bool = False,
    ) -> dict:
        """
        Run the full repair pipeline on *input_path* and write result to *output_path*.

        Intermediate files (repair_stage1.glb, repair_stage2.glb, repair_stage3.glb)
        are preserved in *session_dir* for inspection.

        Returns a report dict with:
            cost_before         — cleanup_cost before any repair
            cost_after_stage1   — cleanup_cost after PyMeshLab (or None if not run)
            cost_after_stage2   — cleanup_cost after ManifoldPlus (or None if not run)
            cost_after_stage3   — cleanup_cost after Instant Meshes (or None if not run)
            cost_final          — cleanup_cost of the final mesh written to output_path
            stages_run          — list of stage names that executed
            face_count_final    — face count of the final mesh

        Stage 2 regression guard: if ManifoldPlus raises cost_after_stage2 above
        cost_after_stage1, the stage1 result is kept and a warning is logged.
        The cost_after_stage2 field still records the regressed value for visibility.
        """
        report: dict[str, Any] = {
            "cost_before": check_mesh_file(input_path),
            "cost_after_stage1": None,
            "cost_after_stage2": None,
            "cost_after_stage3": None,
            "cost_final": None,
            "stages_run": [],
            "face_count_final": None,
        }
        log.info(
            "[repair] starting — input=%s  cost_before=%d",
            input_path.name, report["cost_before"],
        )

        current = input_path

        # ── Stage 1: PyMeshLab ────────────────────────────────────────
        stage1_out = session_dir / "repair_stage1.glb"
        try:
            self._stage1_pymeshlab(current, stage1_out)
            cost1 = check_mesh_file(stage1_out)
            report["cost_after_stage1"] = cost1
            report["stages_run"].append("pymeshlab")
            current = stage1_out
            log.info("[repair] stage1 (PyMeshLab) done — cost=%d", cost1)
        except Exception as exc:
            log.error("[repair] stage1 (PyMeshLab) failed: %s — skipping", exc)

        # ── Stage 2: ManifoldPlus (only if issues remain) ─────────────
        cost_after_s1 = report.get("cost_after_stage1") or report["cost_before"]
        best_before_s2 = current   # fallback if ManifoldPlus regresses
        if cost_after_s1 > 0:
            stage2_out = session_dir / "repair_stage2.glb"
            try:
                self._stage2_manifoldplus(current, stage2_out)
                cost2 = check_mesh_file(stage2_out)
                report["cost_after_stage2"] = cost2
                report["stages_run"].append("manifoldplus")
                if cost2 <= cost_after_s1:
                    current = stage2_out
                    log.info("[repair] stage2 (ManifoldPlus) done — cost=%d", cost2)
                else:
                    log.warning(
                        "[repair] stage2 (ManifoldPlus) regressed cost %d → %d "
                        "— reverting to stage1 result",
                        cost_after_s1, cost2,
                    )
                    current = best_before_s2
            except FileNotFoundError:
                log.warning(
                    "[repair] stage2 skipped — 'manifoldplus' binary not found "
                    "(install from https://github.com/hjwdzh/ManifoldPlus)"
                )
            except Exception as exc:
                log.warning("[repair] stage2 (ManifoldPlus) failed: %s — skipping", exc)

        # ── Stage 3: Instant Meshes (optional) ───────────────────────
        if remesh:
            stage3_out = session_dir / "repair_stage3.glb"
            try:
                self._stage3_instant_meshes(current, stage3_out)
                cost3 = check_mesh_file(stage3_out)
                report["cost_after_stage3"] = cost3
                report["stages_run"].append("instant_meshes")
                current = stage3_out
                log.info("[repair] stage3 (Instant Meshes) done — cost=%d", cost3)
            except FileNotFoundError:
                log.warning(
                    "[repair] stage3 skipped — 'InstantMeshes' binary not found "
                    "(install from https://github.com/wjakob/instant-meshes)"
                )
            except Exception as exc:
                log.warning("[repair] stage3 (Instant Meshes) failed: %s — skipping", exc)

        # ── Copy best result to output_path ───────────────────────────
        shutil.copy2(current, output_path)

        # Final quality check
        report["cost_final"] = check_mesh_file(output_path)
        report["face_count_final"] = _face_count(output_path)

        log.info(
            "[repair] complete — stages=%s  cost_before=%d  cost_final=%d  faces=%s",
            report["stages_run"],
            report["cost_before"],
            report["cost_final"],
            report["face_count_final"],
        )
        return report

    # ------------------------------------------------------------------
    # Stage 1 — PyMeshLab
    # ------------------------------------------------------------------

    def _stage1_pymeshlab(self, input_path: Path, output_path: Path) -> None:
        """
        PyMeshLab-based repair following the reference workflow:
          1. Remove floating fragments (< min_component_diameter_pct % of bbox diagonal)
          2. Fix non-manifold edges
          3. Fix non-manifold vertices
          4. Close holes (≤ max_hole_size boundary edges)
          5. Remove duplicate + null faces
          6. Decimate to target_faces (if over budget)
          7. Re-orient normals consistently
        """
        try:
            import pymeshlab  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "pymeshlab is not installed. Run: uv sync  (or pip install pymeshlab)"
            ) from exc

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(input_path))
        cfg = self._cfg

        # 1. Remove floating fragments
        try:
            ms.meshing_remove_connected_component_by_diameter(
                mincomponentdiag=pymeshlab.PercentageValue(cfg.min_component_diameter_pct)
            )
            log.debug("[repair/s1] removed floating fragments")
        except Exception as exc:
            log.debug("[repair/s1] remove_connected_component failed (non-fatal): %s", exc)

        # 2-3. Fix non-manifold geometry
        try:
            ms.meshing_repair_non_manifold_edges()
            log.debug("[repair/s1] repaired non-manifold edges")
        except Exception as exc:
            log.debug("[repair/s1] repair_non_manifold_edges failed (non-fatal): %s", exc)

        try:
            ms.meshing_repair_non_manifold_vertices()
            log.debug("[repair/s1] repaired non-manifold vertices")
        except Exception as exc:
            log.debug("[repair/s1] repair_non_manifold_vertices failed (non-fatal): %s", exc)

        # 4. Close holes
        try:
            ms.meshing_close_holes(maxholesize=cfg.max_hole_size)
            log.debug("[repair/s1] closed holes (max_hole_size=%d)", cfg.max_hole_size)
        except Exception as exc:
            log.debug("[repair/s1] close_holes failed (non-fatal): %s", exc)

        # 5. Remove degenerate faces
        try:
            ms.meshing_remove_duplicate_faces()
            log.debug("[repair/s1] removed duplicate faces")
        except Exception as exc:
            log.debug("[repair/s1] remove_duplicate_faces failed (non-fatal): %s", exc)

        try:
            ms.meshing_remove_null_faces()
            log.debug("[repair/s1] removed null faces")
        except Exception as exc:
            log.debug("[repair/s1] remove_null_faces failed (non-fatal): %s", exc)

        # 6. Decimate to target face count
        try:
            current_faces = ms.current_mesh().face_number()
            log.debug("[repair/s1] face count before decimation: %d", current_faces)
            if current_faces > cfg.target_faces:
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=cfg.target_faces,
                    preservenormal=True,
                    preservetopology=True,
                )
                log.debug(
                    "[repair/s1] decimated %d → %d faces",
                    current_faces, ms.current_mesh().face_number(),
                )
        except Exception as exc:
            log.debug("[repair/s1] decimation failed (non-fatal): %s", exc)

        # 7. Re-orient normals consistently
        try:
            ms.meshing_re_orient_faces_coherentely()
            log.debug("[repair/s1] re-oriented normals")
        except Exception as exc:
            log.debug("[repair/s1] re_orient_faces failed (non-fatal): %s", exc)

        # PyMeshLab does not support GLB export; save as PLY then convert via trimesh
        with tempfile.TemporaryDirectory() as tmp:
            ply_path = Path(tmp) / "stage1.ply"
            ms.save_current_mesh(str(ply_path))
            mesh = trimesh.load(str(ply_path), force="mesh", process=False)
            mesh.export(str(output_path))
        log.debug("[repair/s1] saved → %s", output_path)

    # ------------------------------------------------------------------
    # Stage 2 — ManifoldPlus
    # ------------------------------------------------------------------

    def _stage2_manifoldplus(self, input_path: Path, output_path: Path) -> None:
        """
        Run ManifoldPlus binary to produce a watertight manifold mesh.

        ManifoldPlus works with OBJ/OFF format, so we convert GLB→OBJ→GLB
        using trimesh around the subprocess call.

        Binary path: cfg.manifoldplus_path (default: "manifoldplus")
        Command: manifoldplus --input <in.obj> --output <out.obj> [--depth <N>]
        """
        binary = self._cfg.manifoldplus_path

        # Probe binary existence before doing any file I/O
        if shutil.which(binary) is None:
            raise FileNotFoundError(f"ManifoldPlus binary not found: {binary!r}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obj_in  = tmp_path / "in.obj"
            obj_out = tmp_path / "out.obj"

            # Convert input (any format) → OBJ for ManifoldPlus
            mesh = trimesh.load(str(input_path), force="mesh", process=False)
            if isinstance(mesh, trimesh.Scene):
                meshes = [g for g in mesh.geometry.values()
                          if isinstance(g, trimesh.Trimesh)]
                mesh = trimesh.util.concatenate(meshes) if meshes else mesh
            mesh.export(str(obj_in))

            cmd = [binary, "--input", str(obj_in), "--output", str(obj_out),
                   "--depth", str(self._cfg.manifoldplus_depth)]
            log.debug("[repair/s2] running: %s", shlex.join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                raise RuntimeError(
                    f"ManifoldPlus exited with code {result.returncode}. "
                    f"stderr: {result.stderr[:500]}"
                )
            if not obj_out.exists():
                raise RuntimeError("ManifoldPlus produced no output file")

            # Convert OBJ → GLB
            repaired = trimesh.load(str(obj_out), force="mesh", process=False)
            repaired.export(str(output_path))
            log.debug("[repair/s2] saved → %s", output_path)

    # ------------------------------------------------------------------
    # Stage 3 — Instant Meshes  (optional)
    # ------------------------------------------------------------------

    def _stage3_instant_meshes(self, input_path: Path, output_path: Path) -> None:
        """
        Run Instant Meshes to produce a clean uniform-triangle remesh.

        Produces regular topology suited for vertex-level morphing.
        Instant Meshes works best with OBJ input, so we convert around it.

        Binary path: cfg.instant_meshes_path (default: "InstantMeshes")
        Command: InstantMeshes <input> -o <output> -f <faces>
        """
        binary = self._cfg.instant_meshes_path

        if shutil.which(binary) is None:
            raise FileNotFoundError(f"Instant Meshes binary not found: {binary!r}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            obj_in  = tmp_path / "in.obj"
            obj_out = tmp_path / "out.obj"

            # Convert input → OBJ
            mesh = trimesh.load(str(input_path), force="mesh", process=False)
            if isinstance(mesh, trimesh.Scene):
                meshes = [g for g in mesh.geometry.values()
                          if isinstance(g, trimesh.Trimesh)]
                mesh = trimesh.util.concatenate(meshes) if meshes else mesh
            mesh.export(str(obj_in))

            cmd = [binary, str(obj_in), "-o", str(obj_out),
                   "-f", str(self._cfg.target_faces)]
            # Instant Meshes (nanogui/GLFW) needs a display even in batch mode.
            # Wrap with xvfb-run when no DISPLAY is available (Docker / CI).
            import os
            if not os.environ.get("DISPLAY") and shutil.which("xvfb-run"):
                cmd = ["xvfb-run", "-a"] + cmd
            log.debug("[repair/s3] running: %s", shlex.join(str(c) for c in cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                raise RuntimeError(
                    f"Instant Meshes exited with code {result.returncode}. "
                    f"stderr: {result.stderr[:500]}"
                )
            if not obj_out.exists():
                raise RuntimeError("Instant Meshes produced no output file")

            remeshed = trimesh.load(str(obj_out), force="mesh", process=False)
            remeshed.export(str(output_path))
            log.debug("[repair/s3] saved → %s", output_path)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _face_count(path: Path) -> int | None:
    """Return face count of mesh at *path*, or None on failure."""
    try:
        m = trimesh.load(str(path), force="mesh", process=False)
        if isinstance(m, trimesh.Scene):
            meshes = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
            return sum(len(g.faces) for g in meshes)
        return len(m.faces) if isinstance(m, trimesh.Trimesh) else None
    except Exception:
        return None
