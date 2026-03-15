"""
LangGraph pipeline nodes for the image-to-3D evaluation benchmark.

Each node receives the full BenchmarkState, performs its stage of work,
and returns a partial state update dict.

Node execution order (see graph.py):
  setup → acquire → render → [reconstruct → evaluate] → report
                           └── dry_run? → END
  setup → recompute → END  (when --recompute-metrics is passed)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .state import BenchmarkState

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _null_metrics(error_msg: str) -> dict:
    """Return a metrics dict marking a (shape, model) pair as failed."""
    from src.reporter import null_metrics
    return null_metrics(error_msg)


def _result_key(shape: str, model: str) -> str:
    return f"{shape}__{model}"


def _print_summary(results: dict[tuple[str, str], dict]) -> None:
    print("\n─── Quick summary ───")
    print(f"{'Shape':<10} {'Model':<12} {'CD':>9} {'F':>7} {'MRS':>7}")
    print("─" * 50)
    for (shape, model), m in sorted(results.items()):
        if m.get("error"):
            print(f"{shape:<10} {model:<12} {'ERR':>9} {'—':>7} {'—':>7}  ✗ {m['error'][:30]}")
        else:
            cd = m.get("chamfer_distance")
            fs = m.get("f_score")
            mrs = m.get("morphing_readiness_score")
            print(f"{shape:<10} {model:<12} {cd:>9.5f} {fs:>7.3f} {mrs:>7.3f}")


# ------------------------------------------------------------------
# Node: setup
# ------------------------------------------------------------------

def setup_node(state: BenchmarkState) -> dict:
    """Create a new timestamped Session directory."""
    from src.session import Session

    cfg = state["cfg"]
    session_base = state.get("session_base") or (
        Path(__file__).parent.parent.parent / cfg.pipeline.session_dir
    )
    session = Session(base_dir=session_base)
    log.info("[setup] session: %s", session.dir)
    return {"session": session}


# ------------------------------------------------------------------
# Node: acquire
# ------------------------------------------------------------------

def acquire_node(state: BenchmarkState) -> dict:
    """Download or generate meshes for all requested shapes."""
    from src.downloader import Downloader

    cfg = state["cfg"]
    shapes: dict[str, dict] = state["shapes"]
    models: dict[str, dict] = state["models"]

    downloader = Downloader(cfg)
    mesh_paths: dict[str, str] = {}
    results: dict[str, dict] = dict(state.get("results", {}))
    exit_code: int = state.get("exit_code", 0)

    for shape_name, shape_cfg in shapes.items():
        try:
            mesh_path = downloader.download(shape_name, shape_cfg)
            mesh_paths[shape_name] = str(mesh_path)
            log.info("[acquire] %s -> %s", shape_name, mesh_path)
        except Exception as exc:
            log.error("[acquire] %s failed: %s", shape_name, exc)
            for model_name in models:
                results[_result_key(shape_name, model_name)] = _null_metrics(
                    f"download failed: {exc}"
                )
            exit_code = 1

    return {"mesh_paths": mesh_paths, "results": results, "exit_code": exit_code}


# ------------------------------------------------------------------
# Node: render
# ------------------------------------------------------------------

def render_node(state: BenchmarkState) -> dict:
    """Render all successfully acquired meshes to 2D images."""
    import trimesh
    from src.renderer import Renderer

    cfg = state["cfg"]
    shapes: dict[str, dict] = state["shapes"]
    models: dict[str, dict] = state["models"]
    mesh_paths: dict[str, str] = state.get("mesh_paths", {})
    session = state["session"]
    dry_run: bool = state.get("dry_run", False)

    renderer = Renderer(cfg.render)
    render_images: dict[str, Any] = {}
    results: dict[str, dict] = dict(state.get("results", {}))
    exit_code: int = state.get("exit_code", 0)

    for shape_name in shapes:
        if shape_name not in mesh_paths:
            continue  # download failed; errors already recorded in acquire_node

        mesh_path = Path(mesh_paths[shape_name])
        try:
            mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
            img = renderer.render(mesh)
            render_images[shape_name] = img
        except Exception as exc:
            log.error("[render] %s failed: %s", shape_name, exc)
            for model_name in models:
                results[_result_key(shape_name, model_name)] = _null_metrics(
                    f"render failed: {exc}"
                )
            exit_code = 1
            continue

        if dry_run:
            dry_dir = session.render_only_dir(shape_name)
            dry_dir.save_mesh(mesh_path, ext=mesh_path.suffix.lstrip("."))
            dry_dir.save_render(img)
            log.info("[render] dry-run saved %s", shape_name)

    return {"render_images": render_images, "results": results, "exit_code": exit_code}


# ------------------------------------------------------------------
# Node: reconstruct
# ------------------------------------------------------------------

def reconstruct_node(state: BenchmarkState) -> dict:
    """Call each model API to reconstruct 3D meshes from rendered images."""
    from src.models import get_model_client

    cfg = state["cfg"]
    shapes: dict[str, dict] = state["shapes"]
    models: dict[str, dict] = state["models"]
    mesh_paths: dict[str, str] = state.get("mesh_paths", {})
    render_images: dict[str, Any] = state.get("render_images", {})
    session = state["session"]
    api_delay: float = cfg.pipeline.api_delay_seconds

    recon_paths: dict[str, str] = {}
    results: dict[str, dict] = dict(state.get("results", {}))
    exit_code: int = state.get("exit_code", 0)

    for shape_name in shapes:
        if shape_name not in mesh_paths or shape_name not in render_images:
            continue

        mesh_path = Path(mesh_paths[shape_name])
        img = render_images[shape_name]

        for model_name, model_cfg in models.items():
            key = _result_key(shape_name, model_name)
            if key in results:
                # already failed at an earlier stage
                time.sleep(api_delay)
                continue

            art = session.artifact_dir(shape_name, model_name)
            gt_path = art.save_mesh(mesh_path)
            render_path = art.save_render(img)

            log.info("[reconstruct] %s / %s", shape_name, model_name)
            try:
                client = get_model_client(model_name, model_cfg)
                recon_raw = client.reconstruct(render_path)
                recon_path = art.save_reconstructed(recon_raw)
                recon_paths[key] = str(recon_path)
                log.info("[reconstruct] saved -> %s", recon_path)
            except Exception as exc:
                log.error("[reconstruct] %s/%s failed: %s", shape_name, model_name, exc)
                art.save_error(str(exc))
                results[key] = _null_metrics(str(exc))
                exit_code = 1

            time.sleep(api_delay)

    return {"recon_paths": recon_paths, "results": results, "exit_code": exit_code}


# ------------------------------------------------------------------
# Node: evaluate
# ------------------------------------------------------------------

def evaluate_node(state: BenchmarkState) -> dict:
    """Compute all 8 quality metrics for each successful reconstruction."""
    from src.metrics import MetricsCalculator

    cfg = state["cfg"]
    shapes: dict[str, dict] = state["shapes"]
    models: dict[str, dict] = state["models"]
    recon_paths: dict[str, str] = state.get("recon_paths", {})
    session = state["session"]

    calc = MetricsCalculator(cfg.metrics)
    results: dict[str, dict] = dict(state.get("results", {}))
    exit_code: int = state.get("exit_code", 0)

    for shape_name in shapes:
        for model_name in models:
            key = _result_key(shape_name, model_name)
            if key in results:
                continue  # already failed
            if key not in recon_paths:
                continue  # reconstruction was not attempted

            art = session.artifact_dir(shape_name, model_name)
            gt_path = art.original_path
            recon_path = Path(recon_paths[key])

            if gt_path is None:
                results[key] = _null_metrics("no GT mesh in artifact dir")
                continue

            try:
                metrics = calc.compute_all(gt_path, recon_path)
                art.save_metrics(metrics)
                results[key] = metrics
                log.info(
                    "[evaluate] %s/%s CD=%.5f F=%.3f MRS=%.3f",
                    shape_name, model_name,
                    metrics["chamfer_distance"],
                    metrics["f_score"],
                    metrics["morphing_readiness_score"],
                )
            except Exception as exc:
                log.error("[evaluate] %s/%s failed: %s", shape_name, model_name, exc)
                art.save_error(f"metrics failed: {exc}")
                results[key] = _null_metrics(f"metrics failed: {exc}")
                exit_code = 1

    return {"results": results, "exit_code": exit_code}


# ------------------------------------------------------------------
# Node: report
# ------------------------------------------------------------------

def report_node(state: BenchmarkState) -> dict:
    """Write summary.csv, summary_per_model.csv, and summary.md."""
    from src.reporter import Reporter

    cfg = state["cfg"]
    session = state["session"]
    results_raw: dict[str, dict] = state.get("results", {})

    reporter = Reporter(cfg)

    # Convert "{shape}__{model}" keys to (shape, model) tuples
    results: dict[tuple[str, str], dict] = {}
    for key, metrics in results_raw.items():
        if "__" in key:
            shape, model = key.split("__", 1)
            results[(shape, model)] = metrics

    if results:
        reporter.write(session, results)
        log.info("[report] session complete: %s", session.dir)
        log.info("[report] summary CSV : %s", session.summary_csv_path)
        log.info("[report] summary MD  : %s", session.summary_md_path)
        _print_summary(results)

    return {}


# ------------------------------------------------------------------
# Node: recompute
# ------------------------------------------------------------------

def recompute_node(state: BenchmarkState) -> dict:
    """Re-run metrics on all pairs found in an existing session directory."""
    from src.metrics import MetricsCalculator
    from src.reporter import Reporter
    from src.session import ArtifactDir, Session

    cfg = state["cfg"]
    session_path: Path = state["recompute_from"]

    try:
        session = Session.open_existing(session_path)
    except FileNotFoundError as exc:
        log.error("[recompute] %s", exc)
        return {"exit_code": 1}

    log.info("[recompute] session: %s", session.dir)

    calc = MetricsCalculator(cfg.metrics)
    reporter = Reporter(cfg)
    all_models = list(cfg.models.keys())

    results_raw: dict[str, dict] = {}
    exit_code = 0

    subdirs = sorted(d for d in session.dir.iterdir() if d.is_dir())
    if not subdirs:
        log.error("[recompute] no sub-directories found in %s", session.dir)
        return {"exit_code": 1}

    for subdir in subdirs:
        art = ArtifactDir(subdir)
        gt_path = art.original_path
        recon_path = art.reconstructed_path
        name = subdir.name

        # Parse shape / model from directory name
        if "_" not in name:
            continue
        shape_name = model_name = None
        for m in all_models:
            if name.endswith(f"_{m}"):
                model_name = m
                shape_name = name[: -(len(m) + 1)]
                break
        if not shape_name:
            shape_name, model_name = name.rsplit("_", 1)

        key = _result_key(shape_name, model_name)

        if gt_path is None or recon_path is None:
            log.warning("[recompute] %s — missing mesh file(s), skipping", name)
            results_raw[key] = _null_metrics("missing mesh files")
            continue

        log.info("[recompute] %s / %s", shape_name, model_name)
        try:
            metrics = calc.compute_all(gt_path, recon_path)
            art.save_metrics(metrics)
            results_raw[key] = metrics
            log.info(
                "[recompute] CD=%.5f F=%.3f MRS=%.3f cost=%s",
                metrics["chamfer_distance"],
                metrics["f_score"],
                metrics["morphing_readiness_score"],
                metrics.get("cleanup_cost", "?"),
            )
        except Exception as exc:
            log.error("[recompute] %s failed: %s", name, exc)
            results_raw[key] = _null_metrics(f"metrics failed: {exc}")
            exit_code = 1

    results: dict[tuple[str, str], dict] = {}
    for key, metrics in results_raw.items():
        if "__" in key:
            s, m = key.split("__", 1)
            results[(s, m)] = metrics

    if results:
        reporter.write(session, results)
        log.info("[recompute] summary CSV : %s", session.summary_csv_path)
        log.info("[recompute] summary MD  : %s", session.summary_md_path)
        _print_summary(results)

    return {"exit_code": exit_code}
