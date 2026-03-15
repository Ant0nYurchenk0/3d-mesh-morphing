"""
LangGraph StateGraph definition for the image-to-3D benchmark pipeline.

Graph topology:

  START
    │
    ▼
  [setup]
    │
    ├── recompute_from set? ──► [recompute] ──► END
    │
    ▼
  [acquire]   ← downloads / generates all meshes
    │
    ▼
  [render]    ← renders meshes to 2D images
    │
    ├── dry_run=True? ──► END
    │
    ▼
  [reconstruct]  ← calls each model API
    │
    ▼
  [evaluate]     ← computes all 8 metrics
    │
    ▼
  [report]       ← writes CSV / Markdown summaries
    │
    ▼
  END

Usage:
    from src.pipeline.graph import build_graph

    graph = build_graph()
    final_state = graph.invoke(initial_state)
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes import (
    acquire_node,
    evaluate_node,
    reconstruct_node,
    recompute_node,
    render_node,
    report_node,
    setup_node,
)
from .state import BenchmarkState


def _route_after_setup(state: BenchmarkState) -> str:
    return "recompute" if state.get("recompute_from") else "acquire"


def _route_after_render(state: BenchmarkState) -> str:
    return END if state.get("dry_run") else "reconstruct"


def build_graph():
    """Compile and return the benchmark pipeline as an executable LangGraph."""
    graph = StateGraph(BenchmarkState)

    graph.add_node("setup", setup_node)
    graph.add_node("acquire", acquire_node)
    graph.add_node("render", render_node)
    graph.add_node("reconstruct", reconstruct_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("report", report_node)
    graph.add_node("recompute", recompute_node)

    graph.set_entry_point("setup")

    graph.add_conditional_edges(
        "setup",
        _route_after_setup,
        {"recompute": "recompute", "acquire": "acquire"},
    )
    graph.add_edge("acquire", "render")
    graph.add_conditional_edges(
        "render",
        _route_after_render,
        {END: END, "reconstruct": "reconstruct"},
    )
    graph.add_edge("reconstruct", "evaluate")
    graph.add_edge("evaluate", "report")
    graph.add_edge("report", END)
    graph.add_edge("recompute", END)

    return graph.compile()
