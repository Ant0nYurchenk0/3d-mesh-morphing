"""
LangGraph StateGraph definition for the 3D mesh morphing pipeline.

Graph topology:

  START
    │
    ▼
  [setup]          ← creates session, loads prompt
    │
    ├── has_image? ──► [enhance_image]    ← (1) GPT identity regen (skippable)
    │                        │
    │                        ▼
    │                [image_to_base_mesh] ← (2) image-to-3D base mesh (skippable)
    │                        │
    └── has_mesh? ──► [render_mesh]       ← (3) render input mesh to image
                             │
                        (both paths)
                             │
                             ▼
                      [morph_image]       ← (4) GPT-guided image morphing
                             │
                             ▼
                      [target_mesh]       ← (5) image-to-3D target mesh
                             │
                             ▼
                      [repair_mesh]       ← (6) mesh repair
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
  [morph_meshes_sdf]          [morph_meshes_differential]
  (7a) SDF interpolation      (7b) differential rendering (placeholder)
              │                             │
              └──────────────┬──────────────┘
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
    enhance_image_node,
    image_to_base_mesh_node,
    morph_image_node,
    morph_meshes_differential_node,
    morph_meshes_sdf_node,
    render_mesh_node,
    repair_mesh_node,
    setup_node,
    target_mesh_node,
)
from .state import MorphingState


def _route_after_setup(state: MorphingState) -> str:
    """Route to image flow or mesh flow based on input type."""
    if state.get("input_image"):
        return "enhance_image"
    return "render_mesh"


def _route_morph_method(state: MorphingState) -> str:
    """Route to the requested morphing node."""
    method = state.get("morph_method", "sdf")
    if method == "differential":
        return "morph_meshes_differential"
    return "morph_meshes_sdf"


def build_graph():
    """Compile and return the morphing pipeline as an executable LangGraph."""
    graph = StateGraph(MorphingState)

    graph.add_node("setup", setup_node)
    graph.add_node("enhance_image", enhance_image_node)
    graph.add_node("image_to_base_mesh", image_to_base_mesh_node)
    graph.add_node("render_mesh", render_mesh_node)
    graph.add_node("morph_image", morph_image_node)
    graph.add_node("target_mesh", target_mesh_node)
    graph.add_node("repair_mesh", repair_mesh_node)
    graph.add_node("morph_meshes_sdf", morph_meshes_sdf_node)
    graph.add_node("morph_meshes_differential", morph_meshes_differential_node)

    graph.set_entry_point("setup")

    # Route after setup: image input → enhance_image; mesh input → render_mesh
    graph.add_conditional_edges(
        "setup",
        _route_after_setup,
        {"enhance_image": "enhance_image", "render_mesh": "render_mesh"},
    )

    # Image flow: enhance → image_to_base_mesh → morph_image
    graph.add_edge("enhance_image", "image_to_base_mesh")
    graph.add_edge("image_to_base_mesh", "morph_image")

    # Mesh flow: render_mesh → morph_image (bypasses image_to_base_mesh)
    graph.add_edge("render_mesh", "morph_image")

    # Common tail: morph_image → target_mesh → repair_mesh → (route) → END
    graph.add_edge("morph_image", "target_mesh")
    graph.add_edge("target_mesh", "repair_mesh")

    # Route based on morph_method after repair
    graph.add_conditional_edges(
        "repair_mesh",
        _route_morph_method,
        {
            "morph_meshes_sdf": "morph_meshes_sdf",
            "morph_meshes_differential": "morph_meshes_differential",
        },
    )

    graph.add_edge("morph_meshes_sdf", END)
    graph.add_edge("morph_meshes_differential", END)

    return graph.compile()
