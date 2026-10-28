"""
Render an STL or STEP CAD model as a rotating 3D GIF.

Install:
    python -m pip install trimesh matplotlib pillow cadquery

Usage:
    python cad_to_gif.py robot_arm.stl
    python cad_to_gif.py robot_arm.step
    python cad_to_gif.py robot_arm.step --output robot_arm.gif --frames 90
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Render without opening a window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import trimesh


def load_model(filename: Path) -> trimesh.Trimesh:
    suffix = filename.suffix.lower()

    if suffix == ".stl":
        loaded = trimesh.load(str(filename), force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError("Could not interpret STL as a mesh.")
        return loaded

    if suffix in {".step", ".stp"}:
        try:
            import cadquery as cq
        except ImportError as exc:
            raise RuntimeError(
                "STEP import needs CadQuery. Install it with:\n"
                "    python -m pip install cadquery"
            ) from exc

        # Import STEP as a CAD shape and tessellate it into triangles.
        assembly_or_workplane = cq.importers.importStep(str(filename))
        shape = assembly_or_workplane.val()
        vertices, faces = shape.tessellate(0.5, 0.1)
        vertices = np.array([v.toTuple() for v in vertices], dtype=float)
        faces = np.array(faces, dtype=int)
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    raise ValueError("Supported file types are .stl, .step, and .stp")


def main():
    parser = argparse.ArgumentParser(description="Turn an STL/STEP model into a rotating GIF.")
    parser.add_argument("input", type=Path, help="Path to an STL or STEP file")
    parser.add_argument("--output", type=Path, default=None, help="Output GIF filename")
    parser.add_argument("--frames", type=int, default=60, help="Number of animation frames")
    parser.add_argument("--fps", type=int, default=20, help="GIF frame rate")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")

    output = args.output or args.input.with_name(args.input.stem + "_rotation.gif")
    mesh = load_model(args.input)

    if mesh.is_empty or len(mesh.vertices) == 0:
        raise ValueError("The imported model contains no vertices.")

    # Use a uniform scale so the model fits in the plot.
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    extent = max(float(np.max(bounds[1] - bounds[0])), 1e-6)
    normalized_vertices = (mesh.vertices - center) / extent

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(args.input.name)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.6, 0.6)
    ax.set_box_aspect((1, 1, 1))

    # Render triangular faces; downsample faces for very large meshes.
    face_indices = np.arange(len(mesh.faces))
    if len(face_indices) > 30000:
        face_indices = face_indices[::int(np.ceil(len(face_indices) / 30000))]
    faces = mesh.faces[face_indices]

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    triangles = normalized_vertices[faces]
    surface = Poly3DCollection(
        triangles,
        facecolor="lightsteelblue",
        edgecolor="none",
        linewidth=0.0,
        alpha=1.0,
    )
    ax.add_collection3d(surface)

    def update(frame):
        ax.view_init(elev=25, azim=frame * 360 / args.frames)
        return (surface,)

    animation = FuncAnimation(
        fig, update, frames=args.frames, interval=1000 / args.fps, blit=False
    )
    animation.save(output, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Saved rotating GIF to: {output.resolve()}")


if __name__ == "__main__":
    main()
