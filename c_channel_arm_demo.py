"""
c_channel_arm_demo.py

Animate the user's C-channel STL as a 3D robotic-arm link.
The STL is assumed to have its shoulder/base at (0, 0, 0) and its length
along +Y. The animation rotates it from horizontal (0 deg) to vertical
(90 deg), matching the angle convention in arm_demo.py.

This is a VISUALISATION prototype. It does not calculate stress from the STL.
You can later connect the displayed angle to your PyNite analysis results.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def rotate_vertices(vertices, angle_deg, pivot):
    """Rotate XYZ points around the Z axis, relative to a pivot."""
    theta = math.radians(angle_deg - 90.0)
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return (vertices - pivot) @ rotation.T + pivot


def main():
    parser = argparse.ArgumentParser(
        description="Render a rotating 3D C-channel STL as a GIF."
    )
    parser.add_argument(
        "stl", nargs="?",
        default="robot_arm.stl",
        help="Path to the STL file (default: robot_arm.stl)",
    )
    parser.add_argument("--output", default="c_channel_animation.gif")
    parser.add_argument("--frames", type=int, default=46)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--start-angle", type=float, default=0)
    parser.add_argument("--end-angle", type=float, default=90)
    args = parser.parse_args()

    stl_path = Path(args.stl)
    if not stl_path.exists():
        raise FileNotFoundError(
            f"Could not find '{stl_path}'. Put the STL beside this script, "
            "or pass its full path as the first argument."
        )

    mesh = trimesh.load_mesh(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("The file did not load as a triangular STL mesh.")

    # Copy the geometry so the original mesh coordinates stay unchanged.
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    faces = np.asarray(mesh.faces, dtype=int)
    pivot = np.array([0.0, 0.0, 0.0])

    # Estimate a suitable viewing area from the model's length.
    length = float(mesh.extents.max())
    reach = max(length * 1.15, 0.1)
    z_half = max(float(mesh.extents[2]) * 2.0, length * 0.08)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    fig.suptitle("3D C-channel arm — camera view follows the mechanism")

    ax.set_title("Arm angle: 0° = horizontal, 90° = vertical")
    ax.set_xlabel("X (model units)")
    ax.set_ylabel("Y (model units)")
    ax.set_zlabel("Z (model units)")
    ax.set_xlim(-0.15 * reach, 1.15 * reach)
    ax.set_ylim(-0.15 * reach, 1.15 * reach)
    ax.set_zlim(-z_half, z_half)
    ax.set_box_aspect((1.2, 1.2, 0.35))
    ax.view_init(elev=25, azim=-60)
    ax.grid(True)

    # Draw a simple fixed shoulder marker.
    ax.scatter([0], [0], [0], s=45, color="black", depthshade=False)
    info = ax.text2D(0.03, 0.95, "", transform=ax.transAxes)

    # Create the first surface collection, then update its triangle vertices.
    first_angle = args.start_angle
    first_vertices = rotate_vertices(vertices, first_angle, pivot)
    collection = Poly3DCollection(
        first_vertices[faces],
        linewidths=0.03,
        edgecolors="none",
        alpha=1.0,
    )
    collection.set_facecolor("#4F8FC9")
    ax.add_collection3d(collection)

    angles = np.linspace(args.start_angle, args.end_angle, args.frames)

    def draw_frame(i):
        angle = float(angles[i])
        rotated = rotate_vertices(vertices, angle, pivot)
        collection.set_verts(rotated[faces])
        info.set_text(f"Angle: {angle:.1f}°   |   STL triangles: {len(faces):,}")
        return collection, info

    animation = FuncAnimation(
        fig, draw_frame, frames=len(angles), interval=1000 / args.fps,
        blit=False
    )

    output_path = Path(args.output)
    animation.save(output_path, writer=PillowWriter(fps=args.fps))
    print(f"Loaded: {stl_path}")
    print(f"STL dimensions: {mesh.extents}")
    print(f"Saved GIF: {output_path.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
