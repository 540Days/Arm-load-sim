"""
c_channel_3d_simulator.py

Combines the existing 2D PyNite arm analysis with the user's 3D C-channel STL.

What it does:
- Loads the STL as a 3D surface.
- Uses the STL length and cross-section envelope as approximate dimensions.
- Solves a simplified, single-beam PyNite model at a sequence of arm angles.
- Animates the real 3D C-channel mesh rotating from horizontal to vertical.
- Shows torque vs angle and a moving marker.
- Colours the C-channel by the simplified beam's estimated stress/yield ratio.
- Saves a GIF, torque chart, and CSV results.

Important limitations:
- This is NOT full 3D solid finite-element analysis of the STL.
- The C-channel cross-section properties are approximated from its outer dimensions;
  holes, lips, local buckling, joints, fasteners, and detailed geometry are ignored.
- The motor capacity and material values are illustrative placeholders. Validate them.
- The STL is assumed to run along +Y, with the shoulder at (0, 0, 0), and to rotate
  in the XY plane about the Z axis. Change the alignment if your model differs.

Install:
    python -m pip install PyNiteFEA trimesh numpy matplotlib pillow

Run (put this script beside "C-Channel (Configurable).stl"):
    python c_channel_3d_simulator.py

Optional:
    python c_channel_3d_simulator.py --show
    python c_channel_3d_simulator.py --stl "/path/to/model.stl"
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

try:
    from Pynite import FEModel3D
except ImportError:
    FEModel3D = None

import matplotlib
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# -------------------- Inputs / approximate placeholders --------------------

GRAVITY = 9.81
LOAD_MASS = 1.0                  # kg payload at the tip
E_ALUMINIUM = 70e9               # Pa
G_ALUMINIUM = 26e9               # Pa
NU_ALUMINIUM = 0.33
RHO_ALUMINIUM = 2680             # kg/m^3
YIELD_ALUMINIUM = 190e6          # Pa (approximate placeholder)

MOTOR_CAPACITY_NM = 11.0         # N·m; placeholder, verify real motor/gearbox data
GEAR_RATIO = 1.0

START_ANGLE_DEG = 0
END_ANGLE_DEG = 90
ANGLE_STEP_DEG = 2
GIF_FPS = 12

# The original model uses a thin-wall approximation. This thickness is NOT
# measured from the STL; update it to the actual channel wall thickness.
WALL_THICKNESS = 0.001           # m


# -------------------- Section properties --------------------

def c_channel_properties(web_h, flange_w, t):
    """Approximate A, Iy, Iz, J for a thin C-channel; holes/details ignored."""
    t = min(t, web_h / 3, flange_w / 3)
    rects = [
        (t, web_h, t / 2, 0.0),                         # web
        (flange_w, t, flange_w / 2, web_h / 2 - t / 2),
        (flange_w, t, flange_w / 2, -(web_h / 2 - t / 2)),
    ]
    area = sum(w * h for w, h, _, _ in rects)
    x_bar = sum(w * h * cx for w, h, cx, _ in rects) / area
    iz_strong = sum(w * h**3 / 12 + w * h * cy**2
                    for w, h, _, cy in rects)
    iy_weak = sum(h * w**3 / 12 + w * h * (cx - x_bar)**2
                  for w, h, cx, _ in rects)
    j_torsion = (web_h + 2 * flange_w) * t**3 / 3
    return area, iy_weak, iz_strong, j_torsion


def analyse_pose(angle_deg, arm_length, area, iy, iz, j_torsion,
                 load_mass, section_depth):
    """Solve one pose using a single straight beam, fixed at the shoulder."""
    angle = math.radians(angle_deg)
    model = FEModel3D()

    model.add_node("shoulder", 0.0, 0.0, 0.0)
    model.add_node(
        "tip",
        arm_length * math.cos(angle),
        arm_length * math.sin(angle),
        0.0,
    )
    model.add_material(
        "aluminium", E_ALUMINIUM, G_ALUMINIUM, NU_ALUMINIUM,
        RHO_ALUMINIUM, YIELD_ALUMINIUM
    )
    model.add_section("c_channel", area, iy, iz, j_torsion)
    model.add_member("arm", "shoulder", "tip", "aluminium", "c_channel")
    model.def_support("shoulder", True, True, True, True, True, True)

    # Payload at the tip, plus self-weight distributed vertically downward.
    model.add_node_load("tip", "FY", -load_mass * GRAVITY)
    self_weight_per_m = RHO_ALUMINIUM * area * GRAVITY
    model.add_member_dist_load(
        "arm", "FY", -self_weight_per_m, -self_weight_per_m
    )
    model.analyze()

    combo = "Combo 1"
    shoulder = model.nodes["shoulder"]
    member = model.members["arm"]
    motor_torque = abs(shoulder.RxnMZ[combo])
    max_moment = max(
        abs(member.max_moment("Mz", combo)),
        abs(member.min_moment("Mz", combo)),
    )
    max_stress = max_moment * (section_depth / 2) / max(iz, 1e-12)
    safety_factor = (
        YIELD_ALUMINIUM / max_stress if max_stress > 1.0 else math.inf
    )
    tip_deflection = abs(model.nodes["tip"].DY[combo])

    return {
        "angle_deg": angle_deg,
        "motor_torque": motor_torque,
        "max_stress": max_stress,
        "safety_factor": safety_factor,
        "tip_deflection": tip_deflection,
    }


def rotate_vertices(vertices, angle_deg, pivot):
    """
    STL long axis is assumed +Y. Rotate by (angle - 90 deg) around Z:
    0 deg -> horizontal +X, 90 deg -> vertical +Y, matching the 2D demo.
    """
    theta = math.radians(angle_deg - 90.0)
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return (vertices - pivot) @ rotation.T + pivot


def stress_colour(stress, yield_stress):
    """Simple green -> yellow -> red colour based on estimated stress/yield."""
    ratio = max(0.0, stress / yield_stress)
    if ratio < 0.5:
        t = ratio / 0.5
        return (t, 0.78, 0.18)
    if ratio < 1.0:
        t = (ratio - 0.5) / 0.5
        return (1.0, 0.78 * (1 - t), 0.18)
    return (0.85, 0.05, 0.05)


def save_csv(results, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "angle_deg", "motor_torque", "max_stress",
                "safety_factor", "tip_deflection"
            ]
        )
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stl", default="C-Channel (Configurable).stl",
        help="Path to STL file (default: C-Channel (Configurable).stl)"
    )
    parser.add_argument("--output", default="c_channel_simulation.gif")
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=GIF_FPS)
    parser.add_argument("--load", type=float, default=LOAD_MASS,
                        help="Payload mass in kg")
    parser.add_argument("--show", action="store_true",
                        help="Open the animation window after saving")
    args = parser.parse_args()

    stl_path = Path(args.stl)
    if not stl_path.exists():
        raise FileNotFoundError(
            f"Cannot find STL: {stl_path}\n"
            "Put it beside this script or run with --stl '/full/path/file.stl'."
        )
    if FEModel3D is None:
        raise SystemExit(
            "PyNiteFEA is not installed. Install dependencies with:\n"
            "python -m pip install PyNiteFEA trimesh numpy matplotlib pillow"
        )

    mesh = trimesh.load_mesh(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError("Could not load a triangular mesh from the STL.")

    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    faces = np.asarray(mesh.faces, dtype=int)
    bounds = mesh.bounds
    extents = mesh.extents

    # The supplied STL is oriented along Y. Its cross-section spans X and Z.
    arm_length = float(extents[1])
    flange_width = float(extents[0])
    section_depth = float(extents[2])
    if arm_length <= 0 or flange_width <= 0 or section_depth <= 0:
        raise ValueError("The STL has invalid or zero dimensions.")

    # Use measured outer dimensions, but approximate wall thickness separately.
    area, iy, iz, j_torsion = c_channel_properties(
        web_h=section_depth,
        flange_w=flange_width,
        t=WALL_THICKNESS,
    )

    load_mass = args.load
    frames = args.frames or (
        int((END_ANGLE_DEG - START_ANGLE_DEG) / ANGLE_STEP_DEG) + 1
    )
    angles = np.linspace(START_ANGLE_DEG, END_ANGLE_DEG, frames)

    print(f"Loaded STL: {stl_path}")
    print(f"STL extents (model units): X={extents[0]:.5f}, "
          f"Y={extents[1]:.5f}, Z={extents[2]:.5f}")
    print(f"Using arm length: {arm_length:.4f} m")
    print(f"Using approximate section width/depth: "
          f"{flange_width:.4f} m x {section_depth:.4f} m")
    print(f"Using placeholder wall thickness: {WALL_THICKNESS * 1000:.1f} mm")
    print("Precomputing simplified PyNite beam poses...")

    start = time.perf_counter()
    results = [
        analyse_pose(
            float(angle), arm_length, area, iy, iz, j_torsion,
            load_mass, section_depth
        )
        for angle in angles
    ]
    elapsed = time.perf_counter() - start
    print(f"Solved {len(results)} poses in {elapsed:.2f} s "
          f"({elapsed / len(results) * 1000:.1f} ms/pose)")
    print(f"{'Angle':>7} {'Torque (N m)':>14} {'Stress (MPa)':>14} "
          f"{'Safety factor':>14} {'Tip defl. (mm)':>15}")
    for r in results:
        if abs(r["angle_deg"] % 10) < 1e-6 or r is results[-1]:
            sf = min(r["safety_factor"], 999)
            print(f"{r['angle_deg']:6.1f}° {r['motor_torque']:14.3f} "
                  f"{r['max_stress']/1e6:14.3f} {sf:14.2f} "
                  f"{r['tip_deflection']*1000:15.3f}")

    save_csv(results, "c_channel_results.csv")
    print("Saved results: c_channel_results.csv")

    # Create the 3D mesh animation beside the torque curve.
    fig = plt.figure(figsize=(13, 6.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    axplot = fig.add_subplot(1, 2, 2)
    fig.suptitle("C-channel arm: 3D geometry + simplified PyNite analysis")

    pivot = np.array([0.0, 0.0, 0.0])
    reach = arm_length * 1.18
    cross_extent = max(flange_width, section_depth)
    z_margin = max(cross_extent * 2.0, arm_length * 0.06)

    ax3d.set_title("3D C-channel (colour = estimated beam stress)")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_xlim(-0.12 * reach, 1.12 * reach)
    ax3d.set_ylim(-0.12 * reach, 1.12 * reach)
    ax3d.set_zlim(-z_margin, z_margin)
    ax3d.set_box_aspect((1.1, 1.1, 0.35))
    ax3d.view_init(elev=24, azim=-58)
    ax3d.grid(True)
    ax3d.scatter([0], [0], [0], s=35, color="black", depthshade=False)
    info = ax3d.text2D(0.02, 0.97, "", transform=ax3d.transAxes, va="top")

    first_vertices = rotate_vertices(vertices, float(angles[0]), pivot)
    collection = Poly3DCollection(
        first_vertices[faces], linewidths=0.02, edgecolors="none",
        alpha=1.0
    )
    collection.set_facecolor(stress_colour(results[0]["max_stress"],
                                           YIELD_ALUMINIUM))
    ax3d.add_collection3d(collection)

    plot_angles = [r["angle_deg"] for r in results]
    plot_torques = [r["motor_torque"] for r in results]
    axplot.plot(plot_angles, plot_torques, label="Required shoulder torque")
    axplot.axhline(MOTOR_CAPACITY_NM * GEAR_RATIO, linestyle="--",
                   label="Placeholder motor capacity")
    marker, = axplot.plot([], [], "o", markersize=8, color="black")
    axplot.set_xlabel("Arm angle (degrees)")
    axplot.set_ylabel("Torque (N·m)")
    axplot.set_title("Torque vs arm angle")
    axplot.grid(True, alpha=0.3)
    axplot.legend(loc="best")
    axplot.set_xlim(min(plot_angles), max(plot_angles))
    max_torque = max(max(plot_torques), MOTOR_CAPACITY_NM * GEAR_RATIO, 0.1)
    axplot.set_ylim(0, max_torque * 1.2)

    def draw_frame(i):
        r = results[i]
        rotated = rotate_vertices(vertices, float(r["angle_deg"]), pivot)
        collection.set_verts(rotated[faces])
        collection.set_facecolor(
            stress_colour(r["max_stress"], YIELD_ALUMINIUM)
        )
        marker.set_data([r["angle_deg"]], [r["motor_torque"]])
        motor_ratio = r["motor_torque"] / (MOTOR_CAPACITY_NM * GEAR_RATIO)
        status = "OVER CAPACITY" if motor_ratio >= 1 else "within placeholder"
        sf = min(r["safety_factor"], 999)
        info.set_text(
            f"Angle: {r['angle_deg']:.1f}°\n"
            f"Torque: {r['motor_torque']:.2f} N·m\n"
            f"Motor capacity used: {motor_ratio * 100:.0f}%\n"
            f"Est. beam stress: {r['max_stress'] / 1e6:.2f} MPa\n"
            f"Est. safety factor: {sf:.2f}\n"
            f"Tip deflection: {r['tip_deflection'] * 1000:.2f} mm\n"
            f"{status}"
        )
        return collection, marker, info

    animation = FuncAnimation(
        fig, draw_frame, frames=len(results),
        interval=1000 / args.fps, blit=False
    )

    output_path = Path(args.output)
    animation.save(output_path, writer=PillowWriter(fps=args.fps))
    print(f"Saved animation: {output_path.resolve()}")

    # Also save a standalone torque chart.
    chart_path = Path("c_channel_torque_vs_angle.png")
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    ax2.plot(plot_angles, plot_torques, label="Required shoulder torque")
    ax2.axhline(MOTOR_CAPACITY_NM * GEAR_RATIO, linestyle="--",
                label="Placeholder motor capacity")
    ax2.set_xlabel("Arm angle (degrees)")
    ax2.set_ylabel("Torque (N·m)")
    ax2.set_title("C-channel arm: torque vs angle")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved chart: {chart_path.resolve()}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
