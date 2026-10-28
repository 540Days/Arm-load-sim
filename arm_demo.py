"""
arm_demo.py — First PyNite demo for the Arm Load Simulator.

Models a single-jointed VEX arm as ONE beam:
    shoulder (motor, locked) ---------- tip (load)

For each arm angle, it:
  1. builds a 3D frame model in PyNite,
  2. applies gravity (self-weight + load at the tip),
  3. solves it, then reports motor torque, stress, safety factor, deflection.

This is the "quasi-static pose stepping" idea in its smallest form.

Two phases (like rendering a video, then playing it):
  1. PRECOMPUTE  - solve every pose and store the results (timed = benchmark)
  2. PLAY BACK   - animate the stored results; no physics during playback

Units: SI everywhere inside the code (N, m, kg, Pa). Convert only for display.

!! Section and motor numbers are APPROXIMATE placeholders.
!! Confirm real VEX values with the engineering teacher before trusting results.

Setup (once, using a virtual environment):
    python3 -m venv .venv
    .venv/bin/pip install PyNiteFEA matplotlib pillow

Run:
    .venv/bin/python arm_demo.py            # table + chart + arm_animation.gif
    .venv/bin/python arm_demo.py --show     # also open a live animation window
"""

import math
import sys
import time
from Pynite import FEModel3D


cartridges = {"blue":600, "green":200, "red":100}
cartridge = "blue"

motors = {"large":11, "small":5.5}
motor = "large"

arm_height = {"c-channel": 0.0254, "l-channel": 0.0127, "standoff": 0.00635}
arm_width = {"c-channel": 0.0127, "l-channel": 0.0127, "standoff": 0.00635}
wall_thickness = {"c-channel": 0.001, "l-channel": 0.001, "standoff": 0.00635}
arm = "l-channel"

# ---------------------------------------------------------------------------
# Inputs (change these and re-run)
# ---------------------------------------------------------------------------

GRAVITY = 9.81                 # m/s^2

ARM_LENGTH = 0.4            # m  (15 inches)
LOAD_MASS = 1.0                # kg at the tip (e.g. game object + claw)

# Aluminium (5052-H32, approximate)
E_ALUMINIUM = 70e9             # Pa, stiffness (Young's modulus)
G_ALUMINIUM = 26e9             # Pa, shear modulus
NU_ALUMINIUM = 0.33            # Poisson's ratio
RHO_ALUMINIUM = 2680           # kg/m^3, density
YIELD_ALUMINIUM = 190e6        # Pa, stress where it bends permanently

# C-channel 1x2x1, web standing vertical (approximate, holes ignored)
WEB_HEIGHT = arm_height[arm]           # m  (1 inch)
FLANGE_WIDTH = arm_width[arm]          # m  (0.5 inch)
WALL_THICKNESS = wall_thickness[arm]         # m  (~1 mm, CHECK real value)

# Motor (VEX V5 approximate, CHECK spec sheet)
#MOTOR_STALL_TORQUE = motors[motor] # N·m at the motor shaft

GEAR_RATIO = 1.0               # output / input, e.g. 5.0 for a 5:1 reduction

START_ANGLE_DEG = 0            # 0 = horizontal
END_ANGLE_DEG = 90             # 90 = straight up
ANGLE_STEP_DEG = 2             # smaller step = smoother animation, more solves
TABLE_EVERY_DEG = 10           # print a table row every N degrees


# ---------------------------------------------------------------------------
# Section properties
# ---------------------------------------------------------------------------
def c_channel_properties(web_h, flange_w, t):
    """Return (area, I_strong, I_weak, J) for a thin C-channel.

    Built from three rectangles: one vertical web + two horizontal flanges.
    I_strong: bending in the vertical plane (the one gravity causes).
    """
    # (width, height, centre_x, centre_y) for each rectangle
    rects = [
        (t, web_h, t / 2, 0.0),                                  # web
        (flange_w, t, flange_w / 2, web_h / 2 - t / 2),          # top flange
        (flange_w, t, flange_w / 2, -(web_h / 2 - t / 2)),       # bottom flange
    ]
    area = sum(w * h for w, h, _, _ in rects)
    x_bar = sum(w * h * cx for w, h, cx, _ in rects) / area

    # Parallel axis theorem: I = I_own + A * d^2
    i_strong = sum(w * h**3 / 12 + w * h * cy**2 for w, h, _, cy in rects)
    i_weak = sum(h * w**3 / 12 + w * h * (cx - x_bar) ** 2 for w, h, cx, _ in rects)

    # Open thin-walled section: J ≈ sum(b * t^3 / 3). Very small = weak in twist.
    j_torsion = (web_h + 2 * flange_w) * t**3 / 3
    return area, i_strong, i_weak, j_torsion


# ---------------------------------------------------------------------------
# One pose
# ---------------------------------------------------------------------------
def analyse_pose(angle_deg, area, i_strong, i_weak, j_torsion):
    """Build and solve the arm at one angle. Returns a dict of results."""
    angle = math.radians(angle_deg)
    model = FEModel3D()

    # Geometry: arm moves in the X-Y plane, gravity points in -Y.
    model.add_node("shoulder", 0.0, 0.0, 0.0)
    model.add_node("tip", ARM_LENGTH * math.cos(angle), ARM_LENGTH * math.sin(angle), 0.0)
    model.add_node("center", (ARM_LENGTH * math.cos(angle))/2, (ARM_LENGTH * math.sin(angle))/2, 0.0)

    model.add_material("aluminium", E_ALUMINIUM, G_ALUMINIUM, NU_ALUMINIUM,
                       RHO_ALUMINIUM, YIELD_ALUMINIUM)
    model.add_section("c_channel", area, i_weak, i_strong, j_torsion)
    model.add_member("arm", "shoulder", "tip", "aluminium", "c_channel")

    # Motor is "locked" for this snapshot -> fully fixed support.
    model.def_support("shoulder", True, True, True, True, True, True)

    # Loads: payload at the tip + the arm's own weight along its length.
    model.add_node_load("center", "FY", -LOAD_MASS * GRAVITY)
    self_weight_per_m = RHO_ALUMINIUM * area * GRAVITY
    model.add_member_dist_load("arm", "FY", -self_weight_per_m, -self_weight_per_m)

    model.analyze()

    combo = "Combo 1"
    shoulder = model.nodes["shoulder"]
    arm = model.members["arm"]

    motor_torque = abs(shoulder.RxnMZ[combo])           # what the motor must hold
    max_moment = max(abs(arm.max_moment("Mz", combo)),
                     abs(arm.min_moment("Mz", combo)))
    max_stress = max_moment * (WEB_HEIGHT / 2) / i_strong   # sigma = M * c / I
    safety_factor = YIELD_ALUMINIUM / max_stress if max_stress > 1.0 else math.inf  # ~0 stress -> infinite
    tip_deflection = abs(model.nodes["tip"].DY[combo])

    return {
        "angle_deg": angle_deg,
        "motor_torque": motor_torque,
        "max_stress": max_stress,
        "safety_factor": safety_factor,
        "tip_deflection": tip_deflection,
    }


# ---------------------------------------------------------------------------
# Hand-calculation check (validation habit!)
# ---------------------------------------------------------------------------
def hand_calc_torque(area):
    """Horizontal arm: torque = load * L + self_weight * L / 2."""
    arm_weight = RHO_ALUMINIUM * area * ARM_LENGTH * GRAVITY
    return LOAD_MASS * GRAVITY * ARM_LENGTH + arm_weight * ARM_LENGTH / 2


# ---------------------------------------------------------------------------
# Phase 1: precompute every pose (this is also our benchmark)
# ---------------------------------------------------------------------------
def precompute_poses(area, i_strong, i_weak, j_torsion):
    """Solve every pose once and return the stored results + time taken."""
    angles = range(START_ANGLE_DEG, END_ANGLE_DEG + 1, ANGLE_STEP_DEG)
    start = time.perf_counter()
    results = [analyse_pose(a, area, i_strong, i_weak, j_torsion) for a in angles]
    elapsed = time.perf_counter() - start
    return results, elapsed


def print_table(results, motor_capacity):
    print(f"{'Angle':>6} {'Torque N·m':>11} {'Stress MPa':>11} "
          f"{'Safety F.':>10} {'Sag mm':>8}  Motor")
    for r in results:
        if r["angle_deg"] % TABLE_EVERY_DEG != 0:
            continue
        motor_ok = "OK" if r["motor_torque"] < motor_capacity else "STALL!"
        print(f"{r['angle_deg']:>5}° {r['motor_torque']:>11.2f} "
              f"{r['max_stress']/1e6:>11.1f} {min(r['safety_factor'], 999):>10.1f} "
              f"{r['tip_deflection']*1000:>8.2f}  {motor_ok}")


def save_chart(results, motor_capacity, plt):
    angles = [r["angle_deg"] for r in results]
    torques = [r["motor_torque"] for r in results]
    fig, ax = plt.subplots()
    ax.plot(angles, torques, label="Torque needed")
    ax.axhline(motor_capacity, linestyle="--", color="red", label="Motor capacity")
    ax.set_xlabel("Arm angle (deg, 0 = horizontal)")
    ax.set_ylabel("Torque at shoulder (N·m)")
    ax.set_title("Motor torque vs arm angle")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.savefig("torque_vs_angle.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved chart: torque_vs_angle.png")


# ---------------------------------------------------------------------------
# Phase 2: play back the stored results (no physics here!)
# ---------------------------------------------------------------------------
def motor_load_colour(ratio):
    """Green (easy) -> yellow -> red (at/over stall). ratio = torque / capacity."""
    ratio = max(0.0, min(ratio, 1.0))
    if ratio < 0.5:
        return (ratio * 2, 0.75, 0.2)            # green -> yellow
    return (1.0, 0.75 * (1 - (ratio - 0.5) * 2), 0.2)   # yellow -> red


def build_animation(results, motor_capacity, plt):
    from matplotlib.animation import FuncAnimation

    fig, (ax_arm, ax_torque) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Arm {ARM_LENGTH*100:.0f} cm · load {LOAD_MASS} kg · "
                 f"gear {GEAR_RATIO:g}:1 · motor capacity {motor_capacity:.2f} N·m")

    # Left: the arm
    reach = ARM_LENGTH * 1.15
    ax_arm.set_xlim(-0.1 * reach, 1.7 * reach)   # extra room on the right for the readout
    ax_arm.set_ylim(-0.1 * reach, reach)
    ax_arm.set_aspect("equal")
    ax_arm.set_title("Arm (colour = motor effort)")
    ax_arm.plot([-0.1 * reach, 1.7 * reach], [0, 0], color="grey", lw=1)   # base line
    arm_line, = ax_arm.plot([], [], lw=8, solid_capstyle="round")
    load_dot, = ax_arm.plot([], [], "s", markersize=14, color="black")
    ax_arm.plot(0, 0, "o", markersize=12, color="dimgray")           # motor/shoulder
    info = ax_arm.text(0.64, 0.97, "", transform=ax_arm.transAxes,
                       va="top", family="monospace", fontsize=10)

    # Right: torque curve with a moving marker
    angles = [r["angle_deg"] for r in results]
    torques = [r["motor_torque"] for r in results]
    ax_torque.plot(angles, torques, label="Torque needed")
    ax_torque.axhline(motor_capacity, linestyle="--", color="red", label="Motor capacity")
    marker, = ax_torque.plot([], [], "o", markersize=10, color="black")
    ax_torque.set_xlabel("Arm angle (deg)")
    ax_torque.set_ylabel("Torque at shoulder (N·m)")
    ax_torque.set_title("Motor torque vs angle")
    ax_torque.legend(loc="upper right")
    ax_torque.grid(alpha=0.3)

    def draw_frame(i):
        r = results[i]
        angle = math.radians(r["angle_deg"])
        tip_x, tip_y = ARM_LENGTH * math.cos(angle), ARM_LENGTH * math.sin(angle)
        ratio = r["motor_torque"] / motor_capacity

        arm_line.set_data([0, tip_x], [0, tip_y])
        arm_line.set_color(motor_load_colour(ratio))
        load_dot.set_data([tip_x], [tip_y])
        marker.set_data([r["angle_deg"]], [r["motor_torque"]])

        status = "STALL!" if ratio >= 1 else "OK"
        info.set_text(f"angle   {r['angle_deg']:>4}°\n"
                      f"torque  {r['motor_torque']:>5.2f} N·m\n"
                      f"motor   {ratio*100:>4.0f}%  {status}\n"
                      f"metal SF {min(r['safety_factor'], 999):>5.1f}")
        return arm_line, load_dot, marker, info

    return fig, FuncAnimation(fig, draw_frame, frames=len(results),
                              interval=80, blit=False)


def main():
    area, i_strong, i_weak, j_torsion = c_channel_properties(
        WEB_HEIGHT, FLANGE_WIDTH, WALL_THICKNESS)
    motor_capacity = motors[motor] * GEAR_RATIO

    print(f"Arm {ARM_LENGTH*100:.1f} cm, load {LOAD_MASS} kg, "
          f"motor capacity {motor_capacity:.2f} N·m (gear {GEAR_RATIO}:1)\n")

    # Phase 1: precompute (and benchmark)
    results, elapsed = precompute_poses(area, i_strong, i_weak, j_torsion)
    print(f"Precomputed {len(results)} poses in {elapsed:.2f} s "
          f"({elapsed / len(results) * 1000:.1f} ms per pose)\n")

    print_table(results, motor_capacity)

    # Validation: PyNite vs hand calculation at 0° (horizontal)
    expected = hand_calc_torque(area)
    got = results[0]["motor_torque"]
    print(f"\nCheck at 0°: PyNite {got:.3f} N·m vs hand calc {expected:.3f} N·m "
          f"-> {'MATCH' if math.isclose(got, expected, rel_tol=1e-3) else 'MISMATCH'}")

    # Phase 2: visuals from stored results
    try:
        import matplotlib
        if "--show" not in sys.argv:
            matplotlib.use("Agg")          # no window needed, just save files
        import matplotlib.pyplot as plt
    except ImportError:
        print("(Install matplotlib to get the chart and animation.)")
        return

    save_chart(results, motor_capacity, plt)
    fig, animation = build_animation(results, motor_capacity, plt)
    animation.save("arm_animation.gif", writer="pillow", fps=12)
    print("Saved animation: arm_animation.gif")
    if "--show" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()