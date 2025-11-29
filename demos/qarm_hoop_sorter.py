"""
QArm 4-DOF hoop picker & placer
- Loads hoop coordinates from JSON
- Uses geometric inverse kinematics (base rotation + planar 2-link IK)
- Provides safe hover approach, gripper control and simple joint-space interpolation

Fill in LINK LENGTHS and joint limits from your Quanser docs (Frame Assignments / IK PDF).
Replace ArmAPI.simulated methods with real Quanser QArm API calls.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# -----------------------------
# CONFIG: DH-style robot dimensions (replace with exact measurements if different)
# -----------------------------
# Physical dimensions (user-provided)
# L1 = base height (meters)
# L2 = first link length (shoulder -> elbow)
# L3 = small offset / joint offset
# L4 = third link length (forearm)
# L5 = end-effector extension (gripper)
# beta = fixed angular offset between links (degrees)
# BASEPLATE = height of mounting baseplate (meters)
# The code will add BASEPLATE + L1 to compute the shoulder pivot height above ground.
BASEPLATE = 0.10  # meters (10 cm baseplate height)
L1 = 0.14  # meters (base/shoulder vertical offset)
L2 = 0.35  # meters (shoulder -> elbow)
L3 = 0.05  # meters (small joint offset)
L4 = 0.25  # meters (elbow -> wrist / forearm)
L5 = 0.15  # meters (end-effector extension)
BETA_DEG = 8.13
BETA_RAD = math.radians(BETA_DEG)

# Derived shoulder pivot height above ground
SHOULDER_Z = BASEPLATE + L1
# Joint limits (radians). Replace with real limits from PDF.
JOINT_LIMITS = [
    (-math.pi, math.pi),     # theta1 (base rotation)
    (-1.5, 1.5),             # theta2 (shoulder)
    (-1.5, 1.5),             # theta3 (elbow)
    (-math.pi, math.pi)      # theta4 (wrist rotation)
]

# Safety heights
HOVER_DELTA = 0.15  # 15 cm above target for approach/retreat (raised from 10cm for better geometry)

# Trajectory settings
TRAJ_STEPS = 60
TRAJ_DELAY = 0.02  # seconds between trajectory steps (tune to your controller)

# -----------------------------
# Utility functions
# -----------------------------
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# -----------------------------
# Kinematics
# -----------------------------
def forward_kinematics(theta1: float, theta2: float, theta3: float, theta4: float):
    """
    Compute approximate end-effector position for the 4-DOF QArm.
    This uses a common anthropomorphic chain:
       base rotation (theta1), shoulder (theta2), elbow (theta3), wrist rot (theta4)
    Returns: (x, y, z, roll, pitch, yaw)
    Note: Replace with the exact FK from your Forward Kinematics.pdf if it differs.
    """
    # Use DH-style chain with fixed beta offset between link2 and link3.
    # Joint angles: theta1 (base yaw), theta2 (shoulder), theta3 (elbow), theta4 (wrist rotation)
    # Compute planar projection using link lengths L2, L4 and end-effector L5.
    a = L2
    b = L4
    # angle of the forearm (includes fixed beta)
    ang23 = theta2 + theta3 + BETA_RAD
    # end-effector adds L5 along the same direction plus wrist rotation (theta4) for orientation
    ang_end = ang23 + theta4

    z = SHOULDER_Z + a * math.sin(theta2) + b * math.sin(ang23) + L5 * math.sin(ang_end)
    planar = a * math.cos(theta2) + b * math.cos(ang23) + L5 * math.cos(ang_end)

    x = planar * math.cos(theta1)
    y = planar * math.sin(theta1)

    # simple orientation estimate (roll/pitch/yaw)
    roll = 0.0
    pitch = theta2 + theta3 + BETA_RAD + theta4
    yaw = theta1 + theta4

    return x, y, z, roll, pitch, yaw


def inverse_kinematics(x: float, y: float, z: float, elbow: str = 'either', prev_angles: Optional[Tuple[float,float,float]] = None) -> Optional[Tuple[float, float, float]]:
    """
    Solve IK for the first three joints (theta1, theta2, theta3) to reach desired (x,y,z).
    Returns (theta1, theta2, theta3) in radians, or None if unreachable.
    
    If prev_angles is provided, choose configuration closest to previous state (trajectory continuity).

    Approach:
      - theta1 = atan2(y,x)
      - reduce to 2D problem in r-z plane: r = sqrt(x^2+y^2)
      - account for base offset L0 (vertical) and possible wrist length L3 if needed
    """
    # Base rotation
    theta1 = math.atan2(y, x)

    # Project into arm plane (planar distance r and vertical offset z_rel from shoulder pivot)
    r = math.hypot(x, y)
    z_rel = z - SHOULDER_Z
    d = math.hypot(r, z_rel)

    # Two-link reduction: treat a = L2, b = L4 + L5 (effective forearm)
    a = L2
    b = L4 + L5

    # Reachability check (geometric)
    max_reach = a + b
    min_reach = abs(a - b)
    if d > (max_reach + 1e-6) or d < (min_reach - 1e-6):
        return None

    # Law of cosines for angle between a and b (call it gamma)
    cos_gamma = clamp((a*a + b*b - d*d) / (2.0 * a * b), -1.0, 1.0)
    gamma_a = math.acos(cos_gamma)
    gamma_b = -gamma_a

    # alpha is angle from shoulder to target in r-z plane
    alpha = math.atan2(z_rel, r)

    def compute_theta2_from_gamma(gamma):
        # gamma is internal angle between a and b; relationship to joint angles includes fixed BETA
        # theta3 (actual joint) = gamma - BETA_RAD
        theta3 = gamma - BETA_RAD
        # triangle geometry to compute theta2
        beta_ang = math.atan2(b * math.sin(gamma), a + b * math.cos(gamma))
        theta2 = alpha - beta_ang
        return theta2, theta3

    def angle_diff(a, b):
        """Shortest angular distance (handles wrapping)."""
        diff = a - b
        while diff > math.pi:
            diff -= 2*math.pi
        while diff < -math.pi:
            diff += 2*math.pi
        return abs(diff)

    cand = []
    for gamma in (gamma_a, gamma_b):
        theta2, theta3 = compute_theta2_from_gamma(gamma)
        t1 = ((theta1 + math.pi) % (2*math.pi)) - math.pi
        t2 = ((theta2 + math.pi) % (2*math.pi)) - math.pi
        t3 = ((theta3 + math.pi) % (2*math.pi)) - math.pi
        cand.append((t1, t2, t3))

    # Choose configuration
    if elbow == 'either':
        if prev_angles is not None:
            # Choose config closest to previous angles (trajectory continuity)
            chosen = min(cand, key=lambda t: angle_diff(t[1], prev_angles[1]) + angle_diff(t[2], prev_angles[2]))
        else:
            # Default: prefer elbow-down (more stable for most hoops)
            chosen = min(cand, key=lambda t: t[2])
    elif elbow == 'elbow_up':
        chosen = max(cand, key=lambda t: t[2])
    elif elbow == 'elbow_down':
        chosen = min(cand, key=lambda t: t[2])
    else:
        chosen = cand[0]

    # Final joint limit check (theta4 not included here)
    if not within_joint_limits(chosen):
        other = cand[1] if chosen is cand[0] else cand[0]
        if within_joint_limits(other):
            chosen = other
        else:
            return None

    return chosen


def within_joint_limits(thetas: Tuple[float, float, float]) -> bool:
    theta1, theta2, theta3 = thetas
    limits = JOINT_LIMITS
    checks = [
        limits[0][0] <= theta1 <= limits[0][1],
        limits[1][0] <= theta2 <= limits[1][1],
        limits[2][0] <= theta3 <= limits[2][1]
    ]
    return all(checks)


# -----------------------------
# Motion primitives & trajectory
# -----------------------------
def linear_interpolate(a: np.ndarray, b: np.ndarray, steps: int):
    for s in range(1, steps+1):
        yield a + (b - a) * (s / steps)


def send_trajectory(arm, current_angles: List[float], target_angles: List[float], steps: int = TRAJ_STEPS):
    """
    Simple joint-space linear interpolation and send to arm.
    Replace the printing with real arm API invocation.
    """
    a = np.array(current_angles)
    b = np.array(target_angles)
    for q in linear_interpolate(a, b, steps):
        arm.move_joint_angles(list(q))
        time.sleep(TRAJ_DELAY)


def move_to(arm, x: float, y: float, z: float, yaw: float = 0.0, elbow: str = 'either', current_angles: Optional[List[float]] = None):
    """
    Compute IK and move to (x,y,z). Optionally pass current_angles for smooth interpolation.
    Returns final joint angles or raises exception.
    
    If current_angles is provided, IK will prefer solutions close to current state (trajectory continuity).
    """
    prev_ik = None
    if current_angles is not None:
        prev_ik = (current_angles[0], current_angles[1], current_angles[2])
    
    ik = inverse_kinematics(x, y, z, elbow=elbow, prev_angles=prev_ik)
    if ik is None:
        raise ValueError(f"Target ({x:.3f},{y:.3f},{z:.3f}) unreachable.")

    theta1, theta2, theta3 = ik
    theta4 = yaw - theta1

    target = [theta1, theta2, theta3, theta4]

    # clamp theta4
    target[3] = clamp(target[3], JOINT_LIMITS[3][0], JOINT_LIMITS[3][1])

    if current_angles is None:
        try:
            current_angles = arm.get_joint_angles()
        except Exception:
            current_angles = [0.0, 0.0, 0.0, 0.0]

    send_trajectory(arm, current_angles, target, steps=TRAJ_STEPS)
    return target


def pick_hoop(arm, x: float, y: float, z: float, elbow: str = 'either'):
    """Descend vertically over hoop, pick, and lift vertically."""
    # Adaptive hover distance: reduce for far hoops that may not reach full hover height
    hoop_r = math.hypot(x, y)
    if hoop_r > 0.33:
        hover_delta = 0.0  # Skip hover entirely; go straight to hoop
    elif hoop_r > 0.30:
        hover_delta = 0.05  # Minimal hover for borderline hoops
    else:
        hover_delta = HOVER_DELTA  # Full 15cm hover for near hoops
    
    if hover_delta > 0:
        move_to(arm, x, y, z + hover_delta, elbow=elbow, current_angles=arm.get_joint_angles())  # approach at hover height
    else:
        move_to(arm, x, y, z, elbow=elbow, current_angles=arm.get_joint_angles())  # approach directly (no hover)
    # Descend to the hoop nominal height
    move_to(arm, x, y, z, elbow=elbow, current_angles=arm.get_joint_angles())
    # Extra micro-descent (up to 2 cm) to ensure gripper makes contact before closing.
    # If the extra descent is unreachable, ignore and proceed to close gripper.
    try:
        micro_penetration = 0.02
        move_to(arm, x, y, z - micro_penetration, elbow=elbow, current_angles=arm.get_joint_angles())
    except Exception:
        # not reachable — continue and attempt to close at nominal height
        pass
    arm.control_gripper(close=True)
    time.sleep(0.15)
    # Lift vertically to hover (use previous hover_delta selection)
    move_to(arm, x, y, z + hover_delta, elbow=elbow, current_angles=arm.get_joint_angles())  # lift vertically


def place_hoop(arm, px: float, py: float, pz: float, elbow: str = 'either'):
    """Move to pole surface and drop gripper."""
    # Place directly at pole surface (no hover to maximize reachability)
    move_to(arm, px, py, pz, elbow=elbow)
    arm.control_gripper(close=False)  # drop hoop
    time.sleep(0.12)


# -----------------------------
# JSON loading and high-level routine
# -----------------------------
def load_hoop_positions(json_path: str) -> List[Tuple[float,float,float]]:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Hoops JSON not found: {p}")
    with p.open('r') as f:
        data = json.load(f)
    hoops = []
    for obj in data:
        if isinstance(obj, dict):
            if 'position' in obj:
                pos = obj['position']
                hoops.append((float(pos[0]), float(pos[1]), float(pos[2])))
            elif 'x' in obj and 'y' in obj and 'z' in obj:
                hoops.append((float(obj['x']), float(obj['y']), float(obj['z'])))
    return hoops


def sort_all_hoops(arm, json_path: str, pole_pos: Tuple[float,float,float], elbow: str = 'elbow_down'):
    hoops = load_hoop_positions(json_path)
    print(f"Loaded {len(hoops)} hoops from {json_path}")
    for i, (x,y,z) in enumerate(hoops, start=1):
        print(f"Picking hoop {i}/{len(hoops)} at ({x:.3f},{y:.3f},{z:.3f})")
        ik = inverse_kinematics(x,y,z, elbow=elbow)
        if ik is None:
            print(f"  -> unreachable, skipping")
            continue
        pick_hoop(arm, x,y,z, elbow=elbow)
        print(f"  -> picked, moving to pole at {pole_pos}")
        place_hoop(arm, *pole_pos, elbow=elbow)
        print(f"  -> placed hoop {i}")


# -----------------------------
# Simulated Arm API (for testing)
# Replace these methods with real Quanser QArm API calls
# -----------------------------
class ArmAPI:
    def __init__(self):
        self.joints = [0.0, 0.0, 0.0, 0.0]

    def get_joint_angles(self) -> List[float]:
        return self.joints.copy()

    def move_joint_angles(self, angles: List[float]):
        self.joints = angles.copy()
        print(f"[SIM] move_joint_angles -> {np.round(self.joints, 3).tolist()}")

    def control_gripper(self, close: bool = True):
        print(f"[SIM] gripper -> {'CLOSE' if close else 'OPEN'}")


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QArm hoop sorter (simple geometric IK)")
    parser.add_argument("--json", type=str, default=str(Path(__file__).parent / "hoop_positions.json"), help="Path to hoops JSON")
    # default pole coordinate: close and low for maximum reachability
    parser.add_argument("--pole", nargs=3, type=float, default=(0.0, -0.30, 0.15), help="Pole position X Y Z (default: close, low position)")
    parser.add_argument("--dry", action="store_true", help="Use simulated arm API (print-only)")
    parser.add_argument("--elbow", choices=['either','elbow_up','elbow_down'], default='elbow_down')
    args = parser.parse_args()

    if args.dry:
        arm = ArmAPI()
    else:
        arm = ArmAPI()
        print("[main] Note: running with placeholder ArmAPI; replace with real QArm factory.")

    try:
        # Use a reachable pole position (adjusted from user's original)
        # User pole: (-0.09, -0.57, 0.195) is at the workspace edge; move it closer and higher
        pole = tuple(args.pole)
        # Check reachability: if pole is too far, use a closer alternative
        pole_r = math.sqrt(pole[0]**2 + pole[1]**2)
        if pole_r > 0.40:  # beyond typical reachable radius
            print(f"[main] Warning: pole at radius {pole_r:.3f}m is beyond reachable range. Adjusting...")
            # Move pole closer (to 35cm radius along the same direction) and higher
            scale = 0.35 / pole_r
            adjusted_pole = (pole[0]*scale, pole[1]*scale, 0.25)  # Raise to 25cm
            print(f"[main] Using adjusted pole: {adjusted_pole}")
            pole = adjusted_pole
        
        placed_count = sort_all_hoops(arm, args.json, pole, elbow=args.elbow)
        print(f"Done: placed {placed_count} hoops.")
    except Exception as e:
        print("Error during run:", e)
