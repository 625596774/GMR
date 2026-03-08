#!/usr/bin/env python3
"""
Convert GMR retargeted pkl (root_pos, root_rot, dof_pos) to the format expected by
GR00T-WholeBodyControl's reference/convert_motions.py, so that the result can be
turned into reference CSV directories for sim (Normal mode playback).

GMR pkl format: single motion with keys root_pos (T,3), root_rot (T,4), dof_pos (T,29), fps.
  - GMR uses quaternion (x, y, z, w) = xyzw everywhere (pkl, FK input/output).
GR00T convert_motions expects: dict of motion_name -> motion_data, where motion_data has
  - body_quat_w and CSV body_quat in (w, x, y, z) = wxyz. All "_w" body fields are wxyz.
  joint_pos (T,29), joint_vel (T,29), body_pos_w (T,14,3), body_quat_w (T,14,4),
  body_lin_vel_w (T,14,3), body_ang_vel_w (T,14,3), and optionally _body_indexes, time_step_total.

Usage:
  cd Humanoid/GMR && python scripts/gmr_pkl_to_gr00t_reference.py --pkl retargeted.pkl --output gr00t_format.pkl
  # Then in GR00T: python gear_sonic_deploy/reference/convert_motions.py gr00t_format.pkl reference/gmr_retargeted/
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp

# GR00T reference: 14 body indices in the order expected by CSV (metadata.txt).
# These are body indices in the **GR00T** MuJoCo model (g1_29dof_old.xml, no toe/contour extras).
GR00T_BODY_INDEXES = [0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]
# GMR g1_mocap_29dof.xml has extra bodies (left_toe_link=7, pelvis_contour_link=8, right_toe=15,
# head_link, head_mocap, imu_in_torso, left/right_rubber_hand), so the same-named body has a different
# index. Map: for each of the 14 bodies (in GR00T column order), the index in GMR FK output.
# Order: pelvis, left_knee, right_knee, left_shoulder_yaw, left_ankle_pitch, right_ankle_pitch,
#        left_elbow, right_hip_yaw, left_shoulder_pitch, left_wrist_yaw, right_wrist_pitch,
#        left_shoulder_roll, right_shoulder_pitch, right_wrist_yaw.
GMR_BODY_INDEXES_FOR_GR00T_14 = [0, 4, 12, 24, 5, 13, 25, 11, 22, 28, 35, 23, 30, 36]

# G1 joint order: GMR dof_pos is in MuJoCo (XML depth-first) order. GR00T visualize_motion and
# official reference CSV use: dof[i] = csv[isaaclab_to_mujoco[i]], so CSV column j must equal
# value for MuJoCo joint mujoco_to_isaaclab[j]. So we output joint_pos[:, j] = dof_pos[:, mujoco_to_isaaclab[j]].
# (policy_parameters.hpp: isaaclab_to_mujoco, mujoco_to_isaaclab)
MUJOCO_TO_ISAACLAB = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]
ISAACLAB_G1_JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

# SONIC control loop runs at 50 Hz; reference CSV must be one row per control step
TARGET_FPS = 50
DT_TARGET = 1.0 / TARGET_FPS


def verify_g1_order(xml_path):
    """
    Load GMR G1 model and print body/joint order; verify alignment with GR00T (IsaacLab).
    Use: python gmr_pkl_to_gr00t_reference.py --verify-order [--robot unitree_g1]
    """
    from general_motion_retargeting.kinematics_model import KinematicsModel

    model = KinematicsModel(str(xml_path), device="cpu")
    names = model.body_names
    print("GMR G1 body order (first 38 bodies, depth-first from XML):")
    for i in range(min(38, len(names))):
        mark = "  <-- GR00T 14" if i in GR00T_BODY_INDEXES else ""
        print(f"  body {i:2d}: {names[i]}{mark}")
    print("\nGR00T expects these 14 body indices (column order in body_pos/body_quat):")
    for k, idx in enumerate(GR00T_BODY_INDEXES):
        print(f"  column {k} -> body {idx}: {names[idx] if idx < len(names) else '?'}")
    # GMR dof order = bodies with joint, in tree order (same as IsaacLab for G1)
    dof_names = [j.name for j in model._joints if j.dof_dim > 0]
    print("\nGMR DOF order (joint_pos / dof_pos columns):")
    for i, n in enumerate(dof_names[:29]):
        expect = ISAACLAB_G1_JOINT_NAMES[i].replace("_", " ") if i < len(ISAACLAB_G1_JOINT_NAMES) else "?"
        match = "ok" if expect.replace(" ", "_") in n or n.replace("_link", "").replace("_joint", "") in expect.replace(" ", "_") else "CHECK"
        print(f"  dof {i:2d}: {n}  (expect ~{expect}) [{match}]")
    if len(dof_names) != 29:
        print(f"  WARNING: GMR has {len(dof_names)} DOFs, expected 29 for G1.")
    print("\nConclusion: GMR dof order = MuJoCo (XML depth-first). Script outputs joint_pos in CSV column order expected by visualize_motion and official reference (column j = MuJoCo joint mujoco_to_isaaclab[j]).")


def load_gmr_pkl(pkl_path):
    """Load GMR pkl. Returns root_pos, root_rot, dof_pos, fps.
    root_rot is (x,y,z,w) = xyzw (GMR convention); do not convert before passing to FK."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    root_rot = np.asarray(data["root_rot"], dtype=np.float32)  # xyzw
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
    fps = float(data.get("fps", 50.0))
    return root_pos, root_rot, dof_pos, fps


def quat_xyzw_to_wxyz(q):
    """Convert quaternion from GMR convention (x,y,z,w) to GR00T convention (w,x,y,z).
    Use for: body_rot from GMR FK (xyzw) -> body_quat_w (wxyz). q shape (..., 4)."""
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def resample_motion_to_fps(
    joint_pos,
    joint_vel,
    body_pos_w,
    body_quat_w,
    body_lin_vel_w,
    body_ang_vel_w,
    orig_fps,
    target_fps=TARGET_FPS,
):
    """
    Resample motion to target_fps (default 50 Hz for SONIC control loop).
    body_quat_w: (T, 14, 4) in wxyz. Output body quats remain wxyz.
    Uses linear interpolation for positions/velocities and Slerp for quaternions
    (internally wxyz→xyzw for scipy, then xyzw→wxyz out), then recomputes
    velocities from resampled positions with dt = 1/target_fps.
    """
    T = joint_pos.shape[0]
    if T < 2:
        return (
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        )
    t_orig = np.arange(T, dtype=np.float64) / orig_fps
    duration_sec = t_orig[-1]
    T_new = int(round(duration_sec * target_fps))
    if T_new < 2:
        T_new = 2
    t_new = np.arange(T_new, dtype=np.float64) * (1.0 / target_fps)
    dt_new = 1.0 / target_fps

    # Linear interpolation for joint positions
    f_joint = interp1d(
        t_orig, joint_pos, axis=0, kind="linear", fill_value="extrapolate"
    )
    joint_pos_new = f_joint(t_new).astype(np.float32)

    # Linear interpolation for body positions
    body_pos_new = np.zeros((T_new,) + body_pos_w.shape[1:], dtype=np.float32)
    for i in range(body_pos_w.shape[1]):
        for j in range(3):
            f_b = interp1d(
                t_orig, body_pos_w[:, i, j], kind="linear", fill_value="extrapolate"
            )
            body_pos_new[:, i, j] = f_b(t_new)

    # Slerp for body quaternions: body_quat_w is wxyz (GR00T); scipy uses xyzw.
    body_quat_w_xyzw = body_quat_w[:, :, [1, 2, 3, 0]]  # wxyz -> xyzw for scipy
    body_quat_new = np.zeros((T_new,) + body_quat_w.shape[1:], dtype=np.float32)
    for b in range(body_quat_w.shape[1]):
        R = Rotation.from_quat(body_quat_w_xyzw[:, b, :])  # scipy from_quat(xyzw)
        slerp = Slerp(t_orig, R)
        R_new = slerp(t_new)
        q_xyzw = R_new.as_quat()  # scipy returns xyzw
        body_quat_new[:, b, :] = np.concatenate(
            [q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1
        ).astype(np.float32)  # xyzw -> wxyz for GR00T output

    # Recompute velocities from resampled positions
    joint_vel_new = np.zeros_like(joint_pos_new, dtype=np.float32)
    joint_vel_new[:-1] = np.diff(joint_pos_new, axis=0) / dt_new
    joint_vel_new[-1] = joint_vel_new[-2]

    body_lin_vel_new = np.zeros_like(body_pos_new, dtype=np.float32)
    body_lin_vel_new[:-1] = np.diff(body_pos_new, axis=0) / dt_new
    body_lin_vel_new[-1] = body_lin_vel_new[-2]

    body_ang_vel_new = np.zeros_like(body_pos_new, dtype=np.float32)

    return (
        joint_pos_new,
        joint_vel_new,
        body_pos_new,
        body_quat_new,
        body_lin_vel_new,
        body_ang_vel_new,
    )


def apply_playback_speed(
    joint_pos,
    joint_vel,
    body_pos_w,
    body_quat_w,
    body_lin_vel_w,
    body_ang_vel_w,
    speed,
    target_fps=TARGET_FPS,
):
    """
    Stretch or compress motion by playback speed (still 50 Hz output).
    speed < 1: slow down (e.g. 0.5 = half speed, 2x duration);
    speed > 1: speed up.
    """
    if speed <= 0 or abs(speed - 1.0) < 1e-6:
        return (
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        )
    T = joint_pos.shape[0]
    dt = 1.0 / target_fps
    t_old = np.arange(T, dtype=np.float64) * dt
    t_max = t_old[-1]
    T_out = max(2, int(round(T / speed)))
    t_content = np.arange(T_out, dtype=np.float64) * dt * speed
    t_content = np.clip(t_content, t_old[0], t_max)

    f_joint = interp1d(
        t_old, joint_pos, axis=0, kind="linear", fill_value="extrapolate"
    )
    joint_pos_new = f_joint(t_content).astype(np.float32)

    body_pos_new = np.zeros((T_out,) + body_pos_w.shape[1:], dtype=np.float32)
    for i in range(body_pos_w.shape[1]):
        for j in range(3):
            f_b = interp1d(
                t_old, body_pos_w[:, i, j], kind="linear", fill_value="extrapolate"
            )
            body_pos_new[:, i, j] = f_b(t_content)

    # body_quat_w is wxyz; scipy uses xyzw → convert for Slerp then back to wxyz.
    body_quat_w_xyzw = body_quat_w[:, :, [1, 2, 3, 0]]  # wxyz -> xyzw
    body_quat_new = np.zeros((T_out,) + body_quat_w.shape[1:], dtype=np.float32)
    for b in range(body_quat_w.shape[1]):
        R = Rotation.from_quat(body_quat_w_xyzw[:, b, :])
        slerp = Slerp(t_old, R)
        R_new = slerp(t_content)
        q_xyzw = R_new.as_quat()
        body_quat_new[:, b, :] = np.concatenate(
            [q_xyzw[:, 3:4], q_xyzw[:, :3]], axis=1
        ).astype(np.float32)  # xyzw -> wxyz

    joint_vel_new = np.zeros_like(joint_pos_new, dtype=np.float32)
    joint_vel_new[:-1] = np.diff(joint_pos_new, axis=0) / dt
    joint_vel_new[-1] = joint_vel_new[-2]

    body_lin_vel_new = np.zeros_like(body_pos_new, dtype=np.float32)
    body_lin_vel_new[:-1] = np.diff(body_pos_new, axis=0) / dt
    body_lin_vel_new[-1] = body_lin_vel_new[-2]

    body_ang_vel_new = np.zeros_like(body_pos_new, dtype=np.float32)

    return (
        joint_pos_new,
        joint_vel_new,
        body_pos_new,
        body_quat_new,
        body_lin_vel_new,
        body_ang_vel_new,
    )


def run_fk_and_build_motion(
    root_pos,
    root_rot,
    dof_pos,
    fps,
    xml_file,
    device="cpu",
    target_fps=TARGET_FPS,
    speed=1.0,
    body_indexes=None,
):
    """Run FK via GMR KinematicsModel and build motion_data for GR00T.
    Resamples to target_fps (default 50 Hz); speed < 1 slows playback (e.g. 0.5 = half speed).
    body_indexes: list of 14 body indices in FK output order for GR00T CSV. If None, use
    GMR_BODY_INDEXES_FOR_GR00T_14 (when FK is from GMR XML). Pass GR00T_BODY_INDEXES when
    using GR00T's XML (--fk-xml) so body tree matches."""
    import torch
    from general_motion_retargeting.kinematics_model import KinematicsModel

    if body_indexes is None:
        body_indexes = GMR_BODY_INDEXES_FOR_GR00T_14
    T = root_pos.shape[0]
    kinematics_model = KinematicsModel(str(xml_file), device=device)

    # GMR FK expects and returns quaternions in xyzw. Do NOT convert root_rot before FK.
    root_pos_t = torch.from_numpy(root_pos).to(device=device, dtype=torch.float32)
    root_rot_t = torch.from_numpy(root_rot).to(device=device, dtype=torch.float32)  # xyzw
    dof_pos_t = torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32)

    # FK: input root_rot in xyzw; output body_rot in xyzw (GMR kinematics_model + torch_utils convention).
    body_pos, body_rot = kinematics_model.forward_kinematics(
        root_pos_t, root_rot_t, dof_pos_t
    )
    body_pos = body_pos.cpu().numpy()
    body_rot = body_rot.cpu().numpy()  # (T, num_joint, 4) xyzw

    # Select the 14 bodies GR00T expects (order matches metadata.txt)
    max_idx = max(body_indexes)
    if body_pos.shape[1] <= max_idx:
        raise RuntimeError(
            f"FK returned {body_pos.shape[1]} bodies but need body index {max_idx}. "
            "Check XML matches the body index set (GMR vs GR00T)."
        )
    body_pos_w = body_pos[:, body_indexes, :].astype(np.float32)  # (T, 14, 3)
    # GR00T expects wxyz. body_rot from FK is xyzw → convert once here; all downstream use wxyz.
    body_rot_14 = body_rot[:, body_indexes, :]  # (T, 14, 4) xyzw
    if body_rot_14.shape[-1] != 4:
        raise RuntimeError("Expected body_rot last dim 4 (quat).")
    body_quat_w = quat_xyzw_to_wxyz(body_rot_14).astype(np.float32)  # (T, 14, 4) wxyz

    # joint_pos: CSV column j = value for MuJoCo joint mujoco_to_isaaclab[j] (match visualize_motion + official ref).
    joint_pos = dof_pos[:, MUJOCO_TO_ISAACLAB].astype(np.float32)

    # joint_vel: finite difference
    dt = 1.0 / fps
    joint_vel = np.zeros_like(joint_pos, dtype=np.float32)
    joint_vel[:-1] = np.diff(joint_pos, axis=0) / dt
    joint_vel[-1] = joint_vel[-2]

    # body_lin_vel_w
    body_lin_vel_w = np.zeros_like(body_pos_w, dtype=np.float32)
    body_lin_vel_w[:-1] = np.diff(body_pos_w, axis=0) / dt
    body_lin_vel_w[-1] = body_lin_vel_w[-2]

    # body_ang_vel_w: leave zero (optional for policy)
    body_ang_vel_w = np.zeros_like(body_pos_w, dtype=np.float32)

    # Resample to target_fps (50 Hz) so SONIC control loop advances 1 frame per tick
    if abs(fps - target_fps) > 0.5:
        T_before = T
        (
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        ) = resample_motion_to_fps(
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
            orig_fps=fps,
            target_fps=target_fps,
        )
        T = joint_pos.shape[0]
        print(f"  Resampled {T_before} frames @ {fps} Hz -> {T} frames @ {target_fps} Hz")

    if abs(speed - 1.0) > 1e-6:
        T_before = T
        (
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        ) = apply_playback_speed(
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
            speed=speed,
            target_fps=target_fps,
        )
        T = joint_pos.shape[0]
        print(f"  Playback speed {speed}: {T_before} frames -> {T} frames (~{1/speed:.2f}x duration)")

    motion_data = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "_body_indexes": np.array(GR00T_BODY_INDEXES, dtype=np.int64),
        "time_step_total": np.int64(T),
    }
    return motion_data


def main():
    parser = argparse.ArgumentParser(
        description="Convert GMR retargeted pkl to GR00T convert_motions.py input format."
    )
    parser.add_argument(
        "--pkl",
        type=str,
        default=None,
        help="Path to GMR retargeted pkl (e.g. retargeted.pkl); required unless --verify-order",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output pkl path (e.g. gr00t_format.pkl); required unless --verify-order",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="unitree_g1",
        help="Robot type used in GMR (default: unitree_g1)",
    )
    parser.add_argument(
        "--motion-name",
        type=str,
        default="motion_0",
        help="Key name for the single motion in the output dict (default: motion_0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda:0", "cuda"],
        help="Device for FK (default: cpu)",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=TARGET_FPS,
        help=f"Output motion frame rate for SONIC (default: {TARGET_FPS} Hz)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed: <1 slow down (e.g. 0.5 = half speed), >1 speed up (default: 1.0)",
    )
    parser.add_argument(
        "--verify-order",
        action="store_true",
        help="Only verify G1 body/joint order (GMR vs GR00T); do not convert a pkl.",
    )
    parser.add_argument(
        "--fk-xml",
        type=str,
        default=None,
        help="Optional path to GR00T G1 XML (e.g. gear_sonic_deploy/g1/g1_29dof_old.xml). Use this so FK body order matches sim; body_indexes=GR00T. If not set, use GMR robot XML and GMR->GR00T body mapping.",
    )
    args = parser.parse_args()

    # Ensure we can import GMR
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from general_motion_retargeting.params import ROBOT_XML_DICT

    if args.verify_order:
        if args.robot not in ROBOT_XML_DICT:
            print(f"Error: robot '{args.robot}' not in ROBOT_XML_DICT.")
            sys.exit(1)
        xml_file = ROBOT_XML_DICT[args.robot]
        if not Path(xml_file).is_file():
            print(f"Error: robot XML not found: {xml_file}")
            sys.exit(1)
        verify_g1_order(xml_file)
        return

    if not args.pkl or not args.output:
        parser.error("--pkl and --output are required when not using --verify-order")
    pkl_path = Path(args.pkl)
    if not pkl_path.is_file():
        print(f"Error: pkl not found: {pkl_path}")
        sys.exit(1)

    if args.fk_xml:
        xml_file = Path(args.fk_xml).resolve()
        if not xml_file.is_file():
            print(f"Error: --fk-xml not found: {xml_file}")
            sys.exit(1)
        body_indexes = GR00T_BODY_INDEXES
        print(f"Using GR00T FK model: {xml_file}")
    else:
        if args.robot not in ROBOT_XML_DICT:
            print(f"Error: robot '{args.robot}' not in ROBOT_XML_DICT. Use e.g. unitree_g1.")
            sys.exit(1)
        xml_file = ROBOT_XML_DICT[args.robot]
        if not Path(xml_file).is_file():
            print(f"Error: robot XML not found: {xml_file}")
            sys.exit(1)
        body_indexes = None  # use GMR_BODY_INDEXES_FOR_GR00T_14

    print(f"Loading GMR pkl: {pkl_path}")
    root_pos, root_rot, dof_pos, fps = load_gmr_pkl(pkl_path)
    T, n_dof = dof_pos.shape
    print(f"  Frames: {T}, DoF: {n_dof}, FPS: {fps}")

    if n_dof != 29 and args.robot == "unitree_g1":
        print(f"  Warning: expected 29 DoF for unitree_g1, got {n_dof}. Continuing anyway.")

    print("Running FK and building GR00T motion data...")
    motion_data = run_fk_and_build_motion(
        root_pos,
        root_rot,
        dof_pos,
        fps,
        str(xml_file),
        device=args.device,
        target_fps=args.target_fps,
        speed=args.speed,
        body_indexes=body_indexes,
    )

    out_dict = {args.motion_name: motion_data}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(out_dict, f)

    print(f"Saved GR00T-format pkl: {out_path}")
    print("")
    print("Next step: run GR00T convert_motions.py to generate reference CSV directory:")
    print(f"  cd /path/to/GR00T-WholeBodyControl/gear_sonic_deploy")
    print(f"  python reference/convert_motions.py {out_path.resolve()} reference/gmr_retargeted/")
    print("")
    print("Then in sim use: bash deploy.sh sim --input-type keyboard --motion-data reference/gmr_retargeted/")


if __name__ == "__main__":
    main()
