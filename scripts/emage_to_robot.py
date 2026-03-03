"""
Retarget PantoMatrix/EMAGE output npz to robot motion.
EMAGE npz format: poses (N,165), trans (N,3), betas (300,), gender, mocap_frame_rate.
This script loads EMAGE npz and runs GMR retargeting (no need to convert npz manually).
"""
import argparse
import pathlib
import os
import time

import numpy as np

import torch

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import load_emage_npz_file, get_smplx_data_offline_fast
from general_motion_retargeting.kinematics_model import KinematicsModel

from rich import print

if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser(description="Retarget PantoMatrix/EMAGE npz to robot.")
    parser.add_argument(
        "--emage_file",
        help="Path to EMAGE output npz (e.g. PantoMatrix/emage_out/xxx_output.npz).",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
            "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
            "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
            "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong",
            "tienkung", "fourier_gr3",
        ],
        default="unitree_g1",
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion (.pkl).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the motion.",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Record the video.",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        help="Limit playback to original motion FPS.",
    )
    args = parser.parse_args()

    SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"

    if not os.path.isfile(args.emage_file):
        raise FileNotFoundError(f"EMAGE npz not found: {args.emage_file}")

    # Load EMAGE npz and convert to GMR internal format
    print(f"Loading EMAGE npz: {args.emage_file}")
    smplx_data, body_model, smplx_output, actual_human_height = load_emage_npz_file(
        args.emage_file, str(SMPLX_FOLDER)
    )
    print(f"Frames: {smplx_data['pose_body'].shape[0]}, FPS: {smplx_data['mocap_frame_rate']}")

    tgt_fps = 30
    smplx_data_frames, aligned_fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=tgt_fps
    )

    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    # 让人体最低点贴地，避免机器人沉入地面或倾倒（与 fbx_offline_to_robot 一致）
    min_z = np.inf
    for frame_data in smplx_data_frames:
        for body_name, (pos, _quat) in frame_data.items():
            if pos[2] < min_z:
                min_z = pos[2]
    retarget.set_ground_offset(min_z)
    print(f"Ground offset applied: min_z = {min_z:.4f}")

    base_name = os.path.splitext(os.path.basename(args.emage_file))[0]
    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=aligned_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=f"videos/{args.robot}_{base_name}.mp4",
    )

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    i = 0
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break

        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        frame_data = smplx_data_frames[i]
        qpos = retarget.retarget(frame_data)

        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retarget.scaled_human_data,
            human_pos_offset=np.array([0.0, 0.0, 0.0]),
            show_human_body_name=False,
            rate_limit=args.rate_limit,
            follow_camera=False,
        )
        if args.save_path is not None:
            qpos_list.append(qpos)

    if args.save_path is not None:
        import pickle
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        # 保存用 xyzw；FK 也用 xyzw（与 smplx_to_robot_dataset 一致）
        root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])

        # 机器人脚着地：用正运动学算全身最低点，整体下移 root 使最低点贴地
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        try:
            kinematics_model = KinematicsModel(retarget.xml_file, device=device)
            body_pos, _ = kinematics_model.forward_kinematics(
                torch.from_numpy(root_pos).to(device=device, dtype=torch.float32),
                torch.from_numpy(root_rot).to(device=device, dtype=torch.float32),
                torch.from_numpy(dof_pos).to(device=device, dtype=torch.float32),
            )
            lowest_height = torch.min(body_pos[..., 2]).item()
            root_pos[:, 2] = root_pos[:, 2] - lowest_height
            print(f"Height adjust: lowest_body_z = {lowest_height:.4f}, root_pos z shifted so feet on ground.")
        except Exception as e:
            print(f"KinematicsModel height adjust skipped ({e}), saving as-is.")

        motion_data = {
            "fps": aligned_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    robot_motion_viewer.close()
