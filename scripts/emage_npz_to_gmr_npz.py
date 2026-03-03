"""
将 PantoMatrix/EMAGE 输出的 npz 转为 GMR 可用的 SMPL-X 格式 npz。
转换后可再用: python scripts/smplx_to_robot.py --smplx_file <转换后的npz> --robot unitree_g1 --save_path out.pkl
"""
import argparse
import numpy as np


def convert_emage_to_gmr_npz(emage_path: str, output_path: str):
    """EMAGE npz keys: poses (N,165), trans (N,3), betas (300,), gender, mocap_frame_rate."""
    data = np.load(emage_path, allow_pickle=True)
    poses = data["poses"]
    trans = data["trans"]
    betas_full = data["betas"]
    gender = str(data.get("gender", "neutral"))
    mocap_frame_rate = data["mocap_frame_rate"]
    if hasattr(mocap_frame_rate, "item"):
        mocap_frame_rate = np.int64(mocap_frame_rate.item())
    else:
        mocap_frame_rate = np.int64(mocap_frame_rate)

    assert poses.shape[1] == 165, f"EMAGE poses 应为 (N, 165)，当前: {poses.shape}"
    root_orient = poses[:, 0:3].astype(np.float32)
    pose_body = poses[:, 3:66].astype(np.float32)
    # SMPL-X 2020 等模型 shapedirs 为 26 维，与 PantoMatrix transfer2gmr 一致
    betas = betas_full[:26].astype(np.float32)

    np.savez(
        output_path,
        gender=np.array(gender),
        betas=betas,
        root_orient=root_orient,
        pose_body=pose_body,
        trans=trans.astype(np.float32),
        mocap_frame_rate=mocap_frame_rate,
    )
    print(f"已保存 GMR 格式: {output_path}")
    print(f"  帧数: {pose_body.shape[0]}, FPS: {mocap_frame_rate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EMAGE npz -> GMR SMPL-X npz")
    parser.add_argument("--input", "-i", required=True, help="EMAGE 输出 npz 路径")
    parser.add_argument("--output", "-o", required=True, help="输出的 GMR 格式 npz 路径")
    args = parser.parse_args()
    convert_emage_to_gmr_npz(args.input, args.output)
