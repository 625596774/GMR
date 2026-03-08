## Retargeting from PantoMatrix/EMAGE to Robot

[PantoMatrix/EMAGE](https://github.com/PantoMatrix/PantoMatrix) 从语音生成 SMPL-X 动作并输出为 npz（keys: `poses` (N,165), `trans`, `betas`, `mocap_frame_rate` 等）。GMR 可直接读取该格式并做 retargeting：

```bash
python scripts/emage_to_robot.py --emage_file <path_to_emage_output.npz> --robot unitree_g1 --save_path <path_to_robot_motion.pkl> --rate_limit
```

例如使用 PantoMatrix 生成的文件：

```bash
python scripts/emage_to_robot.py \
  --emage_file ../PantoMatrix/emage_out/2_scott_0_103_103_28s_output.npz \
  --robot unitree_g1 \
  --save_path retargeted.pkl
```

若希望先转成 GMR 标准 SMPL-X npz 再用 `smplx_to_robot.py`，可使用：

```bash
python scripts/emage_npz_to_gmr_npz.py --input <emage.npz> --output <gmr_format.npz>
python scripts/smplx_to_robot.py --smplx_file <gmr_format.npz> --robot unitree_g1 --save_path out.pkl
```