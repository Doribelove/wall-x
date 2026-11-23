# WALL-OSS-0.5 在 Isaac Sim 中做五个双臂任务

这项工作把自变量发布的 WALL-OSS-0.5 接到 Isaac Sim 6.0.1 的 EX001 双臂机器人，在 RTX 5090 上录下五次任务运行。模型使用官方预训练权重，直接读取机头、右腕 RGB 和机器人状态。我们实现了推理适配、仿真接口和任务控制器。五段录像共对应 479 次模型推理、3,832 个控制步。[官方模型](https://huggingface.co/x-square-robot/wall-oss-0.5/tree/b7dc23a82dbf70a65859c9fdd596703234597028) · [代码上游](https://github.com/X-Square-Robot/wall-x/tree/ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f)

## 录像与结果

下表每行都是一次选定运行。时间为仿真时间，判定来自仓库内的任务检查脚本。点开录像文件可播放或下载。[指标原值](evidence/results.json) · [验证脚本](verify_trials.py)

| 任务 | 动作顺序 | 仿真时间 / 推理次数 | 实测结果 | 录像 |
| --- | --- | --- | --- | --- |
| D1 教具分装 | 依次夹取两红一蓝 → 按订单入碗 → 送碗 | 21.12 秒 / 66 | 3/3 块入指定碗，送碗误差 7.12 mm | [原始录像](evidence/D1_task_02/trial.mp4) |
| D2 定向擦板 | 取板擦 → 擦指定栏 → 放回板擦 | 10.24 秒 / 32 | 擦除 8/8 处目标，保留 6/6 处标记，板擦归位误差 1.20 mm | [原始录像](evidence/D2_task_04/trial.mp4) |
| D3 旋盖封存 | 扶瓶取盖 → 对轴 → 旋紧 → 送至成品区并松手 | 53.44 秒 / 167 | 旋进 3.78 圈，盖高误差 0.031 mm，送瓶误差 0.26 mm | [原始录像](evidence/D3_assembly_09/trial.mp4) |
| D4 折巾收纳 | 两次对折 → 夹起折件 → 搬运落台 → 双手松开 | 50.56 秒 / 158 | 折件约 149 × 150 mm，落点中心误差 3.03 mm。最终四角偏差 28.2 mm，超过原定 15 mm，按原标准部分完成 | [双臂回放](evidence/D4_carry_14/bimanual-replay.mp4) · [原始录像](evidence/D4_carry_14/trial.mp4) |
| D5 USB 插接 | 双手持件 → 对正插入 → 先放一手 → 双手释放 | 17.92 秒 / 56 | 终态插深 9.56 mm，单手保持 2.92 秒，双手松开后保持 3.20 秒 | [原始录像](evidence/D5_hold_17/trial.mp4) |

D4 的原始外景没有完整拍到双臂。展示版用同一轮实测关节角和 225 个布面节点重建左侧全景，右侧保留原始机头和右腕画面。两版均为 25 fps、1,265 帧。[回放程序](render_d4_arms.py) · [两版录像校验记录](evidence/D4_carry_14/replay-metrics.json)

## 我们改了什么

1. **让官方权重按原尺寸加载。** 推理服务使用 PyTorch SDPA，把 FlashAttention 的导入移到对应后端，让视觉 SDPA 接受上游传入的序列长度提示，并按 checkpoint 恢复视觉 MLP 和词嵌入表尺寸。服务启动时在临时源码副本上应用补丁。原运行在 GPU 上将 1,061 个参数张量按加载精度逐一和官方文件比对，覆盖 4,162,224,832 个参数。本项目没有微调或训练 LoRA。[实际补丁](runtime/wall-oss-0.5-sdpa.patch) · [加载记录](evidence/model-load.json)
2. **接通观测和动作。** 每次推理输入两张 640 × 480 RGB、语言指令和 26 维状态，沿用官方 EX001 归一化配置 `ex_normal`。模型给出 32 × 26 的动作块。控制循环以 25 Hz 执行前 8 行，每 0.32 秒仿真时间重新观察。[推理服务](runner/policy.py) · [相机](runner/camera_capture.py) · [仿真循环](runner/controller.py)
3. **把任务拆成双臂能稳定执行的动作。** 录像采用 `assisted` 模式。控制器可以读取仿真物体位姿和布面节点，负责阶段切换、左右臂目标和夹爪开合。模型每行输出中的右手 XYZ 经限幅后叠加到右手目标。D1/D2 最大修正为 2.5 mm，D3/D5 为 1.5 mm，D4 折叠为 0.20 mm、搬运为 0.15 mm。差分逆运动学、关节驱动力限制及物理子步插值把目标转成机械臂运动。[D1/D2 控制](runner/task_improved.py) · [D3/D5 装配](runner/mating.py) · [D4 搬运](runner/fold_carry.py)
4. **实现任务所需的仿真接触。** D1 通过指垫摩擦夹块。D2 在板擦进入夹持区时建立抓取约束，用板面接触和擦拭轨迹判定消除。D3 使用螺纹导向，D5 对准后启用插合导向。D4 在夹指闭合时约束邻近布面节点，松手后继续求解布面运动。[抓取与释放](runner/grasp.py) · [场景](runner/assets)

早期 D1 直接控制运行了 48.00 秒，右腕累计移动 5.15 m，三个目标都留在碗外。表中的辅助控制版本还调整了场景和指令。估计模型修正量的独立贡献，还需要在相同布置下做一次关闭修正量的对照。五段结果记录了开发期间选中的运行，仓库提供最终版控制器；同任务多次重跑的成功率尚待测量。[初始直控数据](evidence/d1-direct.json)

## 在普通电脑上复核

从仓库根目录运行。需要 Python、NumPy、SciPy、FFmpeg 和 ffprobe。以下命令会解码五段原始录像，从每个模型动作块取出执行过的 8 行，与控制轨迹逐行比较，再从终态重新计算任务指标。预期 D1、D2、D3、D5 输出 `sim_pass: true`，D4 输出 `false`。

```bash
python3 examples/isaac_sim/verify_trials.py \
  D1_task_02 D2_task_04 D3_assembly_09 D4_carry_14 D5_hold_17
```

每个目录保留原始录像、首尾观测、动作块、推理回执、控制轨迹和任务终态。轨迹用 gzip 无损压缩，终态中的场景路径只保留文件名。[运行记录](evidence)

## 在 GPU 上重新运行

原运行使用 Linux、RTX 5090、PyTorch 2.10.0+cu128 和 Isaac Sim 6.0.1。模型与 Isaac Sim 使用独立 Python 环境，通过本地文件交换观测和动作。两个终端都从这个仓库根目录运行。

**模型环境。** 使用 Python 3.10、CUDA 12.8 工具链、GNU patch 和上游依赖。在已激活的模型环境中执行以下命令；`pip install -e` 会构建上游 CUDA 扩展。示例用 PyTorch 的 SDPA 路径。[上游部署说明](../../workspace/rtx5090/DEPLOY.md) · [官方模型介绍](https://x2robot.com/docs/quantax1pro/v1/model-development/wall-oss/model-introduction)

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0
python -m pip install -r requirements.txt
MAX_JOBS=4 python -m pip install --no-build-isolation -e .
hf download x-square-robot/wall-oss-0.5 \
  --revision b7dc23a82dbf70a65859c9fdd596703234597028 \
  --local-dir examples/isaac_sim/.runtime/checkpoint
python examples/isaac_sim/runner/policy.py \
  --checkpoint examples/isaac_sim/.runtime/checkpoint
```

等模型终端出现 `ALL_PARAMETERS_ON_GPU`。服务会核对官方单文件权重的 SHA-256，再逐张量比对。权重文件为 8,331,486,672 字节，以官方 Hugging Face 版本为准。

**仿真环境。** 在第二个终端激活可运行 Isaac Sim 6.0.1 和 RTX 相机的环境。仿真脚本还使用 NumPy、SciPy、Pillow、FFmpeg、DejaVu Sans 字体。下面的命令使用上表的辅助控制模式和 D4 搬运配置。

```bash
SIM=examples/isaac_sim
RUN_SET=$(date +%Y%m%d-%H%M%S)
for TASK in D1 D2 D3 D4 D5; do
  RUN_ID="${TASK}-${RUN_SET}"
  FOLD_CASE=carry python "$SIM/runner/controller.py" \
    --task "$TASK" --mode assisted --uid "$RUN_ID" --chunks 435 || break
  python "$SIM/runner/assemble_evidence.py" "$RUN_ID" || break
done
```

录像写到 `examples/isaac_sim/.runtime/runs/<RUN_ID>/trial.mp4`。模型和仿真也支持用同一个 `--work-dir` 把新实验写到其他目录。对新录像运行 `verify_trials.py --raw <work-dir>/runs <RUN_ID>` 可复用上面的检查。

重建 D4 展示版时需要 VTK、OpenCV、NumPy、SciPy、Pillow、FFmpeg 和 Noto Sans CJK 字体。命令读取原录像与实测轨迹。

```bash
SIM=examples/isaac_sim
python "$SIM/render_d4_arms.py" --run "$SIM/evidence/D4_carry_14" \
  --model "$SIM/robot/urdf" --out D4-bimanual-replay.mp4
```

EX001 运动学与视觉网格来自自变量 SDK 的 Quanta X1 模型。本项目为桌面接触制作了简化碰撞体、相对路径 URDF 和固定底座 USD，并保留原 SDK 许可。场景、机位、控制器和本页结果均针对这里的仿真设置。[SDK 源版本](https://github.com/X-Square-Robot/sdk_robot/blob/fb48e143f78251ffc7fff12bd727f7e31b21ef38/urdf/quanta_x1/ex001_robot.urdf) · [网格来源及哈希](robot/mesh-sources.json) · [SDK 许可](robot/LICENSE.sdk.txt) · [Wall-X 代码许可](../../LICENSE)
