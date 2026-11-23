"""Run the vendor's native continuous flow policy on EX001 observations from Isaac."""
import argparse
import hashlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np


ORDER = (
    ("follow_left_ee_cartesian_pos", 3), ("follow_left_ee_rotation_6D", 6), ("follow_left_gripper", 1),
    ("follow_right_ee_cartesian_pos", 3), ("follow_right_ee_rotation_6D", 6), ("follow_right_gripper", 1),
    ("velocity_decomposed", 3), ("height", 1), ("head_actions", 2)
)
VENDOR_SHA = "37f8f327d3f9c27f056ed078db43c0e39b8caf3df0dc0163cdb8cf92c167968a"
SOURCE = "ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f"
CAMS = ("face_view", "right_wrist_view")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(16 << 20), b""):
            h.update(data)
    return h.hexdigest()


def write_json(p, data):
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    t.replace(p)


SIM = Path(__file__).resolve().parents[1]
REPO = SIM.parents[1]
PATCH = SIM / "runtime/wall-oss-0.5-sdpa.patch"
MODE = "SDPA, deferred FlashAttention import and checkpoint-native parameter shapes"


def prepare_runtime(repo, target):
    """Apply the recorded inference-only patch to a disposable copy of wall_x."""
    shutil.copytree(repo / "wall_x", target / "wall_x",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    subprocess.run(["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", str(PATCH)],
                   cwd=target, check=True)


def serve(args):
    os.environ["WALLX_VISION_ATTN_IMPLEMENTATION"] = "sdpa"
    os.environ["ENABLE_FAST_PREPROCESS"] = "False"
    import torch
    torch.set_num_threads(3)
    run, ck = args.work_dir.resolve(), args.checkpoint.resolve()
    queue = run / "ipc"
    queue.mkdir(exist_ok=True)
    model_file = ck / "model.safetensors"
    if model_file.stat().st_size != 8331486672 or digest(model_file) != VENDOR_SHA:
        raise ValueError("Expected the official WALL-OSS-0.5 safetensors revision listed in the README.")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch installation and GPU are required for model inference.")
    from wall_x._vendor.harrix.serving._wallx_infer.infer_config import InferConfig
    from wall_x._vendor.harrix.serving._wallx_infer.base_dataclass import RobotStateActionData
    from wall_x._vendor.harrix.serving._wallx_infer.model_wrapper import WallxModelWrapper
    cfg = InferConfig(checkpoint_path=str(ck), train_config_path=str(ck/"config.yml"), robot_type="ex001",
                      robot_use_joint_angle_control=False, model_device="cuda:0", norm_key="ex_normal",
                      cam_names=list(CAMS), action_horizon=32, num_inference_timesteps=10,
                      robot_action_interpolate_multiplier=1, robot_action_end_ratio=1.0,
                      smooth_action=False, smooth_gripper=False)
    assert tuple(cfg.train_config["agent_pos_config"].items()) == ORDER
    info = {"model": "X-Square-Robot/wall-oss-0.5", "model_sha256": VENDOR_SHA, "model_bytes": model_file.stat().st_size,
            "checkpoint": str(ck), "source_commit": SOURCE, "runtime": MODE,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "pid": os.getpid(),
            "state_layout": ORDER, "norm": "ex_normal", "cameras": CAMS,
            "vision_attention": cfg.model_config.vision_config._attn_implementation, "queries": 0}
    print("MODEL_INIT", json.dumps(info), flush=True)
    policy = WallxModelWrapper(cfg)
    policy.model.eval().requires_grad_(False)
    assert policy.norm_key == "ex_normal" and all(y.device.type == "cuda" for y in policy.model.parameters())
    info.update(parameters=sum(y.numel() for y in policy.model.parameters()), cuda_allocated=torch.cuda.memory_allocated(),
                device=str(next(policy.model.parameters()).device))
    from safetensors import safe_open
    proof = []
    with safe_open(str(model_file), framework="pt", device="cpu") as archive:
        for name, param in policy.model.named_parameters():
            original = archive.get_tensor(name)
            same = tuple(original.shape) == tuple(param.shape) and torch.equal(original.to(device=param.device, dtype=param.dtype), param.detach())
            if not same:
                raise RuntimeError("checkpoint tensor mismatch: " + name)
            proof.append(name)
    info["verified_parameter_tensors"] = len(proof)
    info["verified_parameter_elements"] = sum(y.numel() for y in policy.model.parameters())
    info["lm_head_tied_to_input"] = policy.model.lm_head.weight.data_ptr() == policy.model.get_input_embeddings().weight.data_ptr()
    write_json(run / "loaded.json", info)
    print("ALL_PARAMETERS_ON_GPU", info["parameters"], flush=True)
    started = time.monotonic()
    served = set()
    with torch.inference_mode():
        while not args.seconds or time.monotonic() - started < args.seconds:
            requests = sorted(p for p in queue.glob("*.request")
                              if p.name not in served and not p.with_suffix(".reply").exists()
                              and not p.with_suffix(".error").exists())
            if not requests:
                time.sleep(.02)
                continue
            for p in requests:
                served.add(p.name)
                src = p.with_suffix(".npz")
                try:
                    request = json.loads(p.read_text())
                    with np.load(src, allow_pickle=False) as f:
                        state = f["state"]
                        assert state.shape == (26,) and np.isfinite(state).all()
                        obs = {k: f[k].copy() for k in CAMS}
                    assert all(v.shape == (480,640,3) and v.dtype == np.uint8 and v.std() > 1 for v in obs.values())
                    item = RobotStateActionData(config=cfg)
                    at = 0
                    for name, width in ORDER:
                        item.save_state_data_with_key(state[at:at+width].copy(), name, gt_dim=width)
                        at += width
                    item.dof_mask = np.ones((1,32,26), dtype=np.float32)
                    obs["robot_state_action_data"] = item
                    seed = request["seed"]
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    start = time.perf_counter()
                    result = policy.infer_flow_action(obs, request["prompt"])
                    act = result["predict_action"].detach().float().cpu().numpy()
                    assert act.shape == (1,32,26) and np.isfinite(act).all()
                    arr = p.with_suffix(".action.npy")
                    np.save(arr, act[0])
                    info["queries"] += 1
                    write_json(p.with_suffix(".reply"), {"shape":list(act.shape), "seconds":time.perf_counter()-start,
                        "observation_sha256":digest(src), "action_sha256":digest(arr), "seed":seed, "driver":info["runtime"]})
                except Exception:
                    write_json(p.with_suffix(".error"), {"error":traceback.format_exc()})
                    print("REQUEST_ERROR", p.name, traceback.format_exc(), flush=True)
    write_json(run/"policy_final.json", info)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--work-dir", type=Path, default=SIM / ".runtime",
                     help="Shared policy queue and recording directory")
    cli.add_argument("--checkpoint", type=Path, required=True,
                     help="Official WALL-OSS-0.5 checkpoint directory")
    cli.add_argument("--seconds", type=int, default=0,
                     help="Stop accepting new batches after this many seconds (0 runs until interrupted)")
    args = cli.parse_args()
    if args.seconds < 0:
        cli.error("--seconds must be nonnegative")
    run = args.work_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    with (run / "policy.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            cli.error(f"a policy is already serving {run}")
        with tempfile.TemporaryDirectory(prefix="wallx-sdpa-", dir=run) as copy:
            prepare_runtime(REPO, Path(copy))
            sys.path.insert(0, copy)
            serve(args)


if __name__ == "__main__":
    main()
