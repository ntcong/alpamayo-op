import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # geom.py lives one level up
from geom import build_arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--slot", choices=("wide", "tele", "both"), required=True)
    parser.add_argument("--model-id", default="nvidia/Alpamayo2-Edge")
    parser.add_argument(
        "--revision", default="f8951d095de3589b3face1dec3f4394c8ecedb38"
    )
    parser.add_argument("--num-traj", type=int, default=1)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--hybrid", action="store_true", help="vision encoder on CPU (8 GB cards)"
    )
    args = parser.parse_args()
    from alpamayo2_edge.checkpoint import load_model
    from alpamayo2_edge.helper import prepare_model_inputs, to_device
    from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS

    model, _ = load_model(args.model_id, revision=args.revision, device="cuda")
    if args.hybrid:
        from hybrid import to_hybrid

        to_hybrid(model)
    paths = sorted(Path(args.samples).glob("*.npz"))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    done = 0
    for path in paths:
        dest = Path(args.out) / f"{path.stem}.json"
        if dest.exists():
            continue
        with np.load(path, allow_pickle=False) as z:
            data = {key: z[key] for key in z.files}
        images, cameras, timestamps = build_arrays(data, args.slot)
        raw = {
            # build_arrays returns [22, 3, H, W]; Edge wants [22, 1, 3, H, W]
            "image_frames": torch.from_numpy(images).unsqueeze(1).contiguous(),
            "camera_indices": torch.from_numpy(cameras),
            "absolute_timestamps": torch.from_numpy(timestamps),
            "ego_history_xyz": torch.from_numpy(data["ego_history_xyz"])
            .float()
            .unsqueeze(0),
            "ego_history_rot": torch.from_numpy(data["ego_history_rot"])
            .float()
            .unsqueeze(0),
            "nav_text": [str(data["nav_text"].item())],
            "camera_pixel_bounds": dict(CAMERA_PIXEL_BOUNDS),
            "min_pixels": 163840,
            "max_pixels": 196608,
            "input_profile": "camera_subsampled",
            "history_num_steps": 20,
        }
        start = time.perf_counter()
        torch.cuda.synchronize()
        inputs = to_device(
            prepare_model_inputs(raw, model.config, model.tokenizer), device="cuda"
        )
        torch.manual_seed(args.seed)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            xyz, _, _, extra = model.sample_trajectories_from_data(
                inputs,
                num_traj_sets=args.num_traj,
                num_traj_samples=1,
                max_generation_length=256,
                diffusion_kwargs={"inference_step": 10, "generator": generator},
                return_extra=True,
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        # extra["cot"] is a string array shaped [batch, sets, samples]
        cot = np.asarray(extra["cot"]).reshape(-1).tolist()
        pred = xyz[0, :, 0].detach().float().cpu().tolist()
        record = {
            "pred_xyz": pred,
            "cot": cot,
            "seconds": seconds,
            "warmup": done == 0,
            "slot": args.slot,
            "nav_text": raw["nav_text"][0],
            "model_id": args.model_id,
            "revision": args.revision,
            "seed": args.seed,
            "hybrid": args.hybrid,
        }
        dest.write_text(json.dumps(record) + "\n")
        print(dest.name, seconds, flush=True)
        done += 1
        if args.limit and done >= args.limit:
            break
    print(f"peak_cuda_memory={torch.cuda.max_memory_allocated()}")


if __name__ == "__main__":
    main()
