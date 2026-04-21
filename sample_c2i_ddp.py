# Modified from:
#   LlamaGen:   https://github.com/FoundationVision/LlamaGen/

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn.functional as F
import torch.distributed as dist

from tqdm import tqdm
import os

from PIL import Image
import numpy as np
import math
import argparse

from models.arpg import ARPG_models
from models.vq_model import VQ_models
from utils.rejection_tracker import RejectionTracker


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def build_class_schedule(num_samples, num_classes):
    """
    Build enough class labels for the padded sampling count and shard them in
    filename order so multi-GPU runs keep valid labels for every saved image.
    """
    repeats = math.ceil(num_samples / num_classes)
    classes = np.tile(np.arange(num_classes, dtype=np.int64), repeats)
    return classes[:num_samples]


def main(args):
    # Setup PyTorch:
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Rejection modes use dynamic shapes / variable per-step logic that torch.compile cannot trace.
    if args.rejection_mode != 'none' and args.compile:
        print(f"Disabling torch.compile because --rejection-mode={args.rejection_mode} uses dynamic shapes.")
        args.compile = False

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model = VQ_models['VQ-16']()
    vq_model.to(device)
    vq_model.load_state_dict(torch.load(args.vq_ckpt)['model'], strict=False)
    vq_model.eval()

    # create and load gpt model
    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    latent_size = args.image_size // args.downsample_size
    gpt_model = ARPG_models[args.gpt_model](
        vocab_size=args.codebook_size,
        num_classes=args.num_classes,
    ).to(device=device, dtype=precision)
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp: # fsdp
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint: # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")

    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model,
            mode="reduce-overhead",
            fullgraph=True
        ) # requires PyTorch 2.0 (optional)
    else:
        print(f"no model compile") 

    # Create folder to save samples:
    model_string_name = args.gpt_model.replace("/", "-")
    if args.from_fsdp:
        ckpt_string_name = args.gpt_ckpt.split('/')[-2]
    else:
        ckpt_string_name = os.path.basename(args.gpt_ckpt).replace(".pth", "").replace(".pt", "")
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-size-{args.image_size_eval}-{args.vq_model}-" \
                  f"topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-" \
                  f"cfg-{args.cfg_scale}-cfg-schedule-{args.cfg_schedule}-" \
                  f"sample-schedule-{args.sample_schedule}-step-{args.step}-seed-{args.global_seed}"
    if args.rejection_mode == 'rejection':
        folder_name += (
            f"-mode-rejection-metric-{args.confidence_metric}"
            f"-tau-{args.rejection_threshold}-cap-{args.max_reject_rate}"
        )
    elif args.rejection_mode == 'refinement':
        folder_name += (
            f"-mode-refinement-metric-{args.confidence_metric}-k-{args.refinement_k}"
        )
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    all_classes = build_class_schedule(total_samples, args.num_classes)
    all_classes = all_classes[rank::dist.get_world_size()]
    assert len(all_classes) == samples_needed_this_gpu, "rank-local class schedule does not match sample count"
    
    # Single tracker per rank, reused across batches (only rank 0 writes output).
    tracker = None
    if args.log_json and args.rejection_mode == 'rejection' and rank == 0:
        tracker = RejectionTracker(num_samples=n, seq_len=latent_size * latent_size)

    cur_idx = 0
    for batch_idx, _ in enumerate(pbar):
        # Sample inputs:
        c_indices = torch.from_numpy(all_classes[cur_idx * n: (cur_idx+1)*n]).to(device)
        cur_idx += 1
        qzshape = [len(c_indices), latent_size, latent_size, 256]

        if args.rejection_mode == 'none':
            index_sample = gpt_model.generate(
                c_indices,
                guidance_scale=args.cfg_scale,
                temperature=args.temperature,
                num_iter=args.step,
                cfg_schedule=args.cfg_schedule,
                sample_schedule=args.sample_schedule,
            )
        elif args.rejection_mode == 'rejection':
            # Only wire the tracker on rank 0 for the first batch; avoid concatenating huge logs.
            batch_tracker = tracker if (tracker is not None and batch_idx == 0) else None
            index_sample = gpt_model.generate_with_rejection(
                c_indices,
                guidance_scale=args.cfg_scale,
                temperature=args.temperature,
                num_iter=args.step,
                cfg_schedule=args.cfg_schedule,
                sample_schedule=args.sample_schedule,
                threshold=args.rejection_threshold,
                max_reject_rate=args.max_reject_rate,
                confidence_metric=args.confidence_metric,
                tracker=batch_tracker,
                debug=args.debug,
            )
        elif args.rejection_mode == 'refinement':
            index_sample = gpt_model.generate_with_refinement(
                c_indices,
                guidance_scale=args.cfg_scale,
                temperature=args.temperature,
                num_iter=args.step,
                cfg_schedule=args.cfg_schedule,
                sample_schedule=args.sample_schedule,
                refinement_k=args.refinement_k,
                confidence_metric=args.confidence_metric,
            )
        else:
            raise ValueError(f"unknown rejection_mode: {args.rejection_mode}")

        samples = vq_model.decode_code(index_sample.clone(), shape=(index_sample.shape[0], 8, 16, 16)) # output value is between [-1, 1]
        if args.image_size_eval != args.image_size:
            samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode='bicubic')
        samples = (torch.clamp(127.5 * samples + 128.0, 0, 255)).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        
        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        if tracker is not None:
            tracker.finalize()
            tracker.save_json(args.log_json)
            heatmap_path = os.path.splitext(args.log_json)[0] + "_heatmap.png"
            try:
                tracker.make_heatmap(heatmap_path, grid_size=latent_size)
                print(f"Saved rejection heatmap to {heatmap_path}")
            except Exception as e:
                print(f"Failed to save heatmap: {e}")
            print(f"Saved rejection log to {args.log_json}")
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt-model", type=str, choices=list(ARPG_models.keys()), default="ARPG-L")
    parser.add_argument("--gpt-ckpt", type=str, default=None)
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="c2i", help="class-conditional or text-conditional")
    parser.add_argument("--from-fsdp", action='store_true')
    parser.add_argument("--cls-token-num", type=int, default=1, help="max token number of condition input")
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 
    parser.set_defaults(compile=True)
    parser.add_argument("--compile", dest="compile", action="store_true", help="enable torch.compile for sampling")
    parser.add_argument("--no-compile", dest="compile", action="store_false", help="disable torch.compile for sampling")
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=2.0)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-k", type=int, default=0,help="top-k value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--step", type=int, default=256, help="top-k value to sample with")
    parser.add_argument("--cfg-schedule", type=str, default='linear', choices=['linear', 'constant'], help="top-k value to sample with")
    parser.add_argument("--sample-schedule", type=str, default='arccos', choices=['arccos', 'cosine'], help="top-k value to sample with")

    # Confidence-guided rejection (COMP547 project extension)
    parser.add_argument("--rejection-mode", type=str, default='none',
                        choices=['none', 'rejection', 'refinement'],
                        help="'none' = vanilla ARPG; 'rejection' = defer low-confidence tokens; 'refinement' = post-hoc re-decode ablation")
    parser.add_argument("--confidence-metric", type=str, default='max_prob',
                        choices=['max_prob', 'entropy', 'margin'])
    parser.add_argument("--rejection-threshold", type=float, default=0.5,
                        help="tau: tokens with confidence below this are deferred (pilot grid: {0.3, 0.5, 0.7})")
    parser.add_argument("--max-reject-rate", type=float, default=0.2,
                        help="max fraction of tokens that can be deferred per step (pilot grid: {0.1, 0.2})")
    parser.add_argument("--refinement-k", type=float, default=0.1,
                        help="fraction of lowest-confidence tokens to re-decode (ablation grid: {0.1, 0.2})")
    parser.add_argument("--debug", action='store_true',
                        help="enable runtime invariant assertions in rejection mode")
    parser.add_argument("--log-json", type=str, default=None,
                        help="rank-0 rejection log path (JSON). Heatmap saved beside it.")

    args = parser.parse_args()
    main(args)
