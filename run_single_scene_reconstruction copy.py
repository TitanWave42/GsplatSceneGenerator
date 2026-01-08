#!/usr/bin/env python3
"""
Single-view Scene Reconstruction - Single Iteration Script

This script runs a single iteration of Single-view Scene Reconstruction using
the Open-DiffusionGS model. It loads the checkpoint from HuggingFace and
performs inference on a single batch of data.

Usage:
    python run_single_scene_reconstruction.py [--checkpoint_path PATH] [--config_path PATH] [--input_image PATH] [--data_path PATH]
    
    Example with input image:
    python run_single_scene_reconstruction.py --input_image path/to/your/image.png
"""

import os
import sys
import argparse
import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image
from easydict import EasyDict as edict
from einops import rearrange
from tqdm import tqdm

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diffusionGS
from diffusionGS.utils.config import load_config, parse_structured, ExperimentConfig
from diffusionGS.systems.utils import TransformInput
from diffusionGS.models.diffusion import create_diffusion
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf, DictConfig


def download_checkpoint_if_needed(checkpoint_path: str) -> str:
    """Download checkpoint from HuggingFace if it's a URL or doesn't exist."""
    if checkpoint_path.startswith("http") or "huggingface.co" in checkpoint_path:
        # Extract repo_id and filename from URL
        # URL format: https://huggingface.co/CaiYuanhao/DiffusionGS/blob/main/scene_ckpt_256.ckpt
        try:
            # Find the part after "huggingface.co/"
            if "huggingface.co/" in checkpoint_path:
                # Split by "huggingface.co/" and take the part after it
                after_hf = checkpoint_path.split("huggingface.co/")[-1]
                parts = after_hf.split("/")
                
                # First two parts should be username/repo_name
                if len(parts) >= 2:
                    repo_id = f"{parts[0]}/{parts[1]}"
                    # Filename is the last part if it ends with .ckpt, otherwise use default
                    if len(parts) > 2 and parts[-1].endswith(".ckpt"):
                        filename = parts[-1]
                    else:
                        filename = "scene_ckpt_256.ckpt"
                else:
                    # Fallback to default
                    repo_id = "CaiYuanhao/DiffusionGS"
                    filename = "scene_ckpt_256.ckpt"
            else:
                # Fallback to default
                repo_id = "CaiYuanhao/DiffusionGS"
                filename = "scene_ckpt_256.ckpt"
        except Exception as e:
            print(f"Warning: Could not parse URL, using defaults: {e}")
            repo_id = "CaiYuanhao/DiffusionGS"
            filename = "scene_ckpt_256.ckpt"
        
        print(f"Downloading checkpoint from HuggingFace: {repo_id}/{filename}")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir="./checkpoints"
        )
        return local_path
    elif os.path.exists(checkpoint_path):
        return checkpoint_path
    else:
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")


def load_config_with_checkpoint(config_path: str, checkpoint_path: str):
    """Load config and update checkpoint path."""
    if config_path and os.path.exists(config_path):
        try:
            cfg = load_config(config_path)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            print("Using default config structure...")
            cfg = None
    else:
        cfg = None
    
    if cfg is None:
        # Use default config structure
        print("Using default config structure...")
        cfg_dict = {
            'exp_root_dir': 'outputs',
            'name': 'single_scene_reconstruction',
            'tag': 'test',
            'seed': 0,
            'data_type': 'Re10k-datamodule',
            'data': {
                'local_dir': '',
                'local_eval_dir': '',
                'sel_views': 3,
                'eval_subset': 1,
                'sel_views_train': 4,
                'training_res': [256, 256],
                'batch_size': 1,
                'eval_batch_size': 1,
                'num_workers': 0,
                'num_workers_val': 0,
            },
            'system_type': 'diffusion-gs-scene-system',
            'system': {
                'num_inference_steps': 30,
                'save_intermediate_video': False,
                'save_result_for_eval': True,
                'shape_model_type': 'diffusion-gs-model-scene',
                'shape_model': {
                    'pretrained_model_name_or_path': checkpoint_path,
                    'width': 1024,
                    'in_channels': 9,
                    'patch_size': 8,
                    'n_gaussians': 2,
                    'dim_heads': 64,
                    'num_layers': 24,
                    'range_setting_near': 0,
                    'range_setting_far': 500,
                    'prior_distribution': 'gaussian',
                    'ray_pe_type': 'plk',
                    'use_flash': True,
                    'use_checkpoint': True,
                },
                'noise_scheduler_type': 'diffusionGS.models.scheduler.ddim_scheduler.DDIMScheduler',
                'noise_scheduler': {
                    'num_train_timesteps': 1000,
                    'prediction_type': 'sample',
                },
                'loss': {
                    'loss_type': 'mse',
                    'lambda_diffusion': 1.0,
                    'lambda_lpips': 0.1,
                    'lambda_ssim': 0.0,
                    'lambda_pointsdist': 0.0,
                    'lambda_xyz': 0.0,
                    'lambda_depth': 0.0,
                },
                'optimizer': {
                    'name': 'AdamW',
                    'args': {
                        'lr': 1.e-4,
                        'betas': [0.9, 0.99],
                        'eps': 1.e-6,
                    }
                },
            },
        }
        
        # Convert dict to OmegaConf and then to ExperimentConfig
        omega_cfg = OmegaConf.create(cfg_dict)
        OmegaConf.resolve(omega_cfg)
        assert isinstance(omega_cfg, DictConfig)
        cfg = parse_structured(ExperimentConfig, omega_cfg)
    else:
        # Update checkpoint path in config if it's already loaded
        try:
            if hasattr(cfg, 'system') and hasattr(cfg.system, 'shape_model'):
                cfg.system.shape_model.pretrained_model_name_or_path = checkpoint_path
            elif hasattr(cfg, 'system') and isinstance(cfg.system, dict):
                if 'shape_model' in cfg.system:
                    cfg.system['shape_model']['pretrained_model_name_or_path'] = checkpoint_path
            elif isinstance(cfg, dict) and 'system' in cfg:
                if 'shape_model' in cfg['system']:
                    cfg['system']['shape_model']['pretrained_model_name_or_path'] = checkpoint_path
        except Exception as e:
            print(f"Warning: Could not update checkpoint path in config: {e}")
    
    return cfg


def load_image_from_file(image_path: str, target_size=(256, 256), device='cuda'):
    """Load an image from file and preprocess it."""
    img = Image.open(image_path).convert('RGB')
    # Resize to target size
    img = img.resize(target_size, Image.LANCZOS)
    # Convert to tensor and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
    return img_tensor.to(device)


def create_batch_from_image(image_path: str, device='cuda', num_target_views=3):
    """Create a batch from a single input image for single-view reconstruction."""
    print(f"Loading input image from: {image_path}")
    h, w = 256, 256
    c = 3
    
    # Load the input image
    input_image = load_image_from_file(image_path, target_size=(h, w), device=device)
    # Shape: [C, H, W]
    
    batch_size = 1
    num_views = 1 + num_target_views  # 1 input + num_target_views target views
    
    # Create batch: [B, V, C, H, W]
    # First view is the input image, others are zeros (will be generated)
    rgbs_input = torch.zeros(batch_size, num_views, c, h, w, device=device)
    rgbs_input[0, 0] = input_image  # Set first view to input image
    
    # Create camera parameters
    # For single-view, we need to create reasonable camera poses
    # Default camera looking at origin from distance 2.0
    c2ws_input = torch.zeros(batch_size, num_views, 4, 4, device=device)
    for i in range(num_views):
        # Identity matrix for first view (input camera)
        if i == 0:
            c2w = torch.eye(4, device=device)
            c2w[:3, 3] = torch.tensor([0.0, 0.0, 2.0], device=device)  # Camera at z=2.0
        else:
            # Generate target views with slight rotations
            angle = (i - 1) * 0.3  # Small rotation angle
            c2w = torch.eye(4, device=device)
            # Rotate around y-axis
            c2w[0, 0] = np.cos(angle)
            c2w[0, 2] = np.sin(angle)
            c2w[2, 0] = -np.sin(angle)
            c2w[2, 2] = np.cos(angle)
            c2w[:3, 3] = torch.tensor([0.0, 0.0, 2.0], device=device)
        c2ws_input[0, i] = c2w
    
    # Create intrinsics (fx, fy, cx, cy)
    # Assuming a reasonable focal length
    fx = fy = h * 0.7  # Focal length proportional to image size
    cx, cy = w / 2.0, h / 2.0
    fxfycxcys_input = torch.tensor([[fx, fy, cx, cy]], device=device).unsqueeze(0).repeat(batch_size, num_views, 1)
    
    # Extract scene name from image path
    scene_name = Path(image_path).stem
    
    batch = {
        'rgbs_input': rgbs_input,
        'c2ws_input': c2ws_input,
        'fxfycxcys_input': fxfycxcys_input,
        'uid': [scene_name],
        'sel_idx': [0],
    }
    
    return batch


def create_dummy_batch(device='cuda'):
    """Create a dummy batch for testing when no data is available."""
    print("Creating dummy batch for testing...")
    batch_size = 1
    num_views = 4  # 1 input + 3 target views
    h, w = 256, 256
    c = 3
    
    # Create dummy images
    rgbs_input = torch.randn(batch_size, num_views, c, h, w).to(device)
    
    # Create dummy camera parameters
    c2ws_input = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1).to(device)
    # Add some variation to camera positions
    for i in range(num_views):
        c2ws_input[0, i, :3, 3] = torch.tensor([i * 0.1, 0, 2.0]).to(device)
    
    # Create dummy intrinsics (fx, fy, cx, cy)
    fxfycxcys_input = torch.tensor([[256.0, 256.0, 128.0, 128.0]]).unsqueeze(0).repeat(batch_size, num_views, 1).to(device)
    
    batch = {
        'rgbs_input': rgbs_input,
        'c2ws_input': c2ws_input,
        'fxfycxcys_input': fxfycxcys_input,
        'uid': ['dummy_scene_001'],
        'sel_idx': [0],
    }
    
    return batch


def run_single_iteration(system, batch, device='cuda', save_dir='./outputs/single_iteration'):
    """Run a single validation iteration."""
    system.eval()
    system = system.to(device)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    system.set_save_dir(save_dir)
    
    print("Running single iteration of scene reconstruction...")
    
    with torch.no_grad():
        # Transform input to get ray origins and directions
        ray_o, ray_d = TransformInput(
            batch['rgbs_input'],
            batch['c2ws_input'],
            batch['fxfycxcys_input']
        )
        
        input_batch = edict(
            image=batch['rgbs_input'][:, :1],  # First view is the input
            c2w=batch['c2ws_input'],
            fxfycxcy=batch['fxfycxcys_input'],
            ray_o=ray_o,
            ray_d=ray_d,
        )
        
        input_images = batch['rgbs_input']
        b, v, c, h, w = input_images.shape
        
        # Sample noise for the target views (views 1 onwards)
        sample_noise = torch.randn(b, v-1, c, h, w, device=device)
        input_batch["image_noisy"] = sample_noise
        
        image_condition = input_images[:, 0, ...].unsqueeze(1)
        
        # Get diffusion inference object
        num_inference_steps = getattr(system.cfg, 'num_inference_steps', 30)
        diffusion_inference = create_diffusion(str(num_inference_steps), predict_xstart=True)
        traj_timesteps = list(diffusion_inference.timestep_map)[::-1]
        traj_len = len(traj_timesteps)
        
        print(f"Running diffusion process with {traj_len} steps...")
        
        traj_samples = []
        traj_pred_xstart = []
        final_out = None
        
        # Run diffusion sampling loop
        for out in tqdm(
            diffusion_inference.p_sample_loop_progressive(
                system.shape_model,
                sample_noise.shape,
                input_batch,
                clip_denoised=False,
                progress=True,
                device=device,
            ),
            desc="Diffusion steps",
            total=traj_len
        ):
            samples = out["sample"]
            pred_xstart = out["pred_xstart"]
            
            # Concatenate image condition with samples
            samples_present = torch.cat((image_condition, samples), dim=1)
            pred_xstart_present = torch.cat((image_condition, pred_xstart), dim=1)
            
            traj_samples.append(samples_present)
            traj_pred_xstart.append(pred_xstart_present)
            final_out = out
        
        print("Diffusion process completed!")
        
        # Get final rendered images
        if final_out is not None and 'denoiser_output_dict' in final_out:
            render_images = final_out['denoiser_output_dict']['render_images']
            pred_gaussians = final_out['denoiser_output_dict']['pred_gaussians']
            
            print(f"Final render shape: {render_images.shape}")
            
            # Save results
            for i in range(len(batch['uid'])):
                uid = batch['uid'][i]
                sel_idx = batch['sel_idx'][i] if 'sel_idx' in batch else 0
                
                # Save rendered images
                output_image = torch.cat([image_condition[i], render_images[i]], dim=0)
                save_path = os.path.join(save_dir, f"{uid}_{sel_idx}.png")
                system.save_torch_images(save_path, output_image)
                print(f"Saved rendered image to {save_path}")
                
                # Save result package for evaluation
                save_result_for_eval = getattr(system.cfg, 'save_result_for_eval', True)
                if save_result_for_eval:
                    result_pkg = {
                        'render_images': render_images[i].detach().cpu(),
                        'image': batch['rgbs_input'][i].detach().cpu(),
                    }
                    pt_path = os.path.join(save_dir, f"{uid}.pt")
                    system.save_torch(pt_path, result_pkg)
                    print(f"Saved result package to {pt_path}")
                
                # Save gaussians if available
                if 'pred_gaussians' in final_out['denoiser_output_dict']:
                    ply_path = os.path.join(save_dir, f"{uid}_{sel_idx}.ply")
                    system.save_guassians_ply_scene(
                        ply_path,
                        pred_gaussians[i],
                        render_keyframe_c2ws=batch['c2ws_input'][i],
                        render_intrinsics=batch['fxfycxcys_input'][i],
                        render_video=False,
                        h=h,
                        w=w
                    )
                    print(f"Saved gaussians to {ply_path}")
        
        return final_out


def main():
    parser = argparse.ArgumentParser(
        description="Run a single iteration of Single-view Scene Reconstruction"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="https://huggingface.co/CaiYuanhao/DiffusionGS/blob/main/scene_ckpt_256.ckpt",
        help="Path to checkpoint file or HuggingFace URL"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to config YAML file (optional, will use defaults if not provided)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to data JSON file (optional, will use dummy data if not provided)"
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default=None,
        help="Path to input image file for single-view reconstruction (PNG, JPG, etc.)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/single_iteration",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)"
    )
    parser.add_argument(
        "--use_dummy_data",
        action="store_true",
        help="Use dummy data instead of loading from file or image"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Single-view Scene Reconstruction - Single Iteration")
    print("=" * 60)
    
    # Download checkpoint if needed
    print(f"\n1. Loading checkpoint from: {args.checkpoint_path}")
    checkpoint_path = download_checkpoint_if_needed(args.checkpoint_path)
    print(f"   Checkpoint loaded: {checkpoint_path}")
    
    # Load config
    print(f"\n2. Loading configuration...")
    cfg = load_config_with_checkpoint(args.config_path, checkpoint_path)
    print(f"   Config loaded successfully")
    
    # Initialize system
    print(f"\n3. Initializing system...")
    system = diffusionGS.find(cfg.system_type)(cfg.system, resumed=False)
    system.eval()
    print(f"   System initialized: {cfg.system_type}")
    
    # Load or create batch
    print(f"\n4. Preparing data...")
    if args.use_dummy_data:
        batch = create_dummy_batch(device=args.device)
        print(f"   Using dummy data (batch size: {batch['rgbs_input'].shape[0]})")
    elif args.input_image and os.path.exists(args.input_image):
        # Load from image file
        batch = create_batch_from_image(args.input_image, device=args.device, num_target_views=3)
        print(f"   Loaded image from: {args.input_image}")
        print(f"   Batch size: {batch['rgbs_input'].shape[0]}, Views: {batch['rgbs_input'].shape[1]}")
    elif args.data_path and os.path.exists(args.data_path):
        # Load data from JSON file
        with open(args.data_path, 'r') as f:
            data_json = json.load(f)
        # You would need to process this according to the dataset format
        # For now, we'll use dummy data
        print(f"   Data file found but processing not implemented, using dummy data")
        batch = create_dummy_batch(device=args.device)
    else:
        print("   No input image or data file provided. Using dummy data.")
        print("   To use a real image, provide --input_image <path_to_image>")
        batch = create_dummy_batch(device=args.device)
    
    # Run single iteration
    print(f"\n5. Running inference...")
    final_out = run_single_iteration(
        system,
        batch,
        device=args.device,
        save_dir=args.output_dir
    )
    
    print(f"\n6. Results saved to: {args.output_dir}")
    print("=" * 60)
    print("Single iteration completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

