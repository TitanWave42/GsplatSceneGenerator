#!/usr/bin/env python3
"""
Single-view Scene Reconstruction - Single Iteration Script

This script runs a single iteration of Single-view Scene Reconstruction using
the Open-DiffusionGS model. It loads the checkpoint from HuggingFace and
performs inference on a single batch of data.

Usage:
    python run_single_scene_reconstruction.py [--checkpoint_path PATH] [--config_path PATH] [--input_image PATH] [--input_images PATH1 PATH2 ...] [--data_path PATH]
    
    Examples:
    # Single image with default settings (256x256 checkpoint, 4 views, 100 inference steps)
    # Views are concentrated around input image with 30° spread
    python run_single_scene_reconstruction.py --input_image path/to/your/image.png
    
    # Concentrated views with custom angle spread (tighter around input)
    python run_single_scene_reconstruction.py --input_image path/to/your/image.png --angle_spread 15.0
    
    # Wider angle spread for more variation
    python run_single_scene_reconstruction.py --input_image path/to/your/image.png --angle_spread 45.0
    
    # Use distributed views around 360 degrees (original behavior)
    python run_single_scene_reconstruction.py --input_image path/to/your/image.png --distributed_views
    
    # If you get OOM errors, reduce views and inference steps
    python run_single_scene_reconstruction.py --input_image path/to/your/image.png --num_views 2 --reduce_inference_steps
    
    # High quality with 512x512 checkpoint (auto-reduces to 4 views to avoid OOM)
    python run_single_scene_reconstruction.py --input_image path/to/your/image.png --checkpoint_resolution 512
"""

import os
import sys
import argparse
import torch
import json
import numpy as np
import copy
from pathlib import Path
from PIL import Image
from easydict import EasyDict as edict
from einops import rearrange
from tqdm import tqdm

# Set PyTorch memory allocation config to help with fragmentation
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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


def load_config_with_checkpoint(config_path: str, checkpoint_path: str, resolution: int = 256, reduce_steps: bool = False):
    """Load config and update checkpoint path.
    
    Args:
        config_path: Path to config YAML file
        checkpoint_path: Path to checkpoint file
        resolution: Image resolution for quality settings
        reduce_steps: Whether to reduce inference steps for memory savings
    """
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
                'sel_views': 8,  # Increased default number of views
                'eval_subset': 1,
                'sel_views_train': 4,
                'training_res': [resolution, resolution],  # Use provided resolution
                'batch_size': 5,
                'eval_batch_size': 1,
                'num_workers': 0,
                'num_workers_val': 0,
            },
            'system_type': 'diffusion-gs-scene-system',
            'system': {
                'num_inference_steps': 50 if reduce_steps else 100,  # Configurable: 50 for memory, 100 for quality
                'save_intermediate_video': True,
                'save_result_for_eval': True,
                'shape_model_type': 'diffusion-gs-model-scene',
                'shape_model': {
                    'pretrained_model_name_or_path': checkpoint_path,
                    'width': 1024,
                    'in_channels': 9,
                    'patch_size': 8,
                    'n_gaussians': 2,  # Must match checkpoint: scene_ckpt_256.ckpt was trained with n_gaussians=2
                    'dim_heads': 64,
                    'num_layers': 24,
                    'range_setting_near': 0,
                    'range_setting_far': 500,
                    'prior_distribution': 'gaussian',
                    'ray_pe_type': 'plk',
                    'use_flash': True,  # Flash attention for memory efficiency
                    'use_checkpoint': True,  # Gradient checkpointing for memory efficiency
                    'grad_checkpoint_every': 1,  # Checkpoint every layer to save memory
                },
                'noise_scheduler_type': 'diffusionGS.models.scheduler.ddim_scheduler.DDIMScheduler',
                'noise_scheduler': {
                    'num_train_timesteps': 1000,
                    'prediction_type': 'sample',
                },
                'loss': {
                    'loss_type': 'mse',
                    'lambda_diffusion': 1.0,
                    'lambda_lpips': 0.15,  # Increased LPIPS weight for better perceptual quality
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


def generate_camera_poses(num_target_views: int, angle_spread_deg: float = 30.0, 
                         use_concentrated_views: bool = True, device='cuda'):
    """Generate camera poses for target views.
    
    This function controls how camera views are positioned relative to the input view.
    
    Args:
        num_target_views: Number of target views to generate
        angle_spread_deg: Maximum angle spread in degrees around the input view (default: 30.0)
                         Only used if use_concentrated_views=True
        use_concentrated_views: If True, views are concentrated around input view with small angles.
                               If False, views are distributed evenly around 360 degrees.
        device: Device to create tensors on
    
    Returns:
        c2ws: Camera-to-world matrices of shape [num_target_views, 4, 4]
    """
    c2ws = torch.zeros(num_target_views, 4, 4, device=device)
    
    for i in range(num_target_views):
        if use_concentrated_views:
            # Generate views concentrated around the input view
            angle_spread_rad = np.radians(angle_spread_deg)
            
            # Create angles distributed within the spread range
            if num_target_views == 1:
                angle_y = 0.0  # Single view, no rotation
                t = 0.5
            else:
                # Distribute angles evenly within the spread
                t = i / (num_target_views - 1)  # 0 to 1
                angle_y = (t - 0.5) * angle_spread_rad  # -spread/2 to +spread/2
            
            # Add small elevation variation (up/down tilt)
            elevation_spread = angle_spread_rad * 0.3  # 30% of horizontal spread
            elevation = (t - 0.5) * elevation_spread
            
            # Optional: Add small translation shifts (left/right, up/down)
            translation_scale = 0.05  # Small translation relative to camera distance
            tx = float(np.sin(angle_y) * translation_scale)  # Convert to Python float
            ty = float(np.sin(elevation) * translation_scale)  # Convert to Python float
        else:
            # Original behavior: distribute angles evenly around 360 degrees
            angle_y = i * (2 * np.pi / num_target_views)  # Azimuth angle (horizontal rotation)
            elevation = 0.0
            tx, ty = 0.0, 0.0
        
        # Create rotation matrix
        c2w = torch.eye(4, device=device)
        
        # Apply y-axis rotation (azimuth - left/right)
        cos_y = np.cos(angle_y)
        sin_y = np.sin(angle_y)
        c2w[0, 0] = cos_y
        c2w[0, 2] = sin_y
        c2w[2, 0] = -sin_y
        c2w[2, 2] = cos_y
        
        # Apply x-axis rotation (elevation - up/down)
        if elevation != 0.0:
            cos_e = np.cos(elevation)
            sin_e = np.sin(elevation)
            c2w_elev = torch.eye(4, device=device)
            c2w_elev[1, 1] = cos_e
            c2w_elev[1, 2] = -sin_e
            c2w_elev[2, 1] = sin_e
            c2w_elev[2, 2] = cos_e
            c2w = c2w_elev @ c2w
        
        # Position camera at distance 2.0 with optional translation
        base_pos = torch.tensor([0.0, 0.0, 2.0], device=device, dtype=torch.float32)
        # Apply translation in camera's local coordinate system
        translation = torch.tensor([tx, ty, 0.0], device=device, dtype=torch.float32)
        # Rotate translation to world coordinates
        translation_world = c2w[:3, :3] @ translation
        c2w[:3, 3] = base_pos + translation_world
        
        c2ws[i] = c2w
    
    return c2ws


def create_batch_from_image(image_path: str, device='cuda', num_target_views=8, resolution=256, 
                            angle_spread_deg=30.0, use_concentrated_views=True):
    """Create a batch from a single input image for single-view reconstruction.
    
    Args:
        image_path: Path to input image
        device: Device to load tensors on
        num_target_views: Number of target views to generate (default: 8 for more angles)
        resolution: Image resolution (default: 256, can be increased for higher quality)
        angle_spread_deg: Maximum angle spread in degrees around the input view (default: 30.0)
                         Only used if use_concentrated_views=True
        use_concentrated_views: If True, views are concentrated around input view with small angles.
                               If False, views are distributed evenly around 360 degrees.
    """
    print(f"Loading input image from: {image_path}")
    h, w = resolution, resolution
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
    # Default camera looking at origin from distance 2.0
    c2ws_input = torch.zeros(batch_size, num_views, 4, 4, device=device)
    
    # First view (input camera) - front view, looking down -Z axis
    c2w_input = torch.eye(4, device=device)
    c2w_input[:3, 3] = torch.tensor([0.0, 0.0, 2.0], device=device, dtype=torch.float32)
    c2ws_input[0, 0] = c2w_input
    
    # Generate target view camera poses using the exposed function
    target_c2ws = generate_camera_poses(
        num_target_views=num_target_views,
        angle_spread_deg=angle_spread_deg,
        use_concentrated_views=use_concentrated_views,
        device=device
    )
    
    # Assign target views to the batch
    c2ws_input[0, 1:] = target_c2ws
    
    view_type = "concentrated" if use_concentrated_views else "distributed"
    if use_concentrated_views:
        print(f"   Camera views: {view_type} (spread: {angle_spread_deg}° around input view)")
    else:
        print(f"   Camera views: {view_type} (360° distribution)")
    
    # Create intrinsics (fx, fy, cx, cy)
    # Assuming a reasonable focal length (higher for better quality)
    fx = fy = h * 0.8  # Increased focal length for better quality
    cx, cy = w / 2.0, h / 2.0
    fxfycxcys_input = torch.tensor([[fx, fy, cx, cy]], device=device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, num_views, 1)
    
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
        c2ws_input[0, i, :3, 3] = torch.tensor([i * 0.1, 0, 2.0], dtype=torch.float32).to(device)
    
    # Create dummy intrinsics (fx, fy, cx, cy)
    fxfycxcys_input = torch.tensor([[256.0, 256.0, 128.0, 128.0]], dtype=torch.float32).unsqueeze(0).repeat(batch_size, num_views, 1).to(device)
    
    batch = {
        'rgbs_input': rgbs_input,
        'c2ws_input': c2ws_input,
        'fxfycxcys_input': fxfycxcys_input,
        'uid': ['dummy_scene_001'],
        'sel_idx': [0],
    }
    
    return batch


def run_single_iteration(system, batch, device='cuda', save_dir='./outputs/single_iteration', 
                         opacity_threshold=0.05, no_prune=False):
    """Run a single validation iteration.
    
    Args:
        system: The diffusion system
        batch: Input batch data
        device: Device to run on
        save_dir: Directory to save results
        opacity_threshold: Opacity threshold for pruning gaussians (lower = denser)
        no_prune: If True, save without pruning (maximum density)
    """
    system.eval()
    system = system.to(device)
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    system.set_save_dir(save_dir)
    
    print("Running single iteration of scene reconstruction...")
    
    # Clear cache before starting
    torch.cuda.empty_cache()
    
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
        
        # Clear cache after diffusion
        torch.cuda.empty_cache()
        
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
                    ply_path = f"{uid}_{sel_idx}.ply"
                    
                    # Create a copy of gaussians to avoid modifying the original
                    gaussians_to_save = copy.deepcopy(pred_gaussians[i])
                    
                    # Count initial gaussians
                    initial_count = len(gaussians_to_save._xyz)
                    
                    # Apply custom opacity threshold if specified
                    if not no_prune:
                        # Prune with custom threshold
                        gaussians_to_save = gaussians_to_save.prune(opacity_thres=opacity_threshold)
                        pruned_count = len(gaussians_to_save._xyz)
                        print(f"   Pruned gaussians: {initial_count} -> {pruned_count} (opacity threshold: {opacity_threshold})")
                    else:
                        print(f"   Saving all gaussians without pruning (maximum density: {initial_count} gaussians)")
                    
                    # Save the gaussians directly (bypassing system's save which has hardcoded pruning)
                    save_path = system.get_save_path(ply_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    gaussians_to_save.save_ply(save_path)
                    
                    # Also save unpruned version if pruning was applied (for comparison)
                    if not no_prune:
                        unpruned_path = system.get_save_path(ply_path.replace('.ply', '_unpruned.ply'))
                        os.makedirs(os.path.dirname(unpruned_path), exist_ok=True)
                        pred_gaussians[i].save_ply(unpruned_path)
                        print(f"   Also saved unpruned version ({initial_count} gaussians) to: {unpruned_path}")
                    
                    print(f"Saved gaussians to {save_path}")
                    print(f"   Final gaussian count: {len(gaussians_to_save._xyz)}")
        
        # Final cache clear
        torch.cuda.empty_cache()
        
        return final_out


def main():
    parser = argparse.ArgumentParser(
        description="Run a single iteration of Single-view Scene Reconstruction"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint file or HuggingFace URL (if not provided, uses --checkpoint_resolution to select)"
    )
    parser.add_argument(
        "--checkpoint_resolution",
        type=int,
        choices=[256, 512],
        default=256,
        help="Checkpoint resolution to use: 256 (faster) or 512 (higher quality). Default: 256"
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
        "--input_images",
        type=str,
        nargs='+',
        default=None,
        help="Multiple input image paths for batch processing"
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=None,
        help="Number of target views to generate (default: auto-adjusted based on resolution)"
    )
    parser.add_argument(
        "--max_views_512",
        type=int,
        default=4,
        help="Maximum number of views for 512x512 resolution (default: 4, to avoid OOM)"
    )
    parser.add_argument(
        "--max_views_256",
        type=int,
        default=4,
        help="Maximum number of views for 256x256 resolution (default: 4, conservative to avoid OOM)"
    )
    parser.add_argument(
        "--reduce_inference_steps",
        action="store_true",
        help="Reduce inference steps from 100 to 50 to save memory (slightly lower quality)"
    )
    parser.add_argument(
        "--angle_spread",
        type=float,
        default=30.0,
        help="Maximum angle spread in degrees around the input view (default: 30.0). "
             "Smaller values create views closer to the input image."
    )
    parser.add_argument(
        "--distributed_views",
        action="store_true",
        help="Use distributed views around 360 degrees instead of concentrated around input view"
    )
    parser.add_argument(
        "--opacity_threshold",
        type=float,
        default=0.05,
        help="Opacity threshold for pruning gaussians when saving .ply file (default: 0.05). "
             "Lower values (e.g., 0.01) = denser point clouds, higher values (e.g., 0.1) = sparser"
    )
    parser.add_argument(
        "--no_prune",
        action="store_true",
        help="Save .ply file without pruning (maximum density, includes all gaussians)"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Image resolution (default: matches checkpoint resolution - 256 or 512)"
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
    
    # Set default checkpoint path based on resolution if not provided
    if args.checkpoint_path is None:
        if args.checkpoint_resolution == 512:
            args.checkpoint_path = "https://huggingface.co/CaiYuanhao/DiffusionGS/blob/main/scene_ckpt_512.ckpt"
        else:
            args.checkpoint_path = "https://huggingface.co/CaiYuanhao/DiffusionGS/blob/main/scene_ckpt_256.ckpt"
    else:
        # Detect checkpoint resolution from path if provided
        if "512" in args.checkpoint_path and "256" not in args.checkpoint_path.split("512")[0]:
            args.checkpoint_resolution = 512
        elif "256" in args.checkpoint_path:
            args.checkpoint_resolution = 256
    
    # Auto-set resolution to match checkpoint if not specified
    if args.resolution is None:
        args.resolution = args.checkpoint_resolution
        print(f"Auto-setting resolution to {args.resolution} to match checkpoint ({args.checkpoint_resolution}x{args.checkpoint_resolution})")
    
    # Auto-adjust number of views based on resolution to avoid OOM
    # Conservative defaults to prevent OOM on smaller GPUs
    if args.num_views is None:
        if args.resolution == 512:
            args.num_views = args.max_views_512
            print(f"Auto-setting num_views to {args.num_views} for 512x512 resolution (to avoid OOM)")
        else:
            # Use max_views_256 for 256 resolution
            args.num_views = args.max_views_256
            print(f"Auto-setting num_views to {args.num_views} for 256x256 resolution (conservative to avoid OOM)")
    elif args.resolution == 512 and args.num_views > args.max_views_512:
        print(f"Warning: {args.num_views} views with 512x512 resolution may cause OOM.")
        print(f"         Reducing to {args.max_views_512} views. Use --max_views_512 to override.")
        args.num_views = args.max_views_512
    elif args.resolution == 256 and args.num_views > args.max_views_256:
        print(f"Warning: {args.num_views} views with 256x256 resolution may cause OOM on smaller GPUs.")
        print(f"         Consider reducing to {args.max_views_256} views or use --max_views_256 to override.")
        if args.num_views > args.max_views_256 + 2:  # Only auto-reduce if way over
            print(f"         Auto-reducing to {args.max_views_256} views.")
            args.num_views = args.max_views_256
    
    # Memory warning and optimization suggestions
    total_views = args.num_views + 1  # +1 for input view
    estimated_memory_gb = (total_views * args.resolution * args.resolution * 4) / (1024**3) * (50 if args.reduce_inference_steps else 100)  # Rough estimate
    print(f"\n⚠️  Memory Info: {args.resolution}x{args.resolution} with {total_views} total views ({args.num_views} target + 1 input)")
    print(f"   Estimated peak memory: ~{estimated_memory_gb:.1f} GB (may vary)")
    if estimated_memory_gb > 6:
        print(f"   ⚠️  High memory usage detected! If you encounter OOM errors, try:")
        print(f"   - Reducing --num_views (current: {args.num_views}, try 2-3)")
        print(f"   - Using --reduce_inference_steps (reduces from 100 to 50 steps)")
        print(f"   - Setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print(f"   - Processing one image at a time\n")
    else:
        print(f"   Memory usage should be manageable.\n")
    
    print("=" * 60)
    print("Single-view Scene Reconstruction - Single Iteration")
    print("=" * 60)
    
    # Download checkpoint if needed
    print(f"\n1. Loading checkpoint...")
    print(f"   Checkpoint resolution: {args.checkpoint_resolution}x{args.checkpoint_resolution}")
    print(f"   Checkpoint URL: {args.checkpoint_path}")
    checkpoint_path = download_checkpoint_if_needed(args.checkpoint_path)
    print(f"   Checkpoint loaded: {checkpoint_path}")
    
    # Verify resolution matches checkpoint
    if "512" in checkpoint_path and args.resolution != 512:
        print(f"   Warning: Checkpoint is 512x512 but resolution is set to {args.resolution}. Updating to 512.")
        args.resolution = 512
    elif "256" in checkpoint_path and args.resolution != 256 and "512" not in checkpoint_path:
        print(f"   Warning: Checkpoint is 256x256 but resolution is set to {args.resolution}. Updating to 256.")
        args.resolution = 256
    
    # Load config
    print(f"\n2. Loading configuration...")
    cfg = load_config_with_checkpoint(args.config_path, checkpoint_path, resolution=args.resolution, reduce_steps=args.reduce_inference_steps)
    print(f"   Config loaded successfully")
    checkpoint_type = "512x512 (High Quality)" if args.resolution == 512 else "256x256 (Standard)"
    num_steps = 50 if args.reduce_inference_steps else 100
    print(f"   Quality settings: {checkpoint_type} checkpoint, {args.num_views} views, {args.resolution}x{args.resolution} resolution, {num_steps} inference steps")
    
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
    elif args.input_images and len(args.input_images) > 0:
        # Batch processing multiple images
        print(f"   Processing {len(args.input_images)} images in batch...")
        batches = []
        for img_path in args.input_images:
            if os.path.exists(img_path):
                img_batch = create_batch_from_image(
                    img_path, 
                    device=args.device, 
                    num_target_views=args.num_views,
                    resolution=args.resolution,
                    angle_spread_deg=args.angle_spread,
                    use_concentrated_views=not args.distributed_views
                )
                batches.append(img_batch)
            else:
                print(f"   Warning: Image not found: {img_path}, skipping...")
        
        if len(batches) > 0:
            # Combine batches
            batch = {
                'rgbs_input': torch.cat([b['rgbs_input'] for b in batches], dim=0),
                'c2ws_input': torch.cat([b['c2ws_input'] for b in batches], dim=0),
                'fxfycxcys_input': torch.cat([b['fxfycxcys_input'] for b in batches], dim=0),
                'uid': [uid for b in batches for uid in b['uid']],
                'sel_idx': [idx for b in batches for idx in b['sel_idx']],
            }
            print(f"   Batch size: {batch['rgbs_input'].shape[0]}, Views per image: {batch['rgbs_input'].shape[1]}")
        else:
            print("   No valid images found, using dummy data")
            batch = create_dummy_batch(device=args.device)
    elif args.input_image and os.path.exists(args.input_image):
        # Load from single image file
        batch = create_batch_from_image(
            args.input_image, 
            device=args.device, 
            num_target_views=args.num_views,
            resolution=args.resolution,
            angle_spread_deg=args.angle_spread,
            use_concentrated_views=not args.distributed_views
        )
        print(f"   Loaded image from: {args.input_image}")
        print(f"   Batch size: {batch['rgbs_input'].shape[0]}, Views: {batch['rgbs_input'].shape[1]}")
        print(f"   Resolution: {args.resolution}x{args.resolution}")
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
        print("   To process multiple images, provide --input_images <path1> <path2> ...")
        batch = create_dummy_batch(device=args.device)
    
    # Run single iteration
    print(f"\n5. Running inference...")
    final_out = run_single_iteration(
        system,
        batch,
        device=args.device,
        save_dir=args.output_dir,
        opacity_threshold=args.opacity_threshold,
        no_prune=args.no_prune
    )
    
    print(f"\n6. Results saved to: {args.output_dir}")
    print("=" * 60)
    print("Single iteration completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

