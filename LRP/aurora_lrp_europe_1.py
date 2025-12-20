from __future__ import annotations
import argparse
import os
import torch
import torch.nn as nn
import xarray as xr
import pandas as pd
import numpy as np
from collections.abc import Mapping
from zennit.composites import LayerMapComposite
from zennit.rules import Epsilon, Pass
from zennit.types import Linear, Convolution
import torch.nn as nn
from datetime import timedelta
from aurora import AuroraSmallPretrained, Batch, Metadata
import torch.utils.checkpoint
import gc

# Mitigate fragmentation on CUDA
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True

VAR_MAP = {
    "2t": "2m_temperature", "10u": "10m_u_component_of_wind", "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure", "t": "temperature", "u": "u_component_of_wind",
    "v": "v_component_of_wind", "q": "specific_humidity", "z": "geopotential",
}

def nan_hook(grad):
    if torch.isnan(grad).any():
        return torch.nan_to_num(grad, nan=0.0)
    return grad

def compute_patch_resolution(model: AuroraSmallPretrained, batch: Batch) -> tuple[int, int, int]:
    H, W = batch.spatial_shape
    patch = model.patch_size
    if H % patch != 0 or W % patch != 0:
        raise ValueError(f"Spatial dims {(H, W)} not divisible by patch size {patch}")
    return (model.encoder.latent_levels, H // patch, W // patch)

def load_batch_from_zarr(zarr_path: str, static_path: str, date_str: str) -> Batch:
    ds = xr.open_zarr(zarr_path, consolidated=True)
    target_time = pd.to_datetime(f"{date_str}T06:00:00")
    request_times = [target_time - timedelta(hours=6), target_time]
    try:
        frame = ds.sel(time=request_times, method="nearest").load()
        frame = frame.sortby("time").isel(latitude=slice(0, 720))
        if frame.time.size < 2:
            raise ValueError(f"Only found {frame.time.size} time steps; need at least 2")
    except Exception as e:
        raise ValueError(f"Failed to load {date_str}: {e}")

    static = xr.open_dataset(static_path, engine="netcdf4")
    if "valid_time" in static.dims: static = static.isel(valid_time=0)
    static = static.interp(latitude=frame.latitude, longitude=frame.longitude)
    static = static.transpose("latitude", "longitude").isel(latitude=slice(0, 720))

    return Batch(
        surf_vars={
            "2t": torch.from_numpy(frame[VAR_MAP["2t"]].values).unsqueeze(0).float(),
            "10u": torch.from_numpy(frame[VAR_MAP["10u"]].values).unsqueeze(0).float(),
            "10v": torch.from_numpy(frame[VAR_MAP["10v"]].values).unsqueeze(0).float(),
            "msl": torch.from_numpy(frame[VAR_MAP["msl"]].values).unsqueeze(0).float(),
        },
        static_vars={
            "z": torch.from_numpy(static["z"].values[None, None]).float(),
            "slt": torch.from_numpy(static["slt"].values[None, None]).float(),
            "lsm": torch.from_numpy(static["lsm"].values[None, None]).float(),
        },
        atmos_vars={
            "t": torch.from_numpy(frame[VAR_MAP["t"]].values).unsqueeze(0).float(),
            "u": torch.from_numpy(frame[VAR_MAP["u"]].values).unsqueeze(0).float(),
            "v": torch.from_numpy(frame[VAR_MAP["v"]].values).unsqueeze(0).float(),
            "q": torch.from_numpy(frame[VAR_MAP["q"]].values).unsqueeze(0).float(),
            "z": torch.from_numpy(frame[VAR_MAP["z"]].values).unsqueeze(0).float(),
        },
        metadata=Metadata(
            lat=torch.from_numpy(frame.latitude.values),
            lon=torch.from_numpy(frame.longitude.values),
            time=tuple(pd.to_datetime(frame.time.values).to_pydatetime()),
            atmos_levels=tuple(int(lvl) for lvl in frame.level.values),
        ),
    )

class AuroraLatentWrapper(nn.Module):
    def __init__(self, model: AuroraSmallPretrained, original_batch: Batch, patch_res: tuple[int, int, int]):
        super().__init__()
        self.model = model
        self.static_vars = original_batch.static_vars
        self.metadata = original_batch.metadata
        self.lead_time = timedelta(hours=6)
        self._default_rollout = 0
        self._default_patch_res = patch_res
        self.n_levels = len(self.metadata.atmos_levels)

    def _parse_encoder_output(self, enc_out):
        if hasattr(enc_out, "tokens"): return enc_out.tokens
        elif isinstance(enc_out, Mapping): return enc_out["tokens"]
        elif isinstance(enc_out, (list, tuple)): return enc_out[0]
        return enc_out

    def forward(self, combined_tensor):
        # 1. Slice Surface Variables
        current_surf = {
            "2t": combined_tensor[:, :, 0:1].squeeze(2),
            "10u": combined_tensor[:, :, 1:2].squeeze(2),
            "10v": combined_tensor[:, :, 2:3].squeeze(2),
            "msl": combined_tensor[:, :, 3:4].squeeze(2),
        }
        
        # 2. Slice Atmospheric Variables
        L = self.n_levels
        start = 4
        current_atmos = {
            "t": combined_tensor[:, :, start : start+L],
            "u": combined_tensor[:, :, start+L : start + 2*L],
            "v": combined_tensor[:, :, start + 2*L : start + 3*L],
            "q": combined_tensor[:, :, start + 3*L : start + 4*L],
            "z": combined_tensor[:, :, start + 4*L : start + 5*L],
        }

        batch_reconstructed = Batch(
            surf_vars=current_surf,
            static_vars=self.static_vars,
            atmos_vars=current_atmos,
            metadata=self.metadata,
        ).to(combined_tensor.device)

        enc_out = self.model.encoder(batch_reconstructed, lead_time=self.lead_time)
        tokens = self._parse_encoder_output(enc_out)
        
        patch_res = self._default_patch_res
        
        if len(patch_res) != 3:
            raise RuntimeError(f"Patch resolution error: {patch_res}")
        spatial_area = patch_res[1] * patch_res[2]
        time_depth = tokens.shape[1] // spatial_area
        if time_depth != patch_res[0]:
            patch_res = (time_depth, patch_res[1], patch_res[2])
        expected = spatial_area * time_depth
        if tokens.shape[1] > expected:
            tokens = tokens[:, -expected:, :]
        
        def _backbone_fn(toks):
            return self.model.backbone(toks, self.lead_time, self._default_rollout, self._default_patch_res)

        latents = torch.utils.checkpoint.checkpoint(_backbone_fn, tokens, use_reentrant=False)
        return latents[-1] if isinstance(latents, (list, tuple)) else latents

def get_europe_mask(lat_grid, lon_grid, latent_shape):
    if len(latent_shape) == 4: _, _, H, W = latent_shape
    elif len(latent_shape) == 3: _, H, W = latent_shape
    else: raise ValueError(f"Unexpected latent shape: {latent_shape}")
    
    mask = torch.zeros((H, W))
    lat_indices = np.linspace(0, len(lat_grid) - 1, H, dtype=int)
    lon_indices = np.linspace(0, len(lon_grid) - 1, W, dtype=int)
    downsampled_lats = lat_grid[lat_indices]
    downsampled_lons = lon_grid[lon_indices]
    
    for i, lat in enumerate(downsampled_lats):
        for j, lon in enumerate(downsampled_lons):
            lon_norm = lon if lon <= 180 else lon - 360
            if 30.0 <= lat <= 72.0 and -12.0 <= lon_norm <= 45.0:
                mask[i, j] = 1.0
    return mask.to(device='cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr-path", type=str, required=True)
    parser.add_argument("--static-path", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Aurora...")
    model = AuroraSmallPretrained()
    model.load_checkpoint()
    model.eval()
    
    print(f"Loading data for {args.date}...")
    batch = load_batch_from_zarr(args.zarr_path, args.static_path, args.date)
    # Don't move full batch to GPU if not needed immediately, but we do for prediction
    batch = batch.to(device)
  
    # 1. Run Prediction (No Gradients)
    print("Running Aurora prediction...")
    batch_for_rollout = Batch(
        surf_vars=batch.surf_vars,
        static_vars={k: v.squeeze(0).squeeze(0) for k, v in batch.static_vars.items()},
        atmos_vars=batch.atmos_vars,
        metadata=batch.metadata,
    )
    # batch_transform_hook can be memory intensive, cropping helps
    p = next(model.parameters())
    batch_for_rollout = model.batch_transform_hook(batch_for_rollout).type(p.dtype).crop(model.patch_size).to(p.device)

    with torch.inference_mode():
        prediction = model.forward(batch_for_rollout)

    all_preds = {}
    if hasattr(prediction, 'surf_vars'):
        for var_name, tensor in prediction.surf_vars.items():
            all_preds[f"pred_{var_name}"] = tensor[:, -1:].detach().cpu().squeeze(1)
    if hasattr(prediction, 'atmos_vars'):
        for var_name, tensor in prediction.atmos_vars.items():
            all_preds[f"pred_{var_name}"] = tensor[:, -1:].detach().cpu().squeeze(1)

    del prediction, batch_for_rollout, batch
    if device.startswith("cuda"): torch.cuda.empty_cache()
    gc.collect()

    # 2. Prepare LRP Batch
    print("Preparing normalized batch for LRP...")
    raw_batch = load_batch_from_zarr(args.zarr_path, args.static_path, args.date)
    batch_lrp = model.batch_transform_hook(raw_batch)
    del raw_batch
    gc.collect()
    
    batch_lrp = batch_lrp.normalise(surf_stats=model.surf_stats).crop(model.patch_size)

    # Free up GPU memory by moving model to CPU for LRP part
    model.cpu() 
    torch.cuda.empty_cache()
    model = model.to(dtype=torch.float32)

    # 3. Construct Input Tensor
    surf_keys = ["2t", "10u", "10v", "msl"]
    atmos_keys = ["t", "u", "v", "q", "z"]
    
    patch_res = compute_patch_resolution(model, batch_lrp)

    # Unsqueeze surface tensors to make them 5D
    surf_tensors = [batch_lrp.surf_vars[k].unsqueeze(2) for k in surf_keys]
    atmos_tensors = [batch_lrp.atmos_vars[k] for k in atmos_keys]
    
    input_tensor = torch.cat(surf_tensors + atmos_tensors, dim=2).requires_grad_(True)
    input_tensor.register_hook(nan_hook)
    
    # Delete the redundant data copies and clear batch_lrp
    del surf_tensors
    del atmos_tensors
    
    # Clear large tensor data from batch_lrp
    batch_lrp.surf_vars = None
    batch_lrp.atmos_vars = None
    
    gc.collect()
    print("Redundant data deleted. Starting LRP...")

    wrapper = AuroraLatentWrapper(model, batch_lrp, patch_res)
    
    composite = LayerMapComposite(
        layer_map=[
            (Linear, Epsilon(epsilon=0.25)),
            (Convolution, Epsilon(epsilon=0.25)),
            (nn.LayerNorm, Pass()), 
            (nn.GELU, Pass()), 
            (nn.Dropout, Pass()),
            (nn.Softmax, Pass()), 
            (nn.Identity, Pass()),
        ]
    )
    
    print("Running LRP (on CPU)...")
    with composite.context(wrapper) as modified_model:
        latents = modified_model(input_tensor)
        if latents.ndim == 5: latents_spatial = latents.squeeze(2) 
        else: latents_spatial = latents
            
        lats = batch_lrp.metadata.lat.cpu().numpy()
        lons = batch_lrp.metadata.lon.cpu().numpy()
        # Ensure mask is created on CPU if running on CPU to avoid unnecessary transfers
        europe_mask = get_europe_mask(lats, lons, latents_spatial.shape)
        europe_mask = europe_mask.to(latents.device, dtype=latents.dtype)
        
        # Prepare the gradient tensor for the backward pass
        if latents.ndim == 5:
            relevance_target = europe_mask[None, None, None, :, :].expand_as(latents)
        elif latents.ndim == 4:
            relevance_target = europe_mask[None, None, :, :].expand_as(latents)
        elif latents.ndim == 3:
            relevance_target = europe_mask[None, :, :].expand_as(latents)
        
        torch.autograd.backward(latents, relevance_target)
        
        # Zennit computes relevance in the .grad attribute
        heatmap = input_tensor.grad
        
        # Free graph memory immediately
        del latents, relevance_target, latents_spatial
        gc.collect()
        
    # 4. Save Logic
    L = wrapper.n_levels
    start = 4
    
    save_dict = {
        "2t": heatmap[:, :, 0:1].squeeze(2).detach().cpu(),
        "10u": heatmap[:, :, 1:2].squeeze(2).detach().cpu(),
        "10v": heatmap[:, :, 2:3].squeeze(2).detach().cpu(),
        "msl": heatmap[:, :, 3:4].squeeze(2).detach().cpu(),
        
        "t": heatmap[:, :, start : start+L].detach().cpu(),
        "u": heatmap[:, :, start+L : start + 2*L].detach().cpu(),
        "v": heatmap[:, :, start + 2*L : start + 3*L].detach().cpu(),
        "q": heatmap[:, :, start + 3*L : start + 4*L].detach().cpu(),
        "z": heatmap[:, :, start + 4*L : start + 5*L].detach().cpu(),
        
        "lat": batch_lrp.metadata.lat.cpu(),
        "lon": batch_lrp.metadata.lon.cpu(),
        **all_preds
    }
   
    torch.save(save_dict, args.output)
    print(f"Saved LRP heatmaps for Europe target to {args.output}")

if __name__ == "__main__":
    main()
