#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    	: 08 / 08 / 2022
@Author  	: Zhi Qiao
@Contact	: z.qiao1989@gmail.com
@File    	: WSVT.py
@Software	: WaveletSBI
@Desc		: scanning mode wavelet speckle tracking (WaveletSBI) script

1. Qiao, Zhi, Xianbo Shi, and Lahsen Assoufid. “Single-Shot Speckle Tracking Method Based on Wavelet Transform and Multi-Resolution Analysis.” In Advances in Metrology for X-Ray and EUV Optics IX, edited by Lahsen Assoufid, Haruhiko Ohashi, and Anand Asundi, 22. Online Only, United States: SPIE, 2020. https://doi.org/10.1117/12.2569135.
2. Qiao, Zhi, Xianbo Shi, Rafael Celestre, and Lahsen Assoufid. “Wavelet-Transform-Based Speckle Vector Tracking Method for X-Ray Phase Imaging.” Optics Express 28, no. 22 (October 26, 2020): 33053. https://doi.org/10.1364/OE.404606.

'''

import numpy as np
import pywt
import os
import sys
import argparse
import time
from PIL import Image
import glob
from .gui_func import crop_gui
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import scipy.constants as sc
from .func import prColor, frankotchellappa, Wavelet_transform_multiprocess, write_h5, write_json, find_disp, image_roi, WaveletTransformGPU
from .euclidean_dist import dist_numba
from matplotlib import pyplot as plt
from matplotlib import cm
import multiprocessing as ms
import concurrent.futures
import scipy.interpolate as sfit
import cv2
import ctypes
from numpy.ctypeslib import ndpointer

from .backends import choose_c_wavelet_backend, load_c_solver

c_lib, USE_C_SOLVER = load_c_solver()


def load_images(Folder_path, filename_format='*.tif'):
    """
    load_images to load images

    Args:
        Folder_path (str): image folder
        filename_format (str, optional): image filename and extension. Defaults to '*.tif'.

    Returns:
        image data
    """
    f_list = glob.glob(os.path.join(Folder_path, filename_format))
    f_list = sorted(f_list)
    # print(f_list)
    img = []
    for f_single in f_list:
        img.append(np.array(Image.open(f_single)))
        # prColor('load image: {}'.format(f_single), 'green')
    if len(img) == 0:
        prColor(
            'Error: wrong data path. No data is loaded. {}'.format(
                Folder_path), 'red')
        sys.exit()

    return np.array(img)

class ImageDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        return np.array(Image.open(self.file_paths[idx]))

def load_image(file_path):
    """
    load_images to load images

    Args:
        file_path (str): path to each img, single or list
        
    Returns:
        image data
    """
    
    if isinstance(file_path, list):
        # Use DataLoader for parallel loading
        dataset = ImageDataset(file_path)
        # num_workers=min(8, len(file_path)) to avoid too many workers for few files
        # pin_memory=True for faster transfer to CUDA if needed later, but here we return numpy
        loader = DataLoader(dataset, batch_size=1, num_workers=min(8, os.cpu_count()), shuffle=False)
        
        img = []
        for batch in loader:
            # batch is [1, H, W]
            img.append(batch.squeeze(0).numpy())
            
        if len(img) == 0:
            prColor(
                'Error: wrong data path. No data is loaded. {}'.format(
                    file_path), 'red')
            sys.exit()
    else:
        img = np.array(Image.open(file_path))

    return np.array(img).astype(np.float32)

def img_save(folder_path, filename, img):
    if not os.path.exists(os.path.join(folder_path)):
        os.makedirs(folder_path)
    # np.savetxt(os.path.join(folder_path, filename+'.npy.gz'), img)
    # np.save(os.path.join(folder_path, filename+'.npy'), img)
    # print(img.shape)

    im = Image.fromarray(img)
    im.save(os.path.join(folder_path, filename+'.tif'), save_all=True)

def slope_tracking(img, ref, n_window=15):
    """
    slope_tracking to use opencv to track the displacement movement roughly

    Args:
        img (ndarray): sample image
        ref (ndarray): ref image
        n_window (int, optional): window size. Defaults to 15.

    Returns:
        the displacement of the pixels in the images  [dips_H, disp_V]
    """
    # the pyramid scale, make the undersampling image
    pyramid_scal = 0.5
    # the pyramid levels
    levels = 2
    # window size of the displacement calculation
    winsize = n_window
    # iteration for the calculation
    n_iter = 10
    # neighborhood pixel size to calculate the polynomial expansion, which makes the results smooth but blurred
    n_poly = 3
    # standard deviation of the Gaussian that is used to smooth derivatives used as a basis for
    # the polynomial expansion; for poly_n=5, you can set poly_sigma=1.1, for poly_n=7, a good value would be poly_sigma=1.5.
    sigma_poly = 1.2
    flags = 1

    flow = cv2.calcOpticalFlowFarneback(ref, img, None, pyramid_scal, levels,
                                        winsize, n_iter, n_poly, sigma_poly,
                                        flags)

    displace = np.array([flow[..., 1], flow[..., 0]])

    return displace


def cost_volume_spatial(first, second, search_range, use_ncc=False):
    """Build cost volume using spatial correlation (memory-efficient chunked processing).
    Best for small search ranges (< 15 pixels).
    Args:
        first: Level of the feature pyramid of Image1 [C, H, W]
        second: Warped level of the feature pyramid of image2 [C, H, W]
        search_range: Search range (maximum displacement)
        use_ncc: If True, use Normalized Cross-Correlation (more robust to intensity variations)
    Returns:
        cost_vol: [max_offset*max_offset, H, W]
    """
    C, h, w = first.shape
    max_offset = search_range * 2 + 1
    
    # For NCC, normalize both images globally (once) - NOT per-shift
    # This avoids position-dependent bias that causes second-order envelope artifacts
    if use_ncc:
        first_f32 = first.float()
        # Normalize per-pixel across channels
        first_mean = torch.mean(first_f32, dim=0, keepdim=True)
        first_centered = first_f32 - first_mean
        first_std = torch.std(first_f32, dim=0, keepdim=True, correction=0)
        first_std = torch.clamp(first_std, min=1e-6)
        first_norm = first_centered / first_std
        
        # Normalize second image globally (before padding)
        second_f32 = second.float()
        second_mean = torch.mean(second_f32, dim=0, keepdim=True)
        second_centered = second_f32 - second_mean
        second_std = torch.std(second_f32, dim=0, keepdim=True, correction=0)
        second_std = torch.clamp(second_std, min=1e-6)
        second_norm = second_centered / second_std
    else:
        first_norm = first
        second_norm = second
    
    # Pad the second tensor (normalized if NCC, original otherwise)
    padded_lvl = nn.functional.pad(
        second_norm, (search_range, search_range, search_range, search_range))
    
    # Pre-allocate output tensor - always use float32 to match CPU precision
    cost_vol = torch.empty((max_offset * max_offset, h, w), 
                           device=first.device, dtype=torch.float32)
    
    # Memory optimization: Process correlation in chunks of channels if C is large
    # For C=931, H=1500, W=1500, full product is 4.2GB.
    chunk_size = 200
    num_chunks = (C + chunk_size - 1) // chunk_size
    
    # Process iteratively to save memory
    idx = 0
    for y in range(max_offset):
        for x in range(max_offset):
            shifted = padded_lvl[:, y:y + h, x:x + w]
            
            if num_chunks == 1:
                if use_ncc:
                    # NCC: Mean product of normalized vectors
                    corr = torch.mean(first_norm * shifted, dim=0)
                else:
                    # SSD: Negative Sum of Squared Differences
                    # Match CPU implementation (dist_numba)
                    diff = first_norm - shifted
                    ssd = torch.sum(diff * diff, dim=0)
                    corr = -ssd
            else:
                corr_sum = torch.zeros((h, w), device=first.device, dtype=torch.float32)
                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min((i + 1) * chunk_size, C)
                    
                    f_chunk = first_norm[start:end]
                    s_chunk = shifted[start:end]
                    
                    if use_ncc:
                        corr_sum += torch.sum(f_chunk.float() * s_chunk.float(), dim=0)
                    else:
                        diff = f_chunk.float() - s_chunk.float()
                        corr_sum += torch.sum(diff * diff, dim=0)
                
                # Always use float32 to match CPU precision
                if use_ncc:
                    corr = corr_sum / C
                else:
                    corr = -corr_sum
            
            cost_vol[idx] = corr
            idx += 1
    
    return cost_vol


def cost_volume_spatial_centered(first, second, search_range, center_y, center_x):
    """SSD cost volume centered on an integer displacement for every pixel.

    This matches ``calculate_displacement_block``: all candidate offsets for a
    target pixel use that target pixel's own coarse displacement estimate.
    Warping ``second`` first is not equivalent when the estimate varies across
    space, because shifted candidates then sample neighboring estimates.
    """
    C, h, w = first.shape
    window_size = 2 * search_range + 1
    center_y = torch.round(center_y).to(device=first.device, dtype=torch.long)
    center_x = torch.round(center_x).to(device=first.device, dtype=torch.long)
    base_y = torch.arange(h, device=first.device, dtype=torch.long).view(h, 1)
    base_x = torch.arange(w, device=first.device, dtype=torch.long).view(1, w)
    second_flat = second.float().reshape(C, h * w)
    first_f32 = first.float()
    cost_vol = torch.empty(
        (window_size * window_size, h, w), device=first.device, dtype=torch.float32)

    chunk_size = 200
    num_chunks = (C + chunk_size - 1) // chunk_size
    out_index = 0
    for oy in range(window_size):
        offset_y = oy - search_range
        sample_y = base_y + center_y + offset_y
        for ox in range(window_size):
            offset_x = ox - search_range
            sample_x = base_x + center_x + offset_x
            valid = (sample_y >= 0) & (sample_y < h) & (sample_x >= 0) & (sample_x < w)
            flat_index = (
                torch.clamp(sample_y, 0, h - 1) * w + torch.clamp(sample_x, 0, w - 1)
            ).reshape(1, h * w)

            ssd = torch.zeros((h, w), device=first.device, dtype=torch.float32)
            for chunk in range(num_chunks):
                start = chunk * chunk_size
                end = min((chunk + 1) * chunk_size, C)
                gathered = torch.gather(
                    second_flat[start:end], 1, flat_index.expand(end - start, -1)
                ).reshape(end - start, h, w)
                diff = first_f32[start:end] - gathered
                ssd += torch.sum(diff * diff, dim=0)
            cost_vol[out_index] = torch.where(valid, -ssd, torch.full_like(ssd, -1e30))
            out_index += 1

    return cost_vol


def cost_volume_fft(first, second, search_range):
    """Build cost volume using FFT-based cross-correlation.
    More efficient for large search ranges (>= 15 pixels).
    Args:
        first: Level of the feature pyramid of Image1 [C, H, W]
        second: Warped level of the feature pyramid of image2 [C, H, W]
        search_range: Search range (maximum displacement)
    Returns:
        cost_vol: [max_offset*max_offset, H, W]
    """
    C, h, w = first.shape
    max_offset = search_range * 2 + 1
    
    # Compute FFT size (needs to be at least h + max_offset - 1, w + max_offset - 1)
    fft_h = h + max_offset - 1
    fft_w = w + max_offset - 1
    
    # Pad to power of 2 for efficiency (optional but usually faster)
    fft_h = 2 ** int(np.ceil(np.log2(fft_h)))
    fft_w = 2 ** int(np.ceil(np.log2(fft_w)))
    
    # FFT of first (use float32 for FFT)
    first_f32 = first.float()
    second_f32 = second.float()
    
    # Compute FFT
    first_fft = torch.fft.rfft2(first_f32, s=(fft_h, fft_w))
    second_fft = torch.fft.rfft2(second_f32, s=(fft_h, fft_w))
    
    # Cross-correlation in frequency domain: conj(first) * second
    # Sum over channels
    cross_corr_fft = torch.sum(torch.conj(first_fft) * second_fft, dim=0)
    
    # Inverse FFT
    cross_corr = torch.fft.irfft2(cross_corr_fft, s=(fft_h, fft_w))
    
    # Extract the relevant region
    # The cross-correlation peak at (0,0) means no shift
    # We need to extract a (max_offset, max_offset) region around the center
    # Due to FFT shift properties, we need to handle the wrap-around
    
    # Pre-allocate output
    cost_vol = torch.empty((max_offset * max_offset, h, w), 
                           device=first.device, dtype=first.dtype)
    
    # For each output position (oy, ox) in search window:
    # We want correlation at shift (oy - search_range, ox - search_range)
    # In FFT output, shift (dy, dx) appears at position (dy % fft_h, dx % fft_w)
    for oy in range(max_offset):
        for ox in range(max_offset):
            dy = oy - search_range
            dx = ox - search_range
            # Handle negative indices with modulo
            src_y = dy % fft_h
            src_x = dx % fft_w
            # Extract the correlation values for all (h, w) positions
            # This is a simplification - for full correctness we'd need to handle
            # boundary effects, but for large images this approximation works well
            cost_vol[oy * max_offset + ox] = cross_corr[src_y:src_y + h, src_x:src_x + w].to(first.dtype) / C
    
    return cost_vol


def cost_volume(first, second, search_range, use_fft=False, use_ncc=False):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
    Args:
        first: Level of the feature pyramid of Image1 [C, H, W]
        second: Warped level of the feature pyramid of image2 [C, H, W]
        search_range: Search range (maximum displacement)
        use_fft: If True, use FFT-based correlation (better for large search ranges)
        use_ncc: If True, use Normalized Cross-Correlation (more robust)
    Returns:
        cost_vol: [max_offset*max_offset, H, W]
    """
    if use_fft:
        return cost_volume_fft(first, second, search_range)
    else:
        return cost_volume_spatial(first, second, search_range, use_ncc=use_ncc)


def cost_volume_peak_streaming(first, second, search_range, use_ncc=False):
    """Memory-efficient peak search without materializing full cost volume."""
    C, h, w = first.shape
    max_offset = search_range * 2 + 1

    if use_ncc:
        first_f32 = first.float()
        first_mean = torch.mean(first_f32, dim=0, keepdim=True)
        first_std = torch.std(first_f32, dim=0, keepdim=True, correction=0).clamp(min=1e-6)
        first_norm = (first_f32 - first_mean) / first_std

        second_f32 = second.float()
        second_mean = torch.mean(second_f32, dim=0, keepdim=True)
        second_std = torch.std(second_f32, dim=0, keepdim=True, correction=0).clamp(min=1e-6)
        second_norm = (second_f32 - second_mean) / second_std
    else:
        first_norm = first
        second_norm = second

    padded_lvl = nn.functional.pad(
        second_norm, (search_range, search_range, search_range, search_range))

    best_val = torch.full((h, w), -float('inf'), device=first.device, dtype=torch.float32)
    best_idx = torch.zeros((h, w), device=first.device, dtype=torch.long)

    chunk_size = 200
    num_chunks = (C + chunk_size - 1) // chunk_size

    idx = 0
    for y in range(max_offset):
        for x in range(max_offset):
            shifted = padded_lvl[:, y:y + h, x:x + w]
            if num_chunks == 1:
                if use_ncc:
                    corr = torch.mean(first_norm * shifted, dim=0)
                else:
                    diff = first_norm - shifted
                    corr = -torch.sum(diff * diff, dim=0)
            else:
                corr_sum = torch.zeros((h, w), device=first.device, dtype=torch.float32)
                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min((i + 1) * chunk_size, C)
                    f_chunk = first_norm[start:end].float()
                    s_chunk = shifted[start:end].float()
                    if use_ncc:
                        corr_sum += torch.sum(f_chunk * s_chunk, dim=0)
                    else:
                        diff = f_chunk - s_chunk
                        corr_sum += torch.sum(diff * diff, dim=0)
                corr = corr_sum / C if use_ncc else -corr_sum

            update_mask = corr > best_val
            best_val = torch.where(update_mask, corr, best_val)
            best_idx = torch.where(update_mask, torch.full_like(best_idx, idx), best_idx)
            idx += 1

    return best_val, best_idx


def filter_displacement_outliers(disp_y, disp_x, kernel_size=5, threshold=2.0):
    """Filter outliers in displacement maps using median filtering.
    Replaces outlier pixels with local median values.
    Args:
        disp_y, disp_x: Displacement maps [H, W]
        kernel_size: Size of median filter kernel
        threshold: Number of MAD (median absolute deviation) to consider as outlier
    Returns:
        filtered_disp_y, filtered_disp_x, outlier_mask
    """
    import scipy.ndimage as ndi
    
    # Compute median filtered versions
    median_y = ndi.median_filter(disp_y, size=kernel_size)
    median_x = ndi.median_filter(disp_x, size=kernel_size)
    
    # Compute deviation from median
    diff_y = np.abs(disp_y - median_y)
    diff_x = np.abs(disp_x - median_x)
    
    # Compute MAD (median absolute deviation) for robust threshold
    mad_y = np.median(diff_y)
    mad_x = np.median(diff_x)
    
    # Identify outliers (pixels that deviate more than threshold * MAD)
    outlier_mask = (diff_y > threshold * (mad_y + 0.1)) | (diff_x > threshold * (mad_x + 0.1))
    
    # Replace outliers with median values
    filtered_y = np.where(outlier_mask, median_y, disp_y)
    filtered_x = np.where(outlier_mask, median_x, disp_x)
    
    n_outliers = np.sum(outlier_mask)
    if n_outliers > 0:
        prColor(f'Filtered {n_outliers} outlier pixels ({100*n_outliers/outlier_mask.size:.2f}%)', 'yellow')
    
    return filtered_y, filtered_x, outlier_mask


def _subpixel_refinement_parabolic(cost_vol, max_idx, max_val, window_size, cal_half_window):
    """Sub-pixel refinement using parabolic (2nd order polynomial) fitting.
    Classic method, fast but may have slight bias for non-symmetric peaks.
    """
    # Convert idx to wy, wx
    max_wy = torch.div(max_idx, window_size, rounding_mode='floor')
    max_wx = max_idx % window_size
    
    def get_val(offset_y, offset_x):
        yy = torch.clamp(max_wy + offset_y, 0, window_size - 1)
        xx = torch.clamp(max_wx + offset_x, 0, window_size - 1)
        idx = yy * window_size + xx
        return torch.gather(cost_vol, 0, idx.unsqueeze(0)).squeeze(0)

    c11 = max_val
    c01 = get_val(-1, 0)
    c21 = get_val(1, 0)
    c10 = get_val(0, -1)
    c12 = get_val(0, 1)
    c00 = get_val(-1, -1)
    c02 = get_val(-1, 1)
    c20 = get_val(1, -1)
    c22 = get_val(1, 1)
    
    grad_y = (c21 - c01) / 2.0
    grad_yy = c21 + c01 - 2.0 * c11
    grad_x = (c12 - c10) / 2.0
    grad_xx = c12 + c10 - 2.0 * c11
    grad_xy = (c22 - c20 - c02 + c00) / 4.0
    
    det = grad_xx * grad_yy - grad_xy * grad_xy
    
    nonzero = det != 0
    safe_det = torch.where(nonzero, det, torch.ones_like(det))
    dx = torch.where(
        nonzero, -(grad_yy * grad_x - grad_xy * grad_y) / safe_det,
        torch.zeros_like(det))
    dy = torch.where(
        nonzero, -(grad_xx * grad_y - grad_xy * grad_x) / safe_det,
        torch.zeros_like(det))
    
    shift_y = (max_wy.float() - cal_half_window) + dy
    shift_x = (max_wx.float() - cal_half_window) + dx
    shift_y = torch.clamp(shift_y, -cal_half_window, cal_half_window)
    shift_x = torch.clamp(shift_x, -cal_half_window, cal_half_window)
    
    return shift_y, shift_x, max_val


def _subpixel_refinement_gaussian(cost_vol, max_idx, max_val, window_size, cal_half_window):
    """Sub-pixel refinement using Gaussian fitting.
    More accurate for correlation peaks which typically follow Gaussian shape.
    Fits ln(correlation) as a 2nd order polynomial to find Gaussian center.
    
    Note: For NCC which produces values in [-1, 1], we shift values to be positive
    before taking log. This preserves the peak location.
    """
    # Convert idx to wy, wx
    max_wy = torch.div(max_idx, window_size, rounding_mode='floor')
    max_wx = max_idx % window_size
    
    # Mask for valid interior pixels
    valid_mask = (max_wy > 0) & (max_wy < window_size - 1) & \
                 (max_wx > 0) & (max_wx < window_size - 1)
    
    def get_val(offset_y, offset_x):
        idx = max_idx + offset_y * window_size + offset_x
        idx = torch.clamp(idx, 0, window_size*window_size - 1)
        return torch.gather(cost_vol, 0, idx.unsqueeze(0)).squeeze(0)

    c11 = max_val
    c01 = get_val(-1, 0)
    c21 = get_val(1, 0)
    c10 = get_val(0, -1)
    c12 = get_val(0, 1)
    
    # For Gaussian fitting, we use log of correlation values
    # For NCC (values in [-1, 1]), shift by adding 1.1 to ensure all values > 0
    # This shift doesn't affect the peak location
    min_val = torch.min(torch.stack([c11, c01, c21, c10, c12]), dim=0)[0]
    shift_val = torch.where(min_val < 0.1, 1.1 - min_val, torch.zeros_like(min_val))
    
    eps = 1e-6
    ln_c11 = torch.log(torch.clamp(c11 + shift_val, min=eps))
    ln_c01 = torch.log(torch.clamp(c01 + shift_val, min=eps))
    ln_c21 = torch.log(torch.clamp(c21 + shift_val, min=eps))
    ln_c10 = torch.log(torch.clamp(c10 + shift_val, min=eps))
    ln_c12 = torch.log(torch.clamp(c12 + shift_val, min=eps))
    
    # 1D Gaussian fit along y: ln(c) = a*y^2 + b*y + c_const
    # Using 3 points: (-1, ln_c01), (0, ln_c11), (1, ln_c21)
    # Peak at y = -b/(2a)
    # a = (ln_c21 + ln_c01)/2 - ln_c11
    # b = (ln_c21 - ln_c01)/2
    a_y = (ln_c21 + ln_c01) / 2.0 - ln_c11
    b_y = (ln_c21 - ln_c01) / 2.0
    
    # 1D Gaussian fit along x
    a_x = (ln_c12 + ln_c10) / 2.0 - ln_c11
    b_x = (ln_c12 - ln_c10) / 2.0
    
    # Sub-pixel offset: peak at -b/(2a)
    # Only valid if a < 0 (concave down)
    valid_y = (a_y < -eps) & valid_mask
    valid_x = (a_x < -eps) & valid_mask
    
    dy = torch.where(valid_y, -b_y / (2.0 * a_y), torch.zeros_like(b_y))
    dx = torch.where(valid_x, -b_x / (2.0 * a_x), torch.zeros_like(b_x))
    
    # Clamp to [-1, 1] to stay within the 3x3 neighborhood
    dy = torch.clamp(dy, -1.0, 1.0)
    dx = torch.clamp(dx, -1.0, 1.0)
    
    shift_y = (max_wy.float() - cal_half_window) + dy
    shift_x = (max_wx.float() - cal_half_window) + dx
    
    return shift_y, shift_x, max_val


def _subpixel_refinement(cost_vol, max_idx, max_val, window_size, cal_half_window, method='parabolic'):
    """Sub-pixel refinement wrapper.
    Args:
        method: 'parabolic' (classic) or 'gaussian' (more accurate for correlation peaks)
    """
    if method == 'gaussian':
        return _subpixel_refinement_gaussian(cost_vol, max_idx, max_val, window_size, cal_half_window)
    else:
        return _subpixel_refinement_parabolic(cost_vol, max_idx, max_val, window_size, cal_half_window)


# Try to compile the sub-pixel refinement function for better performance
import platform
if platform.system() == 'Windows':
    _subpixel_refinement_compiled = _subpixel_refinement
    USE_TORCH_COMPILE = False
    prColor('torch.compile disabled on Windows (Triton support issues)', 'yellow')
else:
    try:
        _subpixel_refinement_compiled = torch.compile(_subpixel_refinement, mode='reduce-overhead')
        USE_TORCH_COMPILE = True
        prColor('torch.compile enabled for sub-pixel refinement', 'green')
    except Exception as e:
        _subpixel_refinement_compiled = _subpixel_refinement
        USE_TORCH_COMPILE = False
        prColor(f'torch.compile not available: {e}', 'yellow')


class WSVT:
    def __init__(self,
                 img_stack,
                 ref_stack,
                 crop=512,
                 cal_half_window=20,
                 N_template=0,
                 N_s_extend=4,
                 n_cores=4,
                 n_group=4,
                 energy=14e3,
                 p_x=0.65e-6,
                 Mag_factor=1,
                 z=500e-3,
                 wavelet_level_cut=2,
                 pyramid_level=2,
                 n_iter=1,
                 use_estimate=False,
                 use_wavelet=True,
                 use_GPU=0,
                 use_c=True,
                 calc_T_D=False,
                 calc_darkfield=True,
                 use_fft_corr=False,
                 subpixel_method='parabolic',
                 use_ncc=False,
                 filter_outliers=False,
                 memory_efficient=False,
                 save_plots=False,
                 wavelet_chunk_size=0,
                 wavelet_backend='auto'):
        self.img_data = img_stack
        self.ref_data = ref_stack
        self.use_c = use_c
        self.calc_T_D = calc_T_D
        self.calc_darkfield = calc_darkfield
        self.darkfield_time = 0.0
        self.use_fft_corr = use_fft_corr
        # New accuracy improvement parameters
        self.subpixel_method = subpixel_method  # 'parabolic' or 'gaussian'
        self.use_ncc = use_ncc  # Use normalized cross-correlation
        self.filter_outliers = filter_outliers  # Apply outlier filtering to displacement
        self.memory_efficient = memory_efficient
        self.save_plots = save_plots
        self.wavelet_chunk_size = wavelet_chunk_size
        self.wavelet_backend = wavelet_backend
        self._wavelet_c_meta_cache = {}
        # roi of the images
        self.crop = crop
        # the number of the area to calculate for each pixel, 2*cal_half_window X 2*cal_half_window
        self.cal_half_window = cal_half_window
        # the calculation window for high order pyramid
        self.N_s_extend = N_s_extend
        # Template half window size for the correlation searching
        self.N_template = N_template
        # process number for parallel
        self.n_cores = n_cores
        # number to reduce the each memory use
        self.n_group = n_group
        # Magnification factor for the phase calculation
        self.Mag_factor = Mag_factor

        # energy, 10kev
        self.energy = energy
        self.wavelength = sc.value(
            'inverse meter-electron volt relationship') / energy
        # pixel size [m]
        self.p_x = p_x
        # distance [m]
        self.z = z
        self.wavelet_level_cut = wavelet_level_cut
        # pyramid level to wrap the images
        self.pyramid_level = pyramid_level
        # iterations for the calculation
        self.n_iter = n_iter
        # if use the estimated displace as initial guess
        self.use_estimate = use_estimate
        # if use wavelet transform or not
        self.use_wavelet = use_wavelet

        # use GPU or not
        prColor(f"DEBUG: torch.cuda.is_available()={torch.cuda.is_available()}, use_GPU={use_GPU}", 'yellow')
        if torch.cuda.is_available() and use_GPU == 1:
            prColor('Use GPU found. Enable multi-resolution', 'cyan')
            self.use_GPU = True
            # for GPU, we can use multi-resolution process now
            # self.pyramid_level = 0
        else:
            prColor('No gpu found. Use CPU instead.', 'cyan')
            self.use_GPU = False

        if self.use_estimate:
            # get initial estimation from the cv2 flow tracking
            displace_estimate = slope_tracking(self.ref_data[0],
                                               self.img_data[0],
                                               n_window=self.cal_half_window)
            self.displace_estimate = [
                displace_estimate[0], displace_estimate[1]
            ]
        else:
            c, m, n = self.img_data.shape
            self.displace_estimate = [np.zeros((m, n)), np.zeros((m, n))]

    def resampling_spline(self, img, s):
        # img: original
        # s: size of the sampling, (row, col)
        m, n = img.shape
        x_axis = np.arange(n)
        y_axis = np.arange(m)
        fit = sfit.RectBivariateSpline(y_axis, x_axis, img)

        x_new = np.linspace(0, n - 1, s[1])
        y_new = np.linspace(0, m - 1, s[0])

        return fit(y_new, x_new)

    def stack_TemplateWindow(self, img):
        """generate multi resolution image for each window size

        Args:
            img (2D/3D numpy array): original image or image stack [image number, y, x]
            img_Nw: multi resolution image stack, [image number * (2+self.Nw_list+1), y, x]
        """
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        
        N_t = self.N_template
        if N_t == 0:
            # No template stacking needed
            return img
        
        # Vectorized approach using padding and slicing
        N, H, W = img.shape
        window_size = 2 * N_t + 1
        
        # Pad image with wrap mode to handle roll semantics
        img_padded = np.pad(img, ((0, 0), (N_t, N_t), (N_t, N_t)), mode='wrap')
        
        # Pre-allocate output array
        num_shifts = window_size * window_size
        img_stack = np.empty((num_shifts * N, H, W), dtype=img.dtype)
        
        idx = 0
        for dy in range(-N_t, N_t + 1):
            for dx in range(-N_t, N_t + 1):
                # Extract shifted version using slicing (faster than roll)
                shifted = img_padded[:, N_t + dy:N_t + dy + H, N_t + dx:N_t + dx + W]
                img_stack[idx*N:(idx+1)*N] = shifted
                idx += 1
        
        return img_stack
    
    def stack_TemplateWindow_torch(self, img_tensor):
        """
        GPU version of stack_TemplateWindow
        img_tensor: [N, H, W]
        Returns: [N * num_shifts, H, W]
        """
        N, H, W = img_tensor.shape
        N_t = self.N_template
        if N_t == 0:
            return img_tensor
            
        # Pad image with circular mode (equivalent to wrap)
        # pad expects (left, right, top, bottom)
        img_padded = nn.functional.pad(img_tensor.unsqueeze(0), (N_t, N_t, N_t, N_t), mode='circular').squeeze(0)
        
        window_size = 2 * N_t + 1
        slices = []
        for dy in range(window_size):
            for dx in range(window_size):
                # Extract shifted slice
                s = img_padded[:, dy:dy+H, dx:dx+W]
                slices.append(s)
        
        # Stack along channel dimension (dim 0)
        return torch.cat(slices, dim=0)

    def pyramid_data(self):
        """
        generate pyramid data for multi-resolution
        Optimized to use GPU for stacking and normalization if available.
        """
        ref_data = self.ref_data
        img_data = self.img_data

        # Matched WSVT profile: the Python reference normalizes the original
        # frame stack once, then constructs every spatial pyramid level from
        # that normalized level.  Normalizing each raw level independently is a
        # different operator and changes the coarse-to-fine displacement seed.
        if self.N_template == 0:
            ref_data = ((ref_data - np.ndarray.mean(ref_data, axis=0)) /
                        np.ndarray.std(ref_data, axis=0))
            img_data = ((img_data - np.ndarray.mean(img_data, axis=0)) /
                        np.ndarray.std(img_data, axis=0))
            ref_pyramid = [ref_data]
            img_pyramid = [img_data]
            prColor(
                'obtain pyramid image with pyramid level: {}'.format(
                    self.pyramid_level), 'green')
            for kk in range(self.pyramid_level):
                ref_pyramid.append(
                    pywt.dwtn(ref_pyramid[kk], 'db3', mode='zero',
                              axes=(-2, -1))['aa'])
                img_pyramid.append(
                    pywt.dwtn(img_pyramid[kk], 'db3', mode='zero',
                              axes=(-2, -1))['aa'])
            return ref_pyramid, img_pyramid

        ref_pyramid = []
        img_pyramid = []
        prColor(
            'obtain pyramid image with pyramid level: {}'.format(
                self.pyramid_level), 'green')
        
        # Build pyramid levels (Wavelet transform on CPU for now as pywt is fast enough)
        # Level 0
        ref_pyramid.append(ref_data)
        img_pyramid.append(img_data)
        
        for kk in range(self.pyramid_level):
            ref_pyramid.append(
                pywt.dwtn(ref_pyramid[kk], 'db3', mode='zero',
                          axes=(-2, -1))['aa'])
            img_pyramid.append(
                pywt.dwtn(img_pyramid[kk], 'db3', mode='zero',
                          axes=(-2, -1))['aa'])
        
        # Stack and Normalize
        # If GPU is enabled, move to GPU for these heavy operations
        use_gpu_processing = self.use_GPU and torch.cuda.is_available()
        
        for kk in range(len(ref_pyramid)):
            if use_gpu_processing:
                # Move to GPU - use float32 first to avoid half precision overflow
                # (half precision max ~65504, but raw image data can be much larger)
                # ref_pyramid[kk] is numpy [N, H, W]
                ref_tensor = torch.from_numpy(ref_pyramid[kk]).float().cuda()
                img_tensor = torch.from_numpy(img_pyramid[kk]).float().cuda()
                
                # Stack Template Window on GPU
                ref_tensor = self.stack_TemplateWindow_torch(ref_tensor)
                img_tensor = self.stack_TemplateWindow_torch(img_tensor)
                
                # Normalize on GPU (already in float32)
                # mean/std across axis 0 (channels)
                # Use correction=0 to match numpy's default behavior (ddof=0)
                ref_mean = torch.mean(ref_tensor, dim=0, keepdim=True)
                ref_std = torch.std(ref_tensor, dim=0, keepdim=True, correction=0)
                ref_std = torch.clamp(ref_std, min=1e-6)  # Match CPU's 1e-6
                ref_tensor = (ref_tensor - ref_mean) / ref_std
                
                img_mean = torch.mean(img_tensor, dim=0, keepdim=True)
                img_std = torch.std(img_tensor, dim=0, keepdim=True, correction=0)
                img_std = torch.clamp(img_std, min=1e-6)
                img_tensor = (img_tensor - img_mean) / img_std
                
                # Store back as numpy array on CPU to save GPU memory
                # Keep float32 to match CPU precision - half precision causes differences
                ref_pyramid[kk] = ref_tensor.cpu().numpy()
                img_pyramid[kk] = img_tensor.cpu().numpy()
                
            else:
                # CPU path (original)
                ref_pyramid[kk] = self.stack_TemplateWindow(ref_pyramid[kk])
                img_pyramid[kk] = self.stack_TemplateWindow(img_pyramid[kk])
                
                ref_pyramid[kk] = ((ref_pyramid[kk] - np.ndarray.mean(ref_pyramid[kk], axis=0)) /
                        (np.ndarray.std(ref_pyramid[kk], axis=0) + 1e-6))
                img_pyramid[kk] = ((img_pyramid[kk] - np.ndarray.mean(img_pyramid[kk], axis=0)) /
                        (np.ndarray.std(img_pyramid[kk], axis=0) + 1e-6))
                        
        return ref_pyramid, img_pyramid

    def run_wavelet_c(self, img, levels, return_levels):
        # img: [Channels, H, W]
        channels, h, w = img.shape
        return_levels = max(1, min(return_levels, levels + 1))
        key = (channels, levels, return_levels)
        meta = self._wavelet_c_meta_cache.get(key)
        if meta is None:
            dummy = np.zeros(channels)
            coeffs = pywt.wavedec(dummy, 'db2', level=levels, mode='zero')
            kept_coeffs = coeffs[0:return_levels]
            total_output_channels = sum(len(c) for c in kept_coeffs)

            offsets = np.zeros(levels + 1, dtype=np.int32)
            current_offset = 0
            for i in range(return_levels):
                offsets[i] = current_offset
                current_offset += len(coeffs[i])

            level_name = ['D{:d}'.format(kk + 1) for kk in range(levels)]
            level_name.append('A{:d}'.format(levels))
            level_name = level_name[-return_levels:]
            meta = (offsets, total_output_channels, level_name)
            self._wavelet_c_meta_cache[key] = meta

        offsets, total_output_channels, level_name = meta

        img_t = np.ascontiguousarray(np.moveaxis(img, 0, -1), dtype=np.float32)

        n_workers = max(1, min(self.n_cores, h))
        if n_workers == 1 or h < 256:
            full_output = np.zeros((h, w, total_output_channels), dtype=np.float32)
            c_lib.wavelet_transform_db2(
                img_t,
                h * w,
                channels,
                levels,
                return_levels,
                offsets,
                total_output_channels,
                full_output
            )
            return full_output, level_name

        edges = np.linspace(0, h, n_workers + 1, dtype=int)
        results = [None] * n_workers

        def process_chunk(idx, y0, y1):
            sub_h = y1 - y0
            sub_img_t = np.ascontiguousarray(img_t[y0:y1, :, :], dtype=np.float32)
            sub_output = np.zeros((sub_h, w, total_output_channels), dtype=np.float32)
            c_lib.wavelet_transform_db2(
                sub_img_t,
                sub_h * w,
                channels,
                levels,
                return_levels,
                offsets,
                total_output_channels,
                sub_output
            )
            return idx, sub_output

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i in range(n_workers):
                y0, y1 = edges[i], edges[i + 1]
                if y1 > y0:
                    futures.append(executor.submit(process_chunk, i, y0, y1))
            for future in concurrent.futures.as_completed(futures):
                idx, sub_output = future.result()
                results[idx] = sub_output

        full_output = np.concatenate([r for r in results if r is not None], axis=0)
        
        # Return [H, W, Output_Channels] to match Wavelet_transform_multiprocess
        return full_output, level_name

    def run_wavelet_gpu(self, img_data, w_level, return_level):
        """
        Run wavelet transform on GPU
        """
        if not hasattr(self, 'wavelet_gpu'):
            self.wavelet_gpu = WaveletTransformGPU(wavelet_name='db2', device='cuda')
            
        # img_data: (N, H, W)
        # Check if img_data is already a tensor (from pyramid_data optimization)
        if torch.is_tensor(img_data):
            img_tensor = img_data
            if img_tensor.device.type != 'cuda':
                img_tensor = img_tensor.cuda()
            # Use float32 for wavelet transform to match CPU precision
            if img_tensor.dtype != torch.float32:
                img_tensor = img_tensor.float()
            N, H, W = img_tensor.shape
        else:
            N, H, W = img_data.shape
            # Move to GPU - use float32 to match CPU precision
            img_tensor = torch.from_numpy(img_data).float().cuda()
            
        img_reshaped = img_tensor.permute(1, 2, 0).reshape(-1, N)

        num_pixels = img_reshaped.shape[0]
        if self.wavelet_chunk_size and self.wavelet_chunk_size > 0:
            chunk_size = int(self.wavelet_chunk_size)
        else:
            try:
                free_mem, _ = torch.cuda.mem_get_info()
                target_bytes = int(free_mem * 0.10)
                bytes_per_sample = max(1, N * 4 * 4)
                chunk_size = max(15000, min(150000, target_bytes // bytes_per_sample))
            except Exception:
                chunk_size = 60000

        coeffs_combined = None
        with torch.no_grad():
            for i in range(0, num_pixels, chunk_size):
                batch = img_reshaped[i:i + chunk_size]
                batch_coeffs = self.wavelet_gpu.wavedec(batch, level=w_level, mode='zero')
                batch_combined = torch.cat(batch_coeffs[:return_level], dim=2).squeeze(1)

                if coeffs_combined is None:
                    coeffs_combined = torch.empty((num_pixels, batch_combined.shape[1]),
                                                  dtype=batch_combined.dtype,
                                                  device=batch_combined.device)
                coeffs_combined[i:i + batch.shape[0], :] = batch_combined

                del batch, batch_coeffs, batch_combined
        
        # Reshape to (H, W, C_out) then permute to (C_out, H, W)
        # We return (C, H, W) which is what displace_torch expects (after conversion)
        C_out = coeffs_combined.shape[1]
        output = coeffs_combined.view(H, W, C_out).permute(2, 0, 1)
        del coeffs_combined, img_reshaped, img_tensor
        
        # Generate level names (dummy, just for logging)
        level_name = []
        for kk in range(w_level):
            level_name.append('D{:d}'.format(kk + 1))
        level_name.append('A{:d}'.format(w_level))
        level_name = level_name[-return_level:]
        
        return output, level_name

    def wavelet_data(self):
        """
        generate wavelet data from the image data

        """
        # process the data to get the wavelet transform
        # import cProfile
        # import pstats
        # import io
        
        # # Profile pyramid_data
        # pr = cProfile.Profile()
        # pr.enable()
        
        self._wavelet_time = 0.0
        _t_pyramid_start = time.time()
        ref_pyramid, img_pyramid = self.pyramid_data()
        _t_pyramid_end = time.time()
        self.pyramid_time = _t_pyramid_end - _t_pyramid_start
        prColor('pyramid time: {:.6f} s'.format(self.pyramid_time), 'light_purple')

        # pr.disable()
        # s = io.StringIO()
        # ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        # ps.print_stats(10)
        # prColor('=== Pyramid Data Profiling (top 10) ===', 'yellow')
        # print(s.getvalue())
        
        if self.use_wavelet:
            prColor('obtain wavelet data...', 'green')
            wavelet_method = 'db2'
            # wavelet_method = 'bior1.3'
            # wavelet wrapping level. 2 is half, 3 is 1/3 of the size
            max_wavelet_level = pywt.dwt_max_level(ref_pyramid[0].shape[0],
                                                   wavelet_method)
            prColor('max wavelet level: {}'.format(max_wavelet_level), 'green')
            self.wavelet_level = max_wavelet_level
            coefs_level = self.wavelet_level + 1 - self.wavelet_level_cut

            if ref_pyramid[0].shape[0] > 150:
                self.wavelet_add_list = [0, 0, 0, 0, 0, 0]
            elif ref_pyramid[0].shape[0] > 50:
                self.wavelet_add_list = [0, 0, 1, 2, 2, 2]
            else:
                self.wavelet_add_list = [2, 2, 2, 2, 2, 2]

            # wavelet transform and cut for the pyramid images
            start_time = time.time()
            for p_level in range(len(img_pyramid)):
                if p_level > len(self.wavelet_add_list):
                    wavelevel_add = 2
                else:
                    wavelevel_add = self.wavelet_add_list[p_level]

                use_c_backend = choose_c_wavelet_backend(
                    self.wavelet_backend, self.use_c, wavelet_method, USE_C_SOLVER)

                if self.use_GPU:
                    img_wa, level_name = self.run_wavelet_gpu(
                        img_pyramid[p_level],
                        self.wavelet_level,
                        coefs_level + wavelevel_add)
                elif use_c_backend:
                    img_wa, level_name = self.run_wavelet_c(
                        img_pyramid[p_level],
                        self.wavelet_level,
                        coefs_level + wavelevel_add)
                else:
                    img_wa, level_name = Wavelet_transform_multiprocess(
                        img_pyramid[p_level],
                        self.n_cores,
                        wavelet_method,
                        w_level=self.wavelet_level,
                        return_level=coefs_level + wavelevel_add)

                img_pyramid[p_level] = img_wa

                if self.use_GPU:
                    ref_wa, level_name = self.run_wavelet_gpu(
                        ref_pyramid[p_level],
                        self.wavelet_level,
                        coefs_level + wavelevel_add)
                elif use_c_backend:
                    ref_wa, level_name = self.run_wavelet_c(
                        ref_pyramid[p_level],
                        self.wavelet_level,
                        coefs_level + wavelevel_add)
                else:
                    ref_wa, level_name = Wavelet_transform_multiprocess(
                        ref_pyramid[p_level],
                        self.n_cores,
                        wavelet_method,
                        w_level=self.wavelet_level,
                        return_level=coefs_level + wavelevel_add)
                ref_pyramid[p_level] = ref_wa
                
                if torch.is_tensor(ref_wa):
                    vec_len = ref_wa.shape[0]
                else:
                    vec_len = ref_wa.shape[2]
                    
                prColor(
                    'pyramid level: {}\nvector length: {}\nUse wavelet coef: {}'
                    .format(p_level, vec_len, level_name), 'green')

            end_time = time.time()
            print('wavelet time: {}'.format(end_time - start_time))
            self._wavelet_time = end_time - start_time
        else:
            img_pyramid = [
                np.moveaxis(img_data, 0, -1) for img_data in img_pyramid
            ]
            ref_pyramid = [
                np.moveaxis(img_data, 0, -1) for img_data in ref_pyramid
            ]
            self.wavelet_level = None
            self.wavelet_add_list = None
            self.wavelet_level_cut = None

        return ref_pyramid, img_pyramid

    def displace_wavelet(self, y_list, img_wa_stack, ref_wa_stack,
                         displace_pyramid, cal_half_window, n_pad):
        """
        calculate displacement using CPU

        Args:
            y_list (list): y axis of data
            img_wa_stack (ndarray): sample wavelet data
            ref_wa_stack (ndarray): reference wavelet data
            displace_pyramid (ndarray): estimated displacement for multi-resolution
            cal_half_window (int): half size of the searching window
            n_pad (int): padding width

        """
        dim = img_wa_stack.shape
        disp_x = np.zeros((dim[0], dim[1]), dtype=np.float32)
        disp_y = np.zeros((dim[0], dim[1]), dtype=np.float32)
        darkfield_nd = np.zeros((dim[0], dim[1]), dtype=np.float32)

        if self.use_c and USE_C_SOLVER:
            # Prepare data for C solver
            img_wa_c = np.ascontiguousarray(img_wa_stack, dtype=np.float32)
            ref_wa_c = np.ascontiguousarray(ref_wa_stack, dtype=np.float32)
            disp_guess_y = np.ascontiguousarray(displace_pyramid[0], dtype=np.float32)
            disp_guess_x = np.ascontiguousarray(displace_pyramid[1], dtype=np.float32)
            
            c_lib.calculate_displacement_block(
                img_wa_c,
                ref_wa_c,
                disp_guess_y,
                disp_guess_x,
                dim[0], # dim_y
                dim[1], # dim_x
                dim[2], # depth
                ref_wa_stack.shape[0], # ref_h
                ref_wa_stack.shape[1], # ref_w
                n_pad,
                cal_half_window,
                disp_y,
                disp_x
            )
            
            disp_add_y = displace_pyramid[0] + disp_y
            disp_add_x = displace_pyramid[1] + disp_x
            # C solver doesn't return confidence yet
            return disp_add_y, disp_add_x, y_list, darkfield_nd

        # the axis for the peak position finding
        window_size = 2 * cal_half_window + 1
        y_axis = np.arange(window_size) - cal_half_window
        x_axis = np.arange(window_size) - cal_half_window
        XX, YY = np.meshgrid(x_axis, y_axis)

        for yy in range(dim[0]):
            for xx in range(dim[1]):
                img_wa_line = img_wa_stack[yy, xx, :]
                ref_wa_data = ref_wa_stack[
                    n_pad + yy + int(displace_pyramid[0][yy, xx]):n_pad + yy +
                    int(displace_pyramid[0][yy, xx]) + window_size,
                    n_pad + xx + int(displace_pyramid[1][yy, xx]):n_pad + xx +
                    int(displace_pyramid[1][yy, xx]) + window_size, :]

                Corr_img = dist_numba(img_wa_line, ref_wa_data)
                '''
                    use gradient to find the peak
                '''
                disp_y[yy,
                       xx], disp_x[yy,
                                   xx], _, darkfield_nd[yy, xx] = find_disp(Corr_img,
                                                         XX,
                                                         YY,
                                                         sub_resolution=True)

        disp_add_y = displace_pyramid[0] + disp_y
        disp_add_x = displace_pyramid[1] + disp_x
        return disp_add_y, disp_add_x, y_list, darkfield_nd

    def displace_torch(self, img_wa_stack, ref_wa_stack, cal_half_window,
                       center_y=None, center_x=None):
        """
        calculate displacement using GPU (Optimized)

        Args:
            img_wa_stack (ndarray or Tensor): sample wavelet data
            ref_wa_stack (ndarray or Tensor): reference wavelet data
            cal_half_window (int): half size of the searching window

        """
        # dim = img_wa_stack.shape
        
        start_time = time.time()
        # Use float32 for correlation calculation to match CPU precision
        if torch.is_tensor(img_wa_stack):
            # Already on GPU and in (C, H, W) format from run_wavelet_gpu
            img_stack_cuda = img_wa_stack.float()
            ref_stack_cuda = ref_wa_stack.float()
        else:
            img_stack_cuda = torch.from_numpy(np.moveaxis(img_wa_stack, -1, 0)).cuda().float()
            ref_stack_cuda = torch.from_numpy(np.moveaxis(ref_wa_stack, -1, 0)).cuda().float()

        # Determine whether to use FFT based on search range and user preference
        # FFT is more efficient for large search ranges (typically > 15)
        use_fft = self.use_fft_corr or cal_half_window >= 15
        
        window_size = 2 * cal_half_window + 1

        if center_y is not None or center_x is not None:
            if center_y is None or center_x is None:
                raise ValueError('center_y and center_x must be provided together')
            if self.use_ncc or use_fft:
                raise ValueError('centered GPU search currently supports spatial SSD only')
            cost_vol = cost_volume_spatial_centered(
                img_stack_cuda, ref_stack_cuda, cal_half_window, center_y, center_x)
            max_val, max_idx = torch.max(cost_vol, dim=0)
            shift_y, shift_x, confidence = _subpixel_refinement(
                cost_vol, max_idx, max_val, window_size, cal_half_window,
                method=self.subpixel_method
            )
            del cost_vol
        elif self.memory_efficient:
            max_val, max_idx = cost_volume_peak_streaming(
                img_stack_cuda,
                ref_stack_cuda,
                search_range=cal_half_window,
                use_ncc=self.use_ncc)
            shift_y = (max_idx // window_size).float() - cal_half_window
            shift_x = (max_idx % window_size).float() - cal_half_window
            confidence = max_val
        else:
            cost_vol = cost_volume(img_stack_cuda,
                                   ref_stack_cuda,
                                   search_range=cal_half_window,
                                   use_fft=use_fft,
                                   use_ncc=self.use_ncc)
            cost_vol = cost_vol.float()
            max_val, max_idx = torch.max(cost_vol, dim=0)
            shift_y, shift_x, confidence = _subpixel_refinement(
                cost_vol, max_idx, max_val, window_size, cal_half_window,
                method=self.subpixel_method
            )
            del cost_vol
        del img_stack_cuda, ref_stack_cuda
        
        return shift_y, shift_x, confidence

    def warp_image(self, x, flow_y, flow_x, padding_mode='zeros', return_half=True):
        # x: [C, H, W]
        # flow: [H, W]
        C, H, W = x.shape
        
        # Create grid
        xx = torch.arange(0, W, device=x.device).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H, device=x.device).view(-1, 1).repeat(1, W)
        
        # Apply flow
        # We want to sample at (y + flow_y, x + flow_x)
        # grid_sample uses (x, y) convention
        grid_x = xx + flow_x
        grid_y = yy + flow_y
        
        # Normalize to [-1, 1]
        grid_x = 2.0 * grid_x / (W - 1) - 1.0
        grid_y = 2.0 * grid_y / (H - 1) - 1.0
        
        grid = torch.stack([grid_x, grid_y], dim=2).unsqueeze(0) # [1, H, W, 2]
        
        # Grid sample
        # x needs to be [N, C, H, W] -> [1, C, H, W]
        # grid_sample expects float32 usually
        # Use align_corners=True to match the grid generation formula: 2*x/(W-1) - 1
        warped = nn.functional.grid_sample(x.unsqueeze(0).float(), grid.float(), mode='bilinear', padding_mode=padding_mode, align_corners=True)
        
        res = warped.squeeze(0)
        if return_half:
            return res.half()
        else:
            return res

    def solver_cuda(self):
        """
        speckle tracking solver using GPU
        """
        ref_pyramid, img_pyramid = self.wavelet_data()
        # ref_pyramid is a list: [level0, level1, level2, ...]
        # level0 is finest (original size)
        # levelN is coarsest
        
        ref_safe = np.clip(self.ref_data[0].astype(np.float64), 1e-10, None)
        transmission = np.clip(
            self.img_data[0].astype(np.float64) / ref_safe, 0.01, 10.0)
        _t_darkfield_start = time.time()
        if self.calc_darkfield:
            darkfield = np.std(self.stack_TemplateWindow(self.img_data), axis=0) / np.std(self.stack_TemplateWindow(self.ref_data), axis=0)
        else:
            darkfield = None
        self.darkfield_time = time.time() - _t_darkfield_start
        prColor('raw darkfield time: {:.6f} s{}'.format(
            self.darkfield_time, '' if self.calc_darkfield else ' (disabled)'), 'light_purple')

        if self.calc_T_D:
            # Keep data for T/D calculation
            img_data_backup = self.img_data
            ref_data_backup = self.ref_data

        for attr in ('img_data', 'ref_data'):
            self.__dict__.pop(attr, None)

        start_time = time.time()
        
        # Initialize displacement
        current_disp_y = None
        current_disp_x = None
        
        # Iterate from coarsest to finest
        # Note: wavelet_data returns [level0, level1, ...]
        # So we iterate backwards
        num_levels = len(ref_pyramid)
        
        for level in range(num_levels - 1, -1, -1):
            img_wa = img_pyramid[level]
            ref_wa = ref_pyramid[level]
            
            # Determine search window for this level
            if level == num_levels - 1:
                # Coarsest level: search full range (scaled)
                # Effective range at this level: original_range / 2^level
                search_range = int(np.ceil(self.cal_half_window / (2**level)))
            else:
                # Finer levels: search small residual range
                search_range = self.N_s_extend # e.g. 4
            
            prColor(f'Processing level {level}, search range {search_range}', 'cyan')
            
            # Prepare GPU tensors
            if torch.is_tensor(img_wa):
                # Keep the displacement search in float32. The correlation
                # surface can be nearly flat in absolute scale while its local
                # curvature determines the sub-pixel peak; an FP16 round-trip
                # changes that curvature enough to select unstable refinements.
                img_tensor = img_wa.float()
                ref_tensor = ref_wa.float()
            else:
                img_tensor = torch.from_numpy(np.moveaxis(img_wa, -1, 0)).float().cuda()
                ref_tensor = torch.from_numpy(np.moveaxis(ref_wa, -1, 0)).float().cuda()
            
            if current_disp_y is not None:
                # Upsample displacement to current level size
                H, W = img_tensor.shape[1], img_tensor.shape[2]

                # Match the CPU solver's coarse-to-fine guess generation:
                # RectBivariateSpline, scale by two, then round to an integer
                # search center. This host-side step is retained for semantic
                # agreement first; a CUDA-equivalent interpolator can replace
                # it only after it passes the same field gate.
                guess_y_np = np.round(
                    self.resampling_spline(current_disp_y.detach().cpu().numpy(), (H, W)) * 2)
                guess_x_np = np.round(
                    self.resampling_spline(current_disp_x.detach().cpu().numpy(), (H, W)) * 2)
                guess_limit = self.cal_half_window / (2**level)
                guess_y_np = np.clip(guess_y_np, -guess_limit, guess_limit).astype(np.float32)
                guess_x_np = np.clip(guess_x_np, -guess_limit, guess_limit).astype(np.float32)
                current_disp_y = torch.from_numpy(guess_y_np).to(img_tensor.device)
                current_disp_x = torch.from_numpy(guess_x_np).to(img_tensor.device)
                
                # Calculate the residual around each pixel's own integer search
                # center, matching the CPU/C extension implementation.
                res_y, res_x, confidence = self.displace_torch(
                    img_tensor, ref_tensor, search_range,
                    center_y=current_disp_y, center_x=current_disp_x)
                
                # Update
                current_disp_y += res_y
                current_disp_x += res_x
            else:
                # First level (coarsest)
                current_disp_y, current_disp_x, confidence = self.displace_torch(img_tensor, ref_tensor, search_range)

            # Match the CPU solver's per-level admissible displacement range.
            # Without this clamp, unstable sub-pixel curvature at the coarse
            # level can create values in the thousands of pixels and corrupt
            # the finer-level warp before the final output clamp is reached.
            level_limit = self.cal_half_window / (2**level)
            current_disp_y = torch.clamp(current_disp_y, -level_limit, level_limit)
            current_disp_x = torch.clamp(current_disp_x, -level_limit, level_limit)

            if self.memory_efficient:
                del img_tensor, ref_tensor
                if 'warped_ref' in locals():
                    del warped_ref
                torch.cuda.empty_cache()

        end_time = time.time()
        prColor('GPU calculation time: {}'.format(end_time - start_time), 'cyan')
        
        # Save confidence as darkfield_nd
        darkfield_nd = confidence.cpu().numpy()
        
        if self.calc_T_D:
            prColor('Calculating Transmission and Darkfield with warped reference...', 'cyan')
            # Warp reference stack using the calculated displacement
            # ref_data_backup: [N, H, W]
            # current_disp_y/x: [H, W]
            
            # Move ref data to GPU
            ref_tensor = torch.from_numpy(ref_data_backup).float().cuda() # [N, H, W]
            img_tensor = torch.from_numpy(img_data_backup).float().cuda() # [N, H, W]
            
            # Expand displacement to match stack size [N, H, W]
            # Actually warp_image expects [C, H, W] and [H, W] flow, and warps each channel C with the same flow.
            # Here C=N. So we can pass ref_tensor directly as "x" to warp_image.
            # warp_image expects x to be [C, H, W].
            
            warped_ref_stack = self.warp_image(ref_tensor, current_disp_y.float(), current_disp_x.float(), padding_mode='border', return_half=False)
            
            # Calculate T and D on GPU
            # Use stack_TemplateWindow logic for more robust statistics (match CPU logic but with warped reference)
            img_stacked = self.stack_TemplateWindow_torch(img_tensor)
            ref_stacked = self.stack_TemplateWindow_torch(warped_ref_stack)
            
            mean_img = torch.mean(img_stacked, dim=0)
            mean_ref = torch.mean(ref_stacked, dim=0)
            
            std_img = torch.std(img_stacked, dim=0)
            std_ref = torch.std(ref_stacked, dim=0)
            
            transmission = (mean_img / (mean_ref + 1e-6)).cpu().numpy()
            darkfield = (std_img / (std_ref + 1e-6)).cpu().numpy()
            
            # Clean up
            del ref_tensor, img_tensor, warped_ref_stack, mean_img, mean_ref, std_img, std_ref, img_stacked, ref_stacked
            torch.cuda.empty_cache()

        displace_y = current_disp_y.cpu().numpy()
        displace_x = current_disp_x.cpu().numpy()

        # Apply outlier filtering if enabled
        if self.filter_outliers:
            prColor('Applying outlier filtering to displacement...', 'cyan')
            displace_y, displace_x, _ = filter_displacement_outliers(displace_y, displace_x)

        displace = [
            np.fmax(np.fmin(displace_y, self.cal_half_window),
                    -self.cal_half_window),
            np.fmax(np.fmin(displace_x, self.cal_half_window),
                    -self.cal_half_window)
        ]
        prColor('displace map wrapping: {}'.format(displace[0].shape), 'green')
        print('max of displace: {}, min of displace: {}'.format(
            np.amax(displace[0]), np.amin(displace[1])))
        displace[0] = -displace[0][self.cal_half_window:-self.cal_half_window,
                                   self.cal_half_window:-self.cal_half_window]
        displace[1] = -displace[1][self.cal_half_window:-self.cal_half_window,
                                   self.cal_half_window:-self.cal_half_window]
        darkfield_nd = darkfield_nd[self.cal_half_window:-self.cal_half_window,
                                    self.cal_half_window:-self.cal_half_window]

        DPC_y = (displace[0] - np.mean(displace[0])) * self.p_x / (self.z / self.Mag_factor)
        DPC_x = (displace[1] - np.mean(displace[1])) * self.p_x / (self.z / self.Mag_factor)

        phase = -frankotchellappa(
            DPC_x, DPC_y) * self.p_x * 2 * np.pi / self.wavelength
        self.time_cost = end_time - start_time

        return displace, [DPC_y, DPC_x], phase, transmission, darkfield, darkfield_nd

    def solver(self):
        """
        speckle tracking solver using CPU
        """
        _t_solver_start = time.time()
        ref_pyramid, img_pyramid = self.wavelet_data()
        print(ref_pyramid[0].shape, img_pyramid[0].shape)

        ref_safe = np.clip(self.ref_data[0].astype(np.float64), 1e-10, None)
        transmission = np.clip(
            self.img_data[0].astype(np.float64) / ref_safe, 0.01, 10.0)
        _t_darkfield_start = time.time()
        if self.calc_darkfield:
            darkfield = np.std(self.stack_TemplateWindow(self.img_data), axis=0) / np.std(self.stack_TemplateWindow(self.ref_data), axis=0)
        else:
            darkfield = None
        self.darkfield_time = time.time() - _t_darkfield_start
        prColor('raw darkfield time: {:.6f} s{}'.format(
            self.darkfield_time, '' if self.calc_darkfield else ' (disabled)'), 'light_purple')

        for attr in ('img_data', 'ref_data'):
            self.__dict__.pop(attr, None)

        cores = ms.cpu_count()
        prColor('Computer available cores: {}'.format(cores), 'green')

        if cores > self.n_cores:
            cores = self.n_cores
        else:
            cores = ms.cpu_count()
        prColor('Use {} cores'.format(cores), 'light_purple')
        prColor('Process group number: {}'.format(self.n_group), 'light_purple')

        if cores * self.n_group > ref_pyramid[0].shape[0]:
            n_tasks = ref_pyramid[0].shape[0]
        else:
            n_tasks = cores * self.n_group
        
        prColor('seperate into {} subtaskes'.format(n_tasks), 'green')

        _t_displace_start = time.time()
        # use pyramid wrapping
        max_pyramid_searching_window = int(
            np.ceil(self.cal_half_window / 2**self.pyramid_level))
        searching_window_pyramid_list = [self.N_s_extend] * self.pyramid_level + [
            int(max_pyramid_searching_window)
        ]

        m, n, c = img_pyramid[0].shape
        displace = self.displace_estimate

        for k_iter in range(self.n_iter):
            # iteration to approximating the results
            displace = [img / 2**self.pyramid_level for img in displace]

            m, n, c = img_pyramid[-1].shape
            displace[0] = self.resampling_spline(displace[0], (m, n))
            displace[1] = self.resampling_spline(displace[1], (m, n))

            prColor(
                'down sampling the dispalce to size: {}'.format(
                    displace[0].shape), 'green')

            displace = [
                np.fmax(
                    np.fmin(displace[0],
                            self.cal_half_window / 2**self.pyramid_level),
                    -self.cal_half_window / 2**self.pyramid_level),
                np.fmax(
                    np.fmin(displace[1],
                            self.cal_half_window / 2**self.pyramid_level),
                    -self.cal_half_window / 2**self.pyramid_level)
            ]

            for p_level in range(self.pyramid_level, -1, -1):
                # first pyramid, searching the window. Then search nearby
                if p_level == self.pyramid_level:
                    pyramid_seaching_window = searching_window_pyramid_list[
                        p_level]
                    m, n, c = img_pyramid[p_level].shape
                    displace_pyramid = [np.round(img) for img in displace]

                    n_pad = int(np.ceil(self.cal_half_window / 2**p_level))

                else:
                    pyramid_seaching_window = searching_window_pyramid_list[
                        p_level]
                    # extend displace_pyramid with upsampling of 2 and also displace value is 2 times larger
                    m, n, c = img_pyramid[p_level].shape
                    displace_pyramid = [
                        np.round(self.resampling_spline(img, (m, n)) * 2)
                        for img in displace
                    ]

                    displace_pyramid = [
                        np.fmax(
                            np.fmin(displace_pyramid[0],
                                    self.cal_half_window / 2**p_level),
                            -self.cal_half_window / 2**p_level),
                        np.fmax(
                            np.fmin(displace_pyramid[1],
                                    self.cal_half_window / 2**p_level),
                            -self.cal_half_window / 2**p_level)
                    ]

                    n_pad = int(np.ceil(self.cal_half_window / 2**p_level))

                prColor(
                    'pyramid level: {}\nImage size: {}\nsearching window:{}'.
                    format(p_level, ref_pyramid[p_level].shape,
                           pyramid_seaching_window), 'cyan')
                # split the y axis into small groups, all splitted in vertical direction
                y_axis = np.arange(ref_pyramid[p_level].shape[0])
                chunks_idx_y = np.array_split(y_axis, n_tasks)

                dim = img_pyramid[p_level].shape

                ref_wa_pad = np.pad(ref_pyramid[p_level],
                                    ((n_pad + pyramid_seaching_window,
                                      n_pad + pyramid_seaching_window),
                                     (n_pad + pyramid_seaching_window,
                                      n_pad + pyramid_seaching_window),
                                     (0, 0)),
                                    'constant',
                                    constant_values=(0, 0))

                # use CPU parallel to calculate
                result_list = []
                '''
                    calculate the pixel displacement for the pyramid images
                '''
                if self.use_c and USE_C_SOLVER:
                    Executor = concurrent.futures.ThreadPoolExecutor
                else:
                    Executor = concurrent.futures.ProcessPoolExecutor

                with Executor(max_workers=cores) as executor:

                    futures = []
                    for y_list in chunks_idx_y:
                        # get the stack data
                        img_wa_stack = img_pyramid[p_level][y_list, :, :]
                        ref_wa_stack = ref_wa_pad[
                            y_list[0]:y_list[-1] + 2 *
                            (n_pad + pyramid_seaching_window) + 1, :, :]

                        # start the jobs
                        futures.append(
                            executor.submit(self.displace_wavelet, y_list,
                                            img_wa_stack, ref_wa_stack,
                                            (displace_pyramid[0][y_list, :],
                                             displace_pyramid[1][y_list, :]),
                                            pyramid_seaching_window, n_pad))

                    for future in concurrent.futures.as_completed(futures):

                        try:
                            result_list.append(future.result())
                            # display the status of the program
                            Total_iter = cores * self.n_group
                            Current_iter = len(result_list)
                            percent_iter = Current_iter / Total_iter * 100
                            str_bar = '>' * (int(np.ceil(
                                percent_iter / 2))) + ' ' * (int(
                                    (100 - percent_iter) // 2))
                            prColor(
                                '\r' + str_bar + 'processing: [%3.1f%%] ' %
                                (percent_iter), 'purple')

                        except Exception as e:
                            prColor(f'Error in the parallel calculation: {e}', 'red')

                disp_y_list = [item[0] for item in result_list]
                disp_x_list = [item[1] for item in result_list]
                y_list = [item[2] for item in result_list]
                df_nd_list = [item[3] for item in result_list]

                displace_y = np.zeros((dim[0], dim[1]))
                displace_x = np.zeros((dim[0], dim[1]))
                darkfield_nd = np.zeros((dim[0], dim[1]))

                for y, disp_x, disp_y, df_nd in zip(y_list, disp_x_list, disp_y_list, df_nd_list):
                    displace_x[y, :] = disp_x
                    displace_y[y, :] = disp_y
                    darkfield_nd[y, :] = df_nd

                displace = [
                    np.fmax(
                        np.fmin(displace_y, self.cal_half_window / 2**p_level),
                        -self.cal_half_window / 2**p_level),
                    np.fmax(
                        np.fmin(displace_x, self.cal_half_window / 2**p_level),
                        -self.cal_half_window / 2**p_level)
                ]
                prColor('displace map wrapping: {}'.format(displace[0].shape),
                        'green')
                print('max of displace: {}, min of displace: {}'.format(
                    np.amax(displace[0]), np.amin(displace[1])))

        _t_displace_end = time.time()
        self.displace_time = _t_displace_end - _t_displace_start
        prColor('displace time: {:.6f} s'.format(self.displace_time), 'light_purple')

        # remove the padding boundary of the displacement
        displace[0] = -displace[0][self.cal_half_window:-self.cal_half_window,
                                   self.cal_half_window:-self.cal_half_window]
        displace[1] = -displace[1][self.cal_half_window:-self.cal_half_window,
                                   self.cal_half_window:-self.cal_half_window]
        darkfield_nd = darkfield_nd[self.cal_half_window:-self.cal_half_window,
                                    self.cal_half_window:-self.cal_half_window]

        _t_post_start = time.time()
        DPC_y = (displace[0] - np.mean(displace[0])) * self.p_x / (self.z / self.Mag_factor)
        DPC_x = (displace[1] - np.mean(displace[1])) * self.p_x / (self.z / self.Mag_factor)

        phase = -frankotchellappa(
            DPC_x, DPC_y) * self.p_x * 2 * np.pi / self.wavelength
        _t_post_end = time.time()
        self.postprocess_time = _t_post_end - _t_post_start
        prColor('post-process time: {:.6f} s'.format(self.postprocess_time), 'light_purple')
        _t_solver_end = time.time()
        self.time_cost = _t_solver_end - _t_solver_start
        prColor('total time: {:.6f} s'.format(self.time_cost), 'light_purple')
        prColor('  pyramid:    {:.6f} s'.format(self.pyramid_time), 'light_purple')
        prColor('  wavelet:    {:.6f} s'.format(self._wavelet_time), 'light_purple')
        prColor('  darkfield:  {:.6f} s'.format(self.darkfield_time), 'light_purple')
        prColor('  displace:   {:.6f} s'.format(self.displace_time), 'light_purple')
        prColor('  post-proc:  {:.6f} s'.format(self.postprocess_time), 'light_purple')

        return displace, [DPC_y, DPC_x], phase, transmission, darkfield, darkfield_nd

    def run(self, result_path=None, cleansave=False):

        if self.use_GPU:

            self.displace, self.DPC, self.phase, self.transmission, self.darkfield, self.darkfield_nd = self.solver_cuda(
            )
        else:
            self.displace, self.DPC, self.phase, self.transmission, self.darkfield, self.darkfield_nd = self.solver(
            )

        if result_path is not None:
            '''
            save the calculation results
            '''
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            self.result_filename = 'WSVT_result'

            if cleansave:
                result_dict = {
                    'displace_x': self.displace[1].astype(np.float32),
                    'displace_y': self.displace[0].astype(np.float32),
                    'transmission_image': self.transmission.astype(np.float32),
                    'darkfield_nd':self.darkfield_nd.astype(np.float32)
                }
                if self.darkfield is not None:
                    result_dict['darkfield'] = self.darkfield.astype(np.float32)
                write_h5(result_path, self.result_filename, result_dict)
            else:
                result_dict = {
                    'displace_x': self.displace[1],
                    'displace_y': self.displace[0],
                    'DPC_x': self.DPC[1],
                    'DPC_y': self.DPC[0],
                    'phase': self.phase,
                    'transmission_image': self.transmission,
                    'darkfield_nd':self.darkfield_nd
                }
                if self.darkfield is not None:
                    result_dict['darkfield'] = self.darkfield
                write_h5(result_path, self.result_filename, result_dict)
            
            # Save T and D images if calculated
            if self.calc_T_D and self.save_plots and self.darkfield is not None:
                plt.imsave(os.path.join(result_path, 'transmission.png'), self.transmission, cmap='gray')
                plt.imsave(os.path.join(result_path, 'darkfield.png'), self.darkfield, cmap='gray')
                plt.imsave(os.path.join(result_path, 'darkfield_nd.png'), self.darkfield_nd, cmap='gray')
                prColor('Transmission and Darkfield images saved.', 'green')

            phase_rms = np.std(self.phase)
            phase_pv = np.amax(self.phase) - np.amin(self.phase)
            prColor(
                'phase rms: {:0.2f},  phase PV: {:0.2f}'.format(
                    phase_rms, phase_pv), 'purple')
            parameter_dict = {
                'crop': self.crop,
                'N_s extend': self.N_s_extend,
                'half_window': self.cal_half_window,
                'energy': self.energy,
                'wavelength': self.wavelength,
                'p_x': self.p_x,
                'd': self.z,
                'cpu_cores': self.n_cores,
                'n_group': self.n_group,
                'wavelet_level': self.wavelet_level,
                'pyramid_level': self.pyramid_level,
                'n_iter': self.n_iter,
                'time_cost': self.time_cost,
                'pyramid_time': self.pyramid_time,
                'wavelet_time': self._wavelet_time,
                'displace_time': self.displace_time,
                'postprocess_time': self.postprocess_time,
                'darkfield_time': self.darkfield_time,
                'calc_darkfield': self.calc_darkfield,
                'use_wavelet': self.use_wavelet,
                'wavelet_backend': self.wavelet_backend,
                'use_GPU': self.use_GPU,
                'wavelet_level_cut': self.wavelet_level_cut,
                'wavelet_add': self.wavelet_add_list,
                'phase_rms': phase_rms,
                'phase_pv': phase_pv
            }

            write_json(result_path, self.result_filename, parameter_dict)

    
def img_save(folder_path, filename, img):
    if not os.path.exists(os.path.join(folder_path)):
        os.makedirs(folder_path)
    # np.savetxt(os.path.join(folder_path, filename+'.npy.gz'), img)
    # np.save(os.path.join(folder_path, filename+'.npy'), img)
    # print(img.shape)

    im = Image.fromarray(img)
    im.save(os.path.join(folder_path, filename+'.tif'), save_all=True)

def main():
    # print(sys.argv)
    # parameter settings
    parser = argparse.ArgumentParser(
        description='experimental data analysis for WSVT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ============================================================
    parser.add_argument('--folder_ref',
                        type=str,
                        default='./data/',
                        help='reference images folder path')
    parser.add_argument('--folder_img',
                        type=str,
                        default='./data/',
                        help='sample images folder path')
    parser.add_argument('--file_ref',
                        nargs = '+',
                        type=str,
                        default='./data/ref_001.tif',
                        help='reference images; it should be a list of pure filename')
    parser.add_argument('--file_img',
                        nargs = '+',
                        type=str,
                        default='./data/sample_001.tif',
                        help='sample image; it should be a list of pure filename')

    parser.add_argument('--dark', type = str, default = 'None', help='dark image for image correction')
    parser.add_argument('--flat', type = str, default = 'None', help='flat image for image correction')

    parser.add_argument('--result_folder',
                        type=str,
                        default='./data_result/',
                        help='result_folder; where to save all the images and processed data')
    parser.add_argument('--cleansave',
                        default=0,
                        type=int,
                        help='1:only save dx,dy in h5; 0: save figures and full h5')
    
    parser.add_argument('--cal_half_window',
                        default=10,
                        type=int,
                        help='half windows size for displacement searching window')
    parser.add_argument('--N_s_extend',
                        default=4,
                        type=int,
                        help='the calculation window for high order pyramid')
    parser.add_argument('--n_scan',
                        default=20,
                        type=int,
                        help='scan number for calculation')
    parser.add_argument('--energy',
                        default=12e3,
                        type=float,
                        help='X ray energy in [eV]')
    parser.add_argument('--distance',
                        default=0.5,
                        type=float,
                        help='mask to detector distance in [m]')
    parser.add_argument('--p_x',
                        default=0.65e-6,
                        type=float,
                        help='detector pixel size in [m]')
    parser.add_argument('--Mag',
                        default=1,
                        type=float,
                        help='magnification factor for detector vs mask position')
    parser.add_argument('--n_cores',
                        default=4,
                        type=int,
                        help='process number for parallel')
    parser.add_argument('--n_group',
                        default=1,
                        type=int,
                        help='number to reduce the each memory use')
    
    parser.add_argument('--use_wavelet',
                        action='store_true',
                        default=False,
                        help='use wavelet or not')
    parser.add_argument('--use_c',
                        default=True,
                        action=argparse.BooleanOptionalAction,
                        help='use C solver for acceleration or not.')
    parser.add_argument('--wavelet_level_cut',
                        default=0,
                        type=int,
                        help='wavelet cutting to save time')
    parser.add_argument('--wavelet_chunk_size',
                        default=0,
                        type=int,
                        help='GPU wavelet chunk size (0 for auto). Larger values are faster but use more GPU memory.')
    parser.add_argument('--wavelet_backend',
                        type=str,
                        choices=['auto', 'c', 'pywt'],
                        default='auto',
                        help='CPU wavelet backend: auto (prefer C when available), c (force C), pywt (force python wavelet path).')
    parser.add_argument('--pyramid_level',
                        default=1,
                        type=int,
                        help='pyramid level to wrap the images')
    parser.add_argument('--n_iter',
                        default=1,
                        type=int,
                        help='iteration number, 1 should be enough')
    
    parser.add_argument('--use_GPU',
                        action='store_true',
                        default=False,
                        help='use GPU or not')
    
    parser.add_argument('--calc_T_D',
                        action='store_true',
                        default=False,
                        help='calculate transmission and darkfield using warped reference')
    parser.add_argument('--skip_darkfield',
                        action='store_false',
                        dest='calc_darkfield',
                        default=True,
                        help='skip raw darkfield reconstruction')

    parser.add_argument('--do_alignment',
                        action='store_true',
                        default=False,
                        help='perform image alignment between sample and reference')

    parser.add_argument('--use_fft_corr',
                        action='store_true',
                        default=False,
                        help='use FFT-based correlation (better for large search ranges >= 15 pixels)')

    parser.add_argument('--subpixel_method',
                        type=str,
                        choices=['parabolic', 'gaussian'],
                        default='parabolic',
                        help='sub-pixel refinement method: parabolic (classic) or gaussian (more accurate)')

    parser.add_argument('--use_ncc',
                        action='store_true',
                        default=False,
                        help='use normalized cross-correlation (NCC) for robust matching')

    parser.add_argument('--filter_outliers',
                        action='store_true',
                        default=False,
                        help='apply outlier filtering to displacement using median filter and MAD threshold')
    parser.add_argument('--memory_efficient',
                        action='store_true',
                        default=False,
                        help='reduce GPU memory usage by streaming peak search (disables GPU subpixel refinement)')
    parser.add_argument('--save_plots',
                        action='store_true',
                        default=False,
                        help='save PNG plots (disabled by default for faster runs)')

    parser.add_argument('--scan_mode',
                        default=1,
                        type=int,
                        help='n for 1st dim points, 1 for 1D linear scan')
    
    # parameters for ML solver
    parser.add_argument("--N_template",
                        type=int,
                        default=0,
                        help='template list for multi-resolution; it will use each template size in the list together with the iteration number of ML_iter for multi-resolution calculation; most of time, single value is ok')
    
    # image crop
    parser.add_argument(
        "--crop",
        nargs="+",
        type=int,
        default=[256],
        help=
        'image crop, if is [256], central crop. if len()==4, boundary crop, if is 0, use gui crop; if is -1, full image'
    )

    args = parser.parse_args()
    # print(args)

    args.file_ref = [os.path.join(args.folder_ref, item) for item in args.file_ref]
    args.file_img = [os.path.join(args.folder_img, item) for item in args.file_img]
    num_files = len(args.file_ref)

    if args.n_scan > num_files:
        files_ref_list = args.file_ref
        files_img_list = args.file_img
    else:
        if args.scan_mode == 1:
            # if the scan mode is linear, select the first part
            files_ref_list = args.file_ref[0:args.n_scan]
            files_img_list = args.file_img[0:args.n_scan]
        else:
            # if the scan mode is XY 2D square scan, select the 2D subpart with same step size
            num_2ndDim_steps = int(num_files/args.scan_mode)
            n_scan_1stDim = int(np.sqrt(args.n_scan))
            sub_id = []
            for ii in range(args.scan_mode):
                sub_id += [ii*args.scan_mode + jj for jj in range(n_scan_1stDim)]
            sub_id = sub_id[0:args.n_scan]
            print('Scan mode\n1st Dim: {}; 2nd Dim: {}'.format(args.scan_mode, num_2ndDim_steps))
            print('use the following XY scan: ', sub_id)
            files_ref_list = [args.file_ref[id] for id in sub_id]
            files_img_list = [args.file_img[id] for id in sub_id]

    prColor('sample imgs: {}\n'.format(files_img_list), 'green')
    prColor('ref imgs: {}\n'.format(files_ref_list), 'green')
    prColor('result folder: {}\n'.format(args.result_folder), 'green')
    
    wavelength = sc.value('inverse meter-electron volt relationship') / args.energy
    
    use_estimate = False

    ref_data = load_image(files_ref_list)
    img_data = load_image(files_img_list)
    
    if args.flat != 'None':
        flat = load_image(args.flat)
        if len(flat.shape) == 3:
            flat = np.mean(flat, axis=0)
        prColor('flat image size: {}'.format(flat.shape), 'green')
    else:
        flat = 1
    if args.dark != 'None':
        dark = load_image(args.dark)
        if len(dark.shape) == 3:
            dark = np.mean(dark, axis=0)
        prColor('dark image size: {}'.format(dark.shape), 'green')
    else:
        dark = 0

    ref_data = (ref_data - dark) / (flat - dark) * 2**16
    img_data = (img_data - dark) / (flat - dark) * 2**16

    # take out the roi
    if len(args.crop) == 1:
        if args.crop[0] == 0:
            # use gui crop
            print("before crop------------------------------------------------")
            _, corner = crop_gui(img_data[0]/ref_data[0])
            print("after crop------------------------------------------------")
            
            crop_index = [
                int(corner[0][0]),
                int(corner[1][0]),
                int(corner[0][1]),
                int(corner[1][1])
            ]

            boundary_crop = lambda img: img[:, int(crop_index[0]):int(crop_index[1]),int(crop_index[2]):int(crop_index[3])]

            ref_data = boundary_crop(ref_data)
            img_data = boundary_crop(img_data)
            crop = crop_index
        elif args.crop[0] == -1:
            # use full image
            pass
        else:
            ref_data = image_roi(ref_data, args.crop[0])
            img_data = image_roi(img_data, args.crop[0])
    elif len(args.crop) == 4:
        boundary_crop = lambda img: img[:, int(args.crop[0]):int(args.crop[1]),int(args.crop[2]):int(args.crop[3])]
        ref_data = boundary_crop(ref_data)
        img_data = boundary_crop(img_data)
    else:
        prColor('wrong length for crop. Need to be 4 for corner crop; -1 for full image; 0 for gui crop; M for central crop')

    
    # do alignment 
    if args.do_alignment:
        print('do alignment')
        from skimage.registration import phase_cross_correlation
        from scipy.ndimage import fourier_shift

        def stack_image_align(image_stack, offset_image_stack, method='pc'):
            '''
                here's a function to do the alignment of two images.
                the offset_image is shifted relatively to image to find the best position
                and return the shifted back offset_image
                input:
                    image:              the first image stack
                    offset_image:       the second image stack
                output:
                    pos:                best shift postion to maximize the correlation
                    image_back:         the image after alignment
            '''
            if method == 'pc':
                # from skimage.feature import register_translation
                roi = lambda x: x[0:50][0:50]
                # shift, error, diffphase = register_translation(image, offset_image, 10)
                # shift, error, diffphase = phase_cross_correlation(image_stack, offset_image_stack, upsample_factor=10)

                # print('shift dist: {}, alignment error: {} and phase difference: {}'.format(shift, error, diffphase))
                # # image_back = image_shift(offset_image, shift[0], shift[1])
                # image_back = np.array([fourier_shift(np.fft.fftn(img), shift) for img in offset_image_stack])
                # image_back = np.real(np.fft.ifftn(image_back, axes=(-1, -2)))

                normlize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
                detected_shift, error, phasediff = phase_cross_correlation(
                    normlize(roi(image_stack[0], 100)) * 65525,
                    normlize(roi(offset_image_stack[0], 100))* 65525,
                    upsample_factor=100,
                    # normalization="phase"
                )

                print('shift dist: {}, alignment error: {} and phase difference: {}'.format(detected_shift, error, phasediff))
                # --- 3. 执行配准 (Registration) (使用 SciPy) ---
                # 要将 shifted_image 移回去对齐 reference_image，我们需要施加一个反向的位移
                correction_shift = np.array(detected_shift)

                # 使用 fourier_shift 来应用这个纠正位移
                # 这比 warpAffine 对于纯平移更精确
                image_back = np.array([fourier_shift(np.fft.fftn(img), correction_shift) for img in offset_image_stack])
                image_back = np.real(np.fft.ifftn(image_back, axes=(-1, -2)))
                # registered_image_fft = fourier_shift(np.fft.fftn(offset_image_stack), correction_shift)
                # image_back = np.fft.ifftn(registered_image_fft).real
            elif method == 'optflow':
                from .func import slop_tracking
                disp = slop_tracking(image_stack[0], offset_image_stack[0], n_window=15)
                x_mean = np.mean(disp[0][10:50, 10:50])
                y_mean = np.mean(disp[1][10:50, 10:50])
                print('shift dist: {}'.format([-x_mean, -y_mean]))
                # registered_image_fft = fourier_shift(np.fft.fftn(offset_image_stack), [-x_mean, -y_mean])
                # image_back = np.fft.ifftn(registered_image_fft).real

                correction_shift = [-x_mean, -y_mean]

                image_back = np.array([np.fft.ifftn(fourier_shift(np.fft.fftn(img), correction_shift)).real for img in offset_image_stack])
                # image_back = np.real(np.fft.ifftn(image_back, axes=(-1, -2)))

            return correction_shift, image_back
        
        pos_shift, ref_data = stack_image_align(img_data, ref_data, 'optflow')

    ref_data = ref_data.astype(np.float32)
    img_data = img_data.astype(np.float32)
    # from scipy.ndimage import zoom
    # if True:
    #     ref_data = zoom(ref_data, (1, 2, 2), order=1)
    #     img_data = zoom(img_data, (1, 2, 2), order=1)

    # if args.n_scan > ref_data.shape[0]:
    #     pass
    # else:
    #     if args.scan_mode == 1:
    #         # if the scan mode is linear, select the first part
    #         ref_data = ref_data[0:args.n_scan, :, :]
    #         img_data = img_data[0:args.n_scan, :, :]
    #     else:
    #         # if the scan mode is XY 2D square scan, select the 2D subpart with same step size
    #         num_2ndDim_steps = int(ref_data.shape[0]/args.scan_mode)
    #         n_scan_1stDim = int(np.sqrt(args.n_scan))
    #         sub_id = []
    #         for ii in range(args.scan_mode):
    #             sub_id += [ii*args.scan_mode + jj for jj in range(n_scan_1stDim)]
    #         sub_id = sub_id[0:args.n_scan]
    #         print('Scan mode\n1st Dim: {}; 2nd Dim: {}'.format(args.scan_mode, num_2ndDim_steps))
    #         print('use the following XY scan: ', sub_id)
    #         ref_data = ref_data[sub_id, :, :]
    #         img_data = img_data[sub_id, :, :]
            
    print("use {} scan position for calculation".format(ref_data.shape[0]))

    WSVT_solver = WSVT(img_data,
                       ref_data,
                       crop=args.crop,
                       N_s_extend=args.N_s_extend,
                       cal_half_window=args.cal_half_window,
                       N_template=args.N_template,
                       n_cores=args.n_cores,
                       n_group=args.n_group,
                       energy=args.energy,
                       p_x=args.p_x,
                       Mag_factor=args.Mag,
                       z=args.distance,
                       wavelet_level_cut=args.wavelet_level_cut,
                       pyramid_level=args.pyramid_level,
                       n_iter=args.n_iter,
                       use_estimate=False,
                       use_wavelet=args.use_wavelet,
                       use_GPU=args.use_GPU,
                       use_c=args.use_c,
                       calc_T_D=args.calc_T_D,
                       calc_darkfield=args.calc_darkfield,
                       use_fft_corr=args.use_fft_corr,
                       subpixel_method=args.subpixel_method,
                       use_ncc=args.use_ncc,
                       filter_outliers=args.filter_outliers,
                       memory_efficient=args.memory_efficient,
                       save_plots=args.save_plots,
                       wavelet_chunk_size=args.wavelet_chunk_size,
                       wavelet_backend=args.wavelet_backend)

    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)
    
    WSVT_solver.run(result_path=args.result_folder, cleansave=args.cleansave)
    displace = WSVT_solver.displace
    DPC_x = WSVT_solver.DPC[1]
    DPC_y = WSVT_solver.DPC[0]
    phase = WSVT_solver.phase
    result_filename = WSVT_solver.result_filename
    sample_transmission = WSVT_solver.transmission
    sample_darkfield = WSVT_solver.darkfield

    if not args.cleansave:
        img_save(args.result_folder, 'displace_y', displace[0])
        img_save(args.result_folder, 'displace_x', displace[1])
        img_save(args.result_folder, 'transmission', sample_transmission)
        img_save(args.result_folder, 'dark_field', sample_darkfield)
        img_save(args.result_folder, 'phase', phase)
        if args.save_plots:
            plt.figure()
            plt.imshow(displace[0])
            cbar = plt.colorbar()
            cbar.set_label('[pixels]', rotation=90)
            plt.savefig(os.path.join(args.result_folder, 'displace_y_colorbar.png'),
                        dpi=150)
            plt.figure()
            plt.imshow(displace[1])
            cbar = plt.colorbar()
            cbar.set_label('[pixels]', rotation=90)
            plt.savefig(os.path.join(args.result_folder, 'displace_x_colorbar.png'),
                        dpi=150)
            plt.figure()
            plt.imshow(phase)
            cbar = plt.colorbar()
            cbar.set_label('[rad]', rotation=90)
            plt.savefig(os.path.join(args.result_folder, 'phase_colorbar.png'), dpi=150)

            plt.figure()
            plt.imshow(sample_transmission)
            cbar = plt.colorbar()
            cbar.set_label('[a.u.]', rotation=90)
            plt.savefig(os.path.join(args.result_folder, 'rtrans_colorbar.png'), dpi=150)
            
            plt.figure()
            plt.imshow(sample_darkfield)
            cbar = plt.colorbar()
            cbar.set_label('[a.u.]', rotation=90)
            plt.savefig(os.path.join(args.result_folder, 'df_colorbar.png'), dpi=150)
            plt.close()


if __name__ == "__main__":
    main()
