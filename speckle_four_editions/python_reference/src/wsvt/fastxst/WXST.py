#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Time    	: 08 / 08 / 2022
@Author  	: Zhi Qiao
@Contact	: z.qiao1989@gmail.com
@File    	: WSVT.py
@Software	: WaveletSBI
@Desc		: single-shot mode wavelet speckle tracking (WaveletSBI) script

1. Qiao, Zhi, Xianbo Shi, and Lahsen Assoufid. “Single-Shot Speckle Tracking Method Based on Wavelet Transform and Multi-Resolution Analysis.” In Advances in Metrology for X-Ray and EUV Optics IX, edited by Lahsen Assoufid, Haruhiko Ohashi, and Anand Asundi, 22. Online Only, United States: SPIE, 2020. https://doi.org/10.1117/12.2569135.
2. Qiao, Zhi, Xianbo Shi, Rafael Celestre, and Lahsen Assoufid. “Wavelet-Transform-Based Speckle Vector Tracking Method for X-Ray Phase Imaging.” Optics Express 28, no. 22 (October 26, 2020): 33053. https://doi.org/10.1364/OE.404606.

'''

import argparse
import numpy as np
import pywt
import os
import sys
import time
from PIL import Image
import glob
import torch
from torch import nn
import scipy.constants as sc
import scipy.ndimage as snd
from .func import prColor, frankotchellappa, image_roi, Wavelet_transform_multiprocess, write_h5, write_json, find_disp, image_align, auto_crop, WaveletTransformGPU
from .gui_func import crop_gui
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


def load_image(file_path):
    """
    load_images to load images

    Args:
        Folder_path (str): image folder
        filename_format (str, optional): image filename and extension. Defaults to '*.tif'.

    Returns:
        image data
    """
    if os.path.exists(file_path):
        img = np.array(Image.open(file_path))
    else:
        prColor('Error: wrong data path. No data is loaded.', 'red')
        sys.exit()
    return np.array(img)


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


def cost_volume(first, second, search_range, use_ncc=False):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
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
    chunk_size = 200
    num_chunks = (C + chunk_size - 1) // chunk_size
    
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
    """SSD cost volume centered on each target pixel's integer displacement."""
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
        sample_y = base_y + center_y + (oy - search_range)
        for ox in range(window_size):
            sample_x = base_x + center_x + (ox - search_range)
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
                gathered = torch.where(
                    valid.unsqueeze(0), gathered, torch.zeros_like(gathered)
                )
                diff = first_f32[start:end] - gathered
                ssd += torch.sum(diff * diff, dim=0)
            cost_vol[out_index] = -ssd
            out_index += 1

    return cost_vol


def cost_volume_peak_streaming(first, second, search_range, use_ncc=False):
    """Memory-efficient peak search without materializing full cost volume.
    Returns max_val and max_idx maps matching torch.max(cost_vol, dim=0).
    """
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
    # Use scipy.ndimage for efficient median filtering
    import scipy.ndimage as ndimage
    
    # Calculate local median
    median_y = ndimage.median_filter(disp_y, size=kernel_size)
    median_x = ndimage.median_filter(disp_x, size=kernel_size)
    
    # Calculate deviation from median
    diff_y = np.abs(disp_y - median_y)
    diff_x = np.abs(disp_x - median_x)
    
    # Calculate MAD (Median Absolute Deviation) - robust estimator of sigma
    # MAD = median(|x - median(x)|)
    # We can estimate it globally or locally. Locally is better for varying noise.
    # But local MAD is expensive. Let's use global MAD of the difference.
    mad_y = np.median(diff_y)
    mad_x = np.median(diff_x)
    
    # Avoid zero division
    mad_y = max(mad_y, 1e-6)
    mad_x = max(mad_x, 1e-6)
    
    # Identify outliers
    # A pixel is an outlier if it deviates from local median by more than threshold * MAD
    # We use a slightly more robust check: deviation > threshold * MAD + small_epsilon
    mask_y = diff_y > (threshold * mad_y + 0.1) # 0.1 pixel tolerance
    mask_x = diff_x > (threshold * mad_x + 0.1)
    
    outlier_mask = mask_y | mask_x
    
    # Replace outliers with median values
    filtered_y = np.copy(disp_y)
    filtered_x = np.copy(disp_x)
    
    filtered_y[outlier_mask] = median_y[outlier_mask]
    filtered_x[outlier_mask] = median_x[outlier_mask]
    
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
        # Match numpy.pad(..., mode='edge') and the C solver at search
        # boundaries.  Clamping a flattened index would wrap x-neighbours
        # into the preceding/following y row.
        neighbour_y = torch.clamp(max_wy + offset_y, 0, window_size - 1)
        neighbour_x = torch.clamp(max_wx + offset_x, 0, window_size - 1)
        idx = neighbour_y * window_size + neighbour_x
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
    
    nonzero_det = det != 0
    safe_det = torch.where(nonzero_det, det, torch.ones_like(det))
    dx = torch.where(
        nonzero_det,
        -(grad_yy * grad_x - grad_xy * grad_y) / safe_det,
        torch.zeros_like(det),
    )
    dy = torch.where(
        nonzero_det,
        -(grad_xx * grad_y - grad_xy * grad_x) / safe_det,
        torch.zeros_like(det),
    )
    
    shift_y = torch.clamp(
        (max_wy.float() - cal_half_window) + dy,
        -cal_half_window,
        cal_half_window,
    )
    shift_x = torch.clamp(
        (max_wx.float() - cal_half_window) + dx,
        -cal_half_window,
        cal_half_window,
    )
    
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


class WXST:
    def __init__(self,
                 img,
                 ref,
                 M_image=512,
                 N_s=5,
                 cal_half_window=20,
                 N_s_extend=4,
                 n_cores=4,
                 n_group=4,
                 energy=14e3,
                 p_x=0.65e-6,
                 z=500e-3,
                 wavelet_level_cut=2,
                 pyramid_level=2,
                 n_iter=1,
                 use_estimate=False,
                 use_wavelet=True,
                 use_GPU=0,
                 use_c=True,
                 subpixel_method='parabolic',
                 use_ncc=False,
                 filter_outliers=False,
                 use_fft_corr=False,
                 memory_efficient=False,
                 wavelet_chunk_size=0,
                 wavelet_backend='auto'):
        self.img_data = img
        self.ref_data = ref
        self.use_c = use_c
        self.subpixel_method = subpixel_method
        self.use_ncc = use_ncc
        self.filter_outliers = filter_outliers
        self.use_fft_corr = use_fft_corr
        self.memory_efficient = memory_efficient
        self.wavelet_chunk_size = wavelet_chunk_size
        self.wavelet_backend = wavelet_backend
        self._wavelet_c_meta_cache = {}
        # roi of the images
        self.M_image = M_image
        # template window, the N_s nearby pixels used to represent the local pixel, 2*N_s+1
        self.N_s = N_s
        # the number of the area to calculate for each pixel, 2*cal_half_window X 2*cal_half_window
        self.cal_half_window = cal_half_window
        # the calculation window for high order pyramid
        self.N_s_extend = N_s_extend

        # process number for parallel
        self.n_cores = n_cores
        # number to reduce the each memory use
        self.n_group = n_group

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
        if torch.cuda.is_available() and use_GPU == 1:
            prColor('Use GPU found. Enable multi-resolution', 'cyan')
            self.use_GPU = True
            # for GPU, there's no multi-resolution process
            # self.pyramid_level = 0
        else:
            prColor('No gpu found. Use CPU instead.', 'cyan')
            self.use_GPU = False

        if self.use_estimate:
            # get initial estimation from the cv2 flow tracking
            displace_estimate = slope_tracking(self.ref_data,
                                               self.img_data,
                                               n_window=self.cal_half_window)
            self.displace_estimate = [
                displace_estimate[0], displace_estimate[1]
            ]
        else:
            m, n = self.img_data.shape
            self.displace_estimate = [np.zeros((m, n)), np.zeros((m, n))]

    def template_stack(self, img):
        '''
            stack the nearby pixels in 2*N_s+1
        '''
        img_stack = []
        axis_Nw = np.arange(-self.N_s, self.N_s + 1)
        for x in axis_Nw:
            for y in axis_Nw:
                img_stack.append(np.roll(np.roll(img, x, axis=0), y, axis=1))

        return np.array(img_stack)

    def stack_TemplateWindow_torch(self, img_tensor):
        """
        GPU version of template_stack
        img_tensor: [N, H, W] or [H, W]
        Returns: [N * num_shifts, H, W] or [num_shifts, H, W]
        """
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(0)
            
        N, H, W = img_tensor.shape
        N_t = self.N_s
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

        """
        ref_pyramid = []
        img_pyramid = []
        prColor(
            'obtain pyramid image and stack the window with pyramid level: {}'.
            format(self.pyramid_level), 'green')
        ref_pyramid.append(self.ref_data)
        img_pyramid.append(self.img_data)

        for kk in range(self.pyramid_level):
            ref_pyramid.append(
                pywt.dwtn(ref_pyramid[kk], 'db3', mode='zero',
                          axes=(-2, -1))['aa'])
            img_pyramid.append(
                pywt.dwtn(img_pyramid[kk], 'db3', mode='zero',
                          axes=(-2, -1))['aa'])

        normlize_std = lambda img: (
            (img - np.ndarray.mean(img, axis=0)) / np.ndarray.std(img, axis=0))

        ref_pyramid = [
            normlize_std(self.template_stack(img_data))
            for img_data in ref_pyramid
        ]
        img_pyramid = [
            normlize_std(self.template_stack(img_data))
            for img_data in img_pyramid
        ]

        return ref_pyramid, img_pyramid

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

        n_workers = self.n_cores
        if n_workers > h:
            n_workers = h
        chunk_size = max(1, h // n_workers)

        futures = []
        results = [None] * n_workers

        def process_chunk(idx, start_h, end_h):
            sub_img = img[:, start_h:end_h, :]
            sub_img_t = np.ascontiguousarray(np.moveaxis(sub_img, 0, -1), dtype=np.float32)
            sub_h = end_h - start_h
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
            for i in range(n_workers):
                start_h = i * chunk_size
                end_h = (i + 1) * chunk_size if i < n_workers - 1 else h
                if start_h >= end_h:
                    continue
                futures.append(executor.submit(process_chunk, i, start_h, end_h))

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
        N, H, W = img_data.shape

        # Move to GPU and reshape to (H*W, N)
        img_tensor = torch.from_numpy(img_data).float().cuda()
        img_reshaped = img_tensor.permute(1, 2, 0).reshape(-1, N)

        num_pixels = img_reshaped.shape[0]

        if self.wavelet_chunk_size and self.wavelet_chunk_size > 0:
            chunk_size = int(self.wavelet_chunk_size)
        else:
            try:
                free_mem, _ = torch.cuda.mem_get_info()
                target_bytes = int(free_mem * 0.12)
                bytes_per_sample = max(1, N * 2 * 4)
                chunk_size = max(20000, min(250000, target_bytes // bytes_per_sample))
            except Exception:
                chunk_size = 120000

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

        C_out = coeffs_combined.shape[1]
        output = coeffs_combined.view(H, W, C_out).permute(2, 0, 1)
        del coeffs_combined, img_reshaped, img_tensor
        
        # Generate level names
        level_name = []
        for kk in range(w_level):
            level_name.append('D{:d}'.format(kk + 1))
        level_name.append('A{:d}'.format(w_level))
        level_name = level_name[-return_level:]
        
        return output, level_name

    def wavelet_data(self):
        self.pyramid_time = 0.0
        self._wavelet_time = 0.0
        _t_pyramid_start = time.time()
        ref_pyramid, img_pyramid = self.pyramid_data()
        _t_pyramid_end = time.time()
        self.pyramid_time = _t_pyramid_end - _t_pyramid_start
        prColor('pyramid time: {:.6f} s'.format(self.pyramid_time), 'light_purple')
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
            self._wavelet_time = end_time - start_time
            print('wavelet time: {}'.format(self._wavelet_time))
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
            img_stack_cuda = img_wa_stack.float()
            ref_stack_cuda = ref_wa_stack.float()
        else:
            img_stack_cuda = torch.from_numpy(np.moveaxis(img_wa_stack, -1, 0)).cuda().float()
            ref_stack_cuda = torch.from_numpy(np.moveaxis(ref_wa_stack, -1, 0)).cuda().float()

        window_size = 2 * cal_half_window + 1

        if center_y is not None or center_x is not None:
            if center_y is None or center_x is None:
                raise ValueError('center_y and center_x must be provided together')
            if self.use_ncc:
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
        
        if return_half:
            return warped.squeeze(0).half() # Return as half
        else:
            return warped.squeeze(0).float()

    def solver_cuda(self):
        """
        speckle tracking solver using GPU
        """
        ref_pyramid, img_pyramid = self.wavelet_data()

        # Backup data for transmission calculation
        img_raw = self.img_data
        ref_raw = self.ref_data

        # transmission = self.img_data / (np.abs(self.ref_data + 1))
        for attr in ('img_data', 'ref_data'):
            self.__dict__.pop(attr, None)

        start_time = time.time()
        
        # Initialize displacement
        current_disp_y = None
        current_disp_x = None
        
        # Iterate from coarsest to finest
        num_levels = len(ref_pyramid)
        
        for level in range(num_levels - 1, -1, -1):
            img_wa = img_pyramid[level]
            ref_wa = ref_pyramid[level]
            
            # Determine search window for this level
            if level == num_levels - 1:
                search_range = int(np.ceil(self.cal_half_window / (2**level)))
            else:
                search_range = self.N_s_extend
            
            prColor(f'Processing level {level}, search range {search_range}', 'cyan')
            
            # Prepare GPU tensors
            if torch.is_tensor(img_wa):
                img_tensor = img_wa.float()
                ref_tensor = ref_wa.float()
            else:
                img_tensor = torch.from_numpy(np.moveaxis(img_wa, -1, 0)).float().cuda()
                ref_tensor = torch.from_numpy(np.moveaxis(ref_wa, -1, 0)).float().cuda()
            
            if current_disp_y is not None:
                # Match the CPU solver's spline, scale, round, and clip guess.
                H, W = img_tensor.shape[1], img_tensor.shape[2]
                guess_y_np = np.round(self.resampling_spline(
                    current_disp_y.detach().cpu().numpy() * 2, (H, W)))
                guess_x_np = np.round(self.resampling_spline(
                    current_disp_x.detach().cpu().numpy() * 2, (H, W)))
                guess_limit = self.cal_half_window / (2**level)
                guess_y_np = np.clip(guess_y_np, -guess_limit, guess_limit).astype(np.float32)
                guess_x_np = np.clip(guess_x_np, -guess_limit, guess_limit).astype(np.float32)
                current_disp_y = torch.from_numpy(guess_y_np).to(img_tensor.device)
                current_disp_x = torch.from_numpy(guess_x_np).to(img_tensor.device)

                res_y, res_x, confidence = self.displace_torch(
                    img_tensor, ref_tensor, search_range,
                    center_y=current_disp_y, center_x=current_disp_x)
                
                # Update
                current_disp_y += res_y
                current_disp_x += res_x
            else:
                # First level
                current_disp_y, current_disp_x, confidence = self.displace_torch(img_tensor, ref_tensor, search_range)

            level_limit = self.cal_half_window / (2**level)
            current_disp_y = torch.clamp(current_disp_y, -level_limit, level_limit)
            current_disp_x = torch.clamp(current_disp_x, -level_limit, level_limit)

            if self.memory_efficient:
                del img_tensor, ref_tensor
                if 'warped_ref' in locals():
                    del warped_ref
                torch.cuda.empty_cache()

        # Save confidence as darkfield_nd
        darkfield_nd = confidence.cpu().numpy()

        end_time = time.time()
        prColor('GPU calculation time: {}'.format(end_time - start_time), 'cyan')
        
        displace_y = current_disp_y.cpu().numpy()
        displace_x = current_disp_x.cpu().numpy()

        # Preserve the original Python WXST full-ROI transmission semantics.
        ref_safe = np.clip(ref_raw.astype(np.float64), 1e-10, None)
        transmission = np.clip(img_raw.astype(np.float64) / ref_safe, 0.01, 10.0)

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

        DPC_y = (displace[0] - np.mean(displace[0])) * self.p_x / self.z
        DPC_x = (displace[1] - np.mean(displace[1])) * self.p_x / self.z

        phase = -frankotchellappa(
            DPC_x, DPC_y) * self.p_x * 2 * np.pi / self.wavelength
        self.time_cost = end_time - start_time

        return displace, [DPC_y, DPC_x], phase, transmission, darkfield_nd

    def solver(self):
        _t_solver_start = time.time()
        ref_pyramid, img_pyramid = self.wavelet_data()

        # Backup data
        img_raw = self.img_data
        ref_raw = self.ref_data

        # transmission = self.img_data / (np.abs(self.ref_data + 1))
        for attr in ('img_data', 'ref_data'):
            self.__dict__.pop(attr, None)

        cores = ms.cpu_count()
        prColor('Computer available cores: {}'.format(cores), 'green')

        if cores > self.n_cores:
            cores = self.n_cores
        else:
            cores = ms.cpu_count()
        prColor('Use {} cores'.format(cores), 'light_purple')
        prColor('Process group number: {}'.format(self.n_group),
                'light_purple')

        if cores * self.n_group > self.M_image:
            n_tasks = 4
        else:
            n_tasks = cores * self.n_group

        _t_displace_start = time.time()
        # use pyramid wrapping
        max_pyramid_searching_window = int(
            np.ceil(self.cal_half_window / 2**self.pyramid_level))
        searching_window_pyramid_list = [self.N_s_extend
                                         ] * self.pyramid_level + [
                                             int(max_pyramid_searching_window)
                                         ]

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
                        np.round(self.resampling_spline(img * 2, (m, n)))
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

                        except Exception as exc:
                            for pending in futures:
                                pending.cancel()
                            raise RuntimeError(
                                'FastXST WXST parallel displacement task failed'
                            ) from exc

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

        _t_post_start = time.time()

        # Preserve the original Python WXST full-ROI transmission semantics.
        ref_safe = np.clip(ref_raw.astype(np.float64), 1e-10, None)
        transmission = np.clip(img_raw.astype(np.float64) / ref_safe, 0.01, 10.0)

        # remove the padding boundary of the displacement
        displace[0] = -displace[0][self.cal_half_window:-self.cal_half_window,
                                   self.cal_half_window:-self.cal_half_window]
        displace[1] = -displace[1][self.cal_half_window:-self.cal_half_window,
                                   self.cal_half_window:-self.cal_half_window]
        darkfield_nd = darkfield_nd[self.cal_half_window:-self.cal_half_window,
                                    self.cal_half_window:-self.cal_half_window]

        DPC_y = (displace[0] - np.mean(displace[0])) * self.p_x / self.z
        DPC_x = (displace[1] - np.mean(displace[1])) * self.p_x / self.z

        phase = -frankotchellappa(
            DPC_x, DPC_y) * self.p_x * 2 * np.pi / self.wavelength

        _t_solver_end = time.time()
        self.postprocess_time = _t_solver_end - _t_post_start
        self.time_cost = _t_solver_end - _t_solver_start

        prColor('total time: {:.6f} s'.format(self.time_cost), 'light_purple')
        prColor('  pyramid:    {:.6f} s'.format(self.pyramid_time), 'light_purple')
        prColor('  wavelet:    {:.6f} s'.format(self._wavelet_time), 'light_purple')
        prColor('  displace:   {:.6f} s'.format(self.displace_time), 'light_purple')
        prColor('  post-proc:  {:.6f} s'.format(self.postprocess_time), 'light_purple')

        return displace, [DPC_y, DPC_x], phase, transmission, darkfield_nd

    def run(self, result_path=None):
        if self.use_GPU:

            self.displace, self.DPC, self.phase, self.transmission, self.darkfield_nd = self.solver_cuda(
            )
        else:
            self.displace, self.DPC, self.phase, self.transmission, self.darkfield_nd = self.solver(
            )

        if result_path is not None:
            '''
            save the calculation results
            '''
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            self.result_filename = 'WXST_result'

            write_h5(
                result_path, self.result_filename, {
                    'displace_x': self.displace[1],
                    'displace_y': self.displace[0],
                    'DPC_x': self.DPC[1],
                    'DPC_y': self.DPC[0],
                    'phase': self.phase,
                    'transmission_image': self.transmission,
                    'darkfield_nd': self.darkfield_nd
                })

            parameter_dict = {
                'M_image': self.M_image,
                'template_window': self.N_s,
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
                'use_wavelet': self.use_wavelet,
                'wavelet_backend': self.wavelet_backend,
                'use_GPU': self.use_GPU,
                'wavelet_level_cut': self.wavelet_level_cut,
                'wavelet_add': self.wavelet_add_list
            }

            write_json(result_path, self.result_filename, parameter_dict)


def main():
    # paremater settings
    parser = argparse.ArgumentParser(
        description='experimental data analysis for absolute phase measurement',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # shared args
    # ============================================================
    parser.add_argument('--img',
                        type=str,
                        default='../testdata/single-shot/sample_001.tif',
                        help='path to sample image')
    parser.add_argument('--ref',
                        type=str,
                        default='../testdata/single-shot/ref_001.tif',
                        help='path to sample image')
    parser.add_argument('--result_folder',
                        type=str,
                        default='../testdata/single-shot/WXST_results',
                        help='saving folder')
    parser.add_argument(
        "--crop",
        nargs="+",
        type=int,
        default=[450, 1000, 500, 1000],
        help=
        'image crop, if is [256], central crop. if len()==4, boundary crop, if is 0, use gui crop, if is -1, use auto-crop'
    )
    parser.add_argument('--p_x',
                        default=0.65e-6,
                        type=float,
                        help='pixel size')
    parser.add_argument('--energy',
                        default=14e3,
                        type=float,
                        help='X-ray energy')
    parser.add_argument('--distance',
                        default=500e-3,
                        type=float,
                        help='detector to mask distance')
    parser.add_argument('--down_sampling', type=float, default=1, help='down-sample images to reduce memory cost and accelerate speed.')

    parser.add_argument('--GPU',
                        default=False,
                        action='store_true',
                        help='Use GPU or not. GPU can be 2 times faster. But multi-resolution process is disabled.')
    parser.add_argument('--use_wavelet',
                        default=False,
                        action='store_true',
                        help='use wavelet transform or not.')
    parser.add_argument('--use_c',
                        default=True,
                        action=argparse.BooleanOptionalAction,
                        help='use C solver for acceleration or not.')
    parser.add_argument('--wavelet_lv_cut',
                        default=2,
                        type=int,
                        help='wavelet cutting level')
    parser.add_argument('--pyramid_level',
                        default=1,
                        type=int,
                        help='pyramid level used for speckle tracking.')
    parser.add_argument('--n_iter',
                        default=1,
                        type=int,
                        help='number of iteration for speckle tracking. 1 is good.')
    parser.add_argument('--template_size',
                        default=5,
                        type=int,
                        help='template size in the WXST')
    parser.add_argument('--window_searching',
                        default=10,
                        type=int,
                        help='searching window of speckle tracking. Means the largest displacement can be calculated.')
    parser.add_argument('--nCores',
                        default=4,
                        type=int,
                        help='number of CPU cores used for calculation.')
    parser.add_argument('--nGroup',
                        default=1,
                        type=int,
                        help='number of groups that parallel calculation is splitted into.')
    parser.add_argument('--use_ncc',
                        default=False,
                        action='store_true',
                        help='Use Normalized Cross Correlation (NCC) instead of Euclidean distance.')
    parser.add_argument('--subpixel_method',
                        type=str,
                        default='gaussian',
                        choices=['gaussian', 'parabolic', 'center_of_mass'],
                        help='Subpixel refinement method.')
    parser.add_argument('--fft_corr',
                        default=False,
                        action='store_true',
                        help='Use FFT for correlation calculation (faster for large windows).')
    parser.add_argument('--wavelet_chunk_size',
                        default=0,
                        type=int,
                        help='GPU wavelet chunk size (0 for auto). Larger values are faster but use more GPU memory.')
    parser.add_argument('--wavelet_backend',
                        type=str,
                        choices=['auto', 'c', 'pywt'],
                        default='auto',
                        help='CPU wavelet backend: auto (prefer C when available), c (force C), pywt (force python wavelet path).')
    parser.add_argument('--memory_efficient',
                        default=False,
                        action='store_true',
                        help='Reduce GPU memory usage by streaming peak search (disables GPU subpixel refinement).')
    parser.add_argument('--save_plots',
                        default=False,
                        action='store_true',
                        help='Save PNG visualizations (disabled by default for speed).')

    args = parser.parse_args()

    for key, value in args.__dict__.items():
        prColor('{}: {}'.format(key, value), 'cyan')
    
    # if len(sys.argv) == 1:

    #     Folder_path = '../testdata/single-shot/'
    #     File_ref = os.path.join(Folder_path, 'ref_001.tif')
    #     File_img = os.path.join(Folder_path, 'sample_001.tif')

    #     Folder_result = os.path.join(Folder_path, 'XST_result')
    #     # [image_size, template_window, cal_half_window, n_group, n_cores, energy, pixel_size, distance, use_wavelet, wavelet_ct, pyramid level, n_iteration, use_GPU]
    #     parameter_wavelet = [
    #         1024, 5, 20, 4, 4, 20e3, 0.65e-6, 200e-3, 0, 1, 0, 1, 0, 1
    #     ]

    # elif len(sys.argv) == 4:
    #     File_img = sys.argv[1]
    #     File_ref = sys.argv[2]
    #     Folder_result = sys.argv[3]
    #     # [image_size, template_window, cal_half_window, n_group, n_cores, energy, pixel_size, distance, wavelet_ct, pyramid level, n_iteration, use_GPU]
    #     parameter_wavelet = [
    #         512, 5, 20, 4, 4, 14e3, 0.65e-6, 500e-3, 1, 2, 2, 1, 0, 1
    #     ]
    # elif len(sys.argv) == 18:
    #     File_img = sys.argv[1]
    #     File_ref = sys.argv[2]
    #     Folder_result = sys.argv[3]
    #     parameter_wavelet = sys.argv[4:]
    # else:
    #     prColor('Wrong parameters! should be: sample, ref, result', 'red')

    File_ref = args.ref
    File_img = args.img
    Folder_result = args.result_folder
    N_s = args.template_size
    cal_half_window = args.window_searching
    # the calculation window for high order pyramid
    N_s_extend = 4
    n_cores = args.nCores
    n_group = args.nGroup
    energy = args.energy
    wavelength = sc.value('inverse meter-electron volt relationship') / energy
    p_x = args.p_x
    z = args.distance
    pyramid_level = args.pyramid_level
    n_iter = args.n_iter
    use_GPU = args.GPU
    down_sample = args.down_sampling
    use_wavelet = args.use_wavelet
    use_c = args.use_c
    wavelet_level_cut = args.wavelet_lv_cut

    # # roi of the images
    # M_image = int(parameter_wavelet[0])
    # # template window, the N_s nearby pixels used to represent the local pixel, 2*N_s+1
    # N_s = int(parameter_wavelet[1])
    # # the number of the area to calculate for each pixel, 2*cal_half_window X 2*cal_half_window
    # cal_half_window = int(parameter_wavelet[2])
    
    # # process number for parallel
    # n_cores = int(parameter_wavelet[3])
    # # number to reduce the each memory use
    # n_group = int(parameter_wavelet[4])

    # # energy, 10kev
    # energy = float(parameter_wavelet[5])
    
    # p_x = float(parameter_wavelet[6])
    # z = float(parameter_wavelet[7])
    # use_wavelet = int(parameter_wavelet[8])
    # wavelet_level_cut = int(parameter_wavelet[9])
    # # pyramid level to wrap the images
    # pyramid_level = int(parameter_wavelet[10])
    # n_iter = int(parameter_wavelet[11])
    # use_GPU = int(parameter_wavelet[12])
    # down_sample = float(parameter_wavelet[13])

    ref_data = load_image(File_ref)
    img_data = load_image(File_img)

    ref_data = ref_data.astype(np.float32)
    img_data = img_data.astype(np.float32)
    # do image alignment
    if True:
        pos_shift, ref_data = image_align(img_data, ref_data)
        max_shift = int(np.amax(np.abs(pos_shift)) + 1)
        crop_area = lambda img: img[max_shift:-max_shift, max_shift:-max_shift]
        img_data = crop_area(img_data)
        ref_data = crop_area(ref_data)

    # take out the roi
    # modify the cropping part 
    # if M_image > 0:
    #     # boundary crop, use the corner index [y0, y1, x0, x1]
    #     # boundary_crop = lambda img: img[int(args.crop[0]):int(args.crop[1]),
    #     #                                 int(args.crop[2]):int(args.crop[3])]
    #     ref_data = image_roi(ref_data, M_image)
    #     img_data = image_roi(img_data, M_image)

    if len(args.crop) == 4:
        # boundary crop, use the corner index [y0, y1, x0, x1]
        # boundary_crop = lambda img: img[int(args.crop[0]):int(args.crop[1]),
        #                                 int(args.crop[2]):int(args.crop[3])]
        pass
    elif len(args.crop) == 1:
        if args.crop[0] == 0:
            # use gui crop
            print("before crop------------------------------------------------")
            _, corner = crop_gui(img_data/ref_data)
            print("after crop------------------------------------------------")
            
            args.crop = [
                int(corner[0][0]),
                int(corner[1][0]),
                int(corner[0][1]),
                int(corner[1][1])
            ]
        elif args.crop[0] == -1:
            prColor('auto crop------------------------------------------------', 'green')
            # use auto-crop according to the intensity boundary. rectangular shapess
            pattern_size = 5e-6 # assume 5um mask pattern
            flat = snd.uniform_filter(ref_data, size=10*(pattern_size/p_x))
            # plt.figure()
            # plt.imshow(flat)
            # plt.colorbar()
            # plt.show()
            args.crop = auto_crop(flat, shrink=0.85)
        else:
            # central crop
            crop_size = args.crop[0]
            h, w = ref_data.shape
            cy, cx = h // 2, w // 2
            half_size = crop_size // 2
            
            y0 = int(max(0, cy - half_size))
            y1 = int(min(h, cy + half_size))
            x0 = int(max(0, cx - half_size))
            x1 = int(min(w, cx + half_size))
            
            args.crop = [y0, y1, x0, x1]
            prColor('Central crop: {}'.format(args.crop), 'green')
    else:
        # error input
        prColor(
            'error: wrong crop option. 0 for gui crop; [256] for central crop; [y0, y1, x0, x1] for bournday crop',
            'red')
        sys.exit()
    
    boundary_crop = lambda img: img[int(args.crop[0]):int(args.crop[1]),
                                        int(args.crop[2]):int(args.crop[3])]
    ref_data = boundary_crop(ref_data)
    img_data = boundary_crop(img_data)

    M_image = ref_data.shape[0]
    # ref_data = image_roi(ref_data, M_image)
    # img_data = image_roi(img_data, M_image)

    size_origin = ref_data.shape

    ref_data = ref_data.astype(np.float32)
    img_data = img_data.astype(np.float32)

    # down-sample or not
    if down_sample != 1:
        prColor('down-sample image: {}'.format(down_sample),
                'cyan')
        d_size = (int(ref_data.shape[1]*down_sample), int(ref_data.shape[0]*down_sample))

        # from func import binning2
        # img = binning2(img)
        # ref = binning2(ref)
        # print(img.shape)

        img_data = cv2.resize(img_data, d_size)
        ref_data = cv2.resize(ref_data, d_size)

    WXST_solver = WXST(img_data,
                       ref_data,
                       M_image=M_image,
                       N_s=N_s,
                       cal_half_window=cal_half_window,
                       N_s_extend=N_s_extend,
                       n_cores=n_cores,
                       n_group=n_group,
                       energy=energy,
                       p_x=p_x,
                       z=z,
                       wavelet_level_cut=wavelet_level_cut,
                       pyramid_level=pyramid_level,
                       n_iter=n_iter,
                       use_wavelet=use_wavelet,
                       use_GPU=use_GPU,
                       use_c=use_c,
                       use_ncc=args.use_ncc,
                       subpixel_method=args.subpixel_method,
                       use_fft_corr=args.fft_corr,
                       memory_efficient=args.memory_efficient,
                       wavelet_chunk_size=args.wavelet_chunk_size,
                       wavelet_backend=args.wavelet_backend)

    if not os.path.exists(Folder_result):
        os.makedirs(Folder_result)
    # sample_transmission = img_data / (np.abs(ref_data) + 1)
    # plt.imsave(os.path.join(Folder_result, 'transmission.png'),
    #            sample_transmission)

    WXST_solver.run(result_path=Folder_result)
    
    displace = WXST_solver.displace
    DPC_x = WXST_solver.DPC[1]
    DPC_y = WXST_solver.DPC[0]
    phase = WXST_solver.phase
    transmission = WXST_solver.transmission
    result_filename = WXST_solver.result_filename

    # down-sample or not
    if down_sample != 1:
        prColor('scale back', 'green')
        displace_x = cv2.resize(displace[1], (size_origin[1], size_origin[0])) * (1 / down_sample)
        displace_y = cv2.resize(displace[0], (size_origin[1], size_origin[0])) * (1 / down_sample)
        displace = [displace_y, displace_x]
        DPC_x = cv2.resize(DPC_x, (size_origin[1], size_origin[0])) * (1 / down_sample)
        DPC_y = cv2.resize(DPC_y, (size_origin[1], size_origin[0])) * (1 / down_sample)
        phase = cv2.resize(phase, (size_origin[1], size_origin[0])) * (1 / down_sample)
        transmission = cv2.resize(transmission, (size_origin[1], size_origin[0]))

    # save phase directly in the result folder
    img_filename = os.path.basename(File_img).split('.')[0]

    if args.save_plots:
        plt.imsave(os.path.join(Folder_result, 'displace_x.png'), displace[1])
        plt.imsave(os.path.join(Folder_result, 'displace_y.png'), displace[0])
        plt.imsave(os.path.join(Folder_result, 'dpc_x.png'), DPC_x)
        plt.imsave(os.path.join(Folder_result, 'dpc_y.png'), DPC_y)
        plt.imsave(os.path.join(Folder_result, 'phase.png'), phase)
        plt.imsave(os.path.join(Folder_result, 'transmission.png'), transmission)

        plt.figure()
        plt.imshow(displace[0])
        cbar = plt.colorbar()
        cbar.set_label('[pixels]', rotation=90)
        plt.savefig(os.path.join(Folder_result, 'displace_y_colorbar.png'))
        plt.figure()
        plt.imshow(displace[1])
        cbar = plt.colorbar()
        cbar.set_label('[pixels]', rotation=90)
        plt.savefig(os.path.join(Folder_result, 'displace_x_colorbar.png'))
        plt.figure()
        plt.imshow(phase)
        cbar = plt.colorbar()
        cbar.set_label('[rad]', rotation=90)
        plt.savefig(os.path.join(Folder_result, 'phase_colorbar.png'))
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        XX, YY = np.meshgrid(
            np.arange(phase.shape[1]) * p_x * 1e6,
            np.arange(phase.shape[0]) * p_x * 1e6)
        ax1.plot_surface(XX, YY, phase, cmap=cm.get_cmap('hot'))
        ax1.set_xlabel('x [$\mu m$]')
        ax1.set_ylabel('y [$\mu m$]')
        ax1.set_zlabel('phase [rad]')
        plt.savefig(os.path.join(Folder_result, 'Phase_3d.png'))

    # plt.close()

    # d_speckle_sample = 0e-3
    # R_dia = '300e-6 400e-6'
    # hdf5_file = os.path.join(Folder_result, result_filename + '.hdf5')
    # json_file = os.path.join(Folder_result, result_filename + '.json')

    # os.system("python data_PostProcess.py " + hdf5_file + ' ' + json_file +
    #           " {}".format(d_speckle_sample) + " {}".format(R_dia))


if __name__ == "__main__":
    main()
