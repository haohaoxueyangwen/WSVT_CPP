# #########################################################################
# Copyright (c) 2020, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2020. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

import math
import numpy as np
import sys
import pywt
import os
import json
from scipy.linalg import lstsq

class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
import h5py
import glob
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
import datetime
import weakref
import math
from colorama import init, Fore, Style

init(autoreset=True)

def image_roi(img, M):
    '''
        take out the interested area of the all data.
        input:
            img:            image data, 2D or 3D array
            M:              the interested array size
                            if M = 0, use the whole size of the data
        output:
            img_data:       the area of the data
    '''
    img_size = img.shape
    if M == 0:
        return img
    elif len(img_size) == 2:
        if M > min(img_size):
            return img
        else:
            pos_0 = np.arange(M) - np.round(M / 2) + np.round(img_size[0] / 2)
            pos_0 = pos_0.astype('int')
            pos_1 = np.arange(M) - np.round(M / 2) + np.round(img_size[1] / 2)
            pos_1 = pos_1.astype('int')
            img_data = img[pos_0[0]:pos_0[-1] + 1, pos_1[0]:pos_1[-1] + 1]
    elif len(img_size) == 3:
        if M > min(img_size[1:]):
            return img
        else:
            pos_0 = np.arange(M) - np.round(M / 2) + np.round(img_size[1] / 2)
            pos_0 = pos_0.astype('int')
            pos_1 = np.arange(M) - np.round(M / 2) + np.round(img_size[2] / 2)
            pos_1 = pos_1.astype('int')
            img_data = np.zeros((img_size[0], M, M))
            for kk, pp in enumerate(img):
                img_data[kk] = pp[pos_0[0]:pos_0[-1] + 1,
                                  pos_1[0]:pos_1[-1] + 1]

    return img_data


def prColor(word, color_type):
    ''' function to print color text in terminal
        input:
            word:           word to print
            color_type:     which color
                            'red', 'green', 'yellow'
                            'light_purple', 'purple'
                            'cyan', 'light_gray'
                            'black'
    '''
    color_map = {
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'light_purple': Fore.BLUE,  # Using blue for light_purple
        'purple': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'light_gray': Fore.LIGHTBLACK_EX,
        'black': Fore.BLACK,
        'magenta': Fore.MAGENTA
    }

    color = color_map.get(color_type.lower())
    if color:
        print(color + str(word))
    else:
        # print(f"Invalid color type: {color_type}")
        print(word)


def frankotchellappa(dpc_x, dpc_y):
    '''
        Frankt-Chellappa Algrotihm
        input:
            dpc_x:              the differential phase along x
            dpc_y:              the differential phase along y       
        output:
            phi:                phase calculated from the dpc
    '''
    fft2 = lambda x: np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
    ifft2 = lambda x: np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x)))
    fftshift = lambda x: np.fft.fftshift(x)
    # ifftshift = lambda x: np.fft.ifftshift(x)

    NN, MM = dpc_x.shape

    wx, wy = np.meshgrid(np.fft.fftfreq(MM) * 2 * np.pi,
                         np.fft.fftfreq(NN) * 2 * np.pi,
                         indexing='xy')
    wx = fftshift(wx)
    wy = fftshift(wy)
    numerator = -1j * wx * fft2(dpc_x) - 1j * wy * fft2(dpc_y)
    # here use the numpy.fmax method to eliminate the zero point of the division
    denominator = np.fmax((wx)**2 + (wy)**2, np.finfo(float).eps)

    div = numerator / denominator

    phi = np.real(ifft2(div))

    phi -= np.mean(np.real(phi))

    return phi


# calculate the displacement from the correlation map
def find_disp(Corr_img, XX_axis, YY_axis, sub_resolution=True):
    '''
        Adapted from gaussiansubpixel.cpp by Zachary Taylor:
        https://github.com/OpenPIV/openpiv-c--qt/blob/master/src/gaussiansubpixel.cpp
        and a also a code by Ruxandra Cojocaru:
        https://gitlab.esrf.fr/cojocaru/swarp/-/blob/master/swarp/norm_xcorr.py

    '''

    # find the maximal value and postion
    Corr_max = np.amax(Corr_img)
    pos = np.unravel_index(np.argmax(Corr_img, axis=None), Corr_img.shape)

    # Compute displacement on both axes
    Corr_img_pad = np.pad(Corr_img, ((1, 1), (1, 1)), 'edge')
    max_pos_y = pos[0] + 1
    max_pos_x = pos[1] + 1

    dy = (Corr_img_pad[max_pos_y + 1, max_pos_x] -
          Corr_img_pad[max_pos_y - 1, max_pos_x]) / 2.0
    dyy = (Corr_img_pad[max_pos_y + 1, max_pos_x] +
           Corr_img_pad[max_pos_y - 1, max_pos_x] -
           2.0 * Corr_img_pad[max_pos_y, max_pos_x])

    dx = (Corr_img_pad[max_pos_y, max_pos_x + 1] -
          Corr_img_pad[max_pos_y, max_pos_x - 1]) / 2.0
    dxx = (Corr_img_pad[max_pos_y, max_pos_x + 1] +
           Corr_img_pad[max_pos_y, max_pos_x - 1] -
           2.0 * Corr_img_pad[max_pos_y, max_pos_x])

    dxy = (Corr_img_pad[max_pos_y + 1, max_pos_x + 1] -
           Corr_img_pad[max_pos_y + 1, max_pos_x - 1] -
           Corr_img_pad[max_pos_y - 1, max_pos_x + 1] +
           Corr_img_pad[max_pos_y - 1, max_pos_x - 1]) / 4.0

    if ((dxx * dyy - dxy * dxy) != 0.0):
        det = 1.0 / (dxx * dyy - dxy * dxy)
    else:
        det = 0.0
    # the XX, YY axis resolution
    pixel_res_x = XX_axis[0, 1] - XX_axis[0, 0]
    pixel_res_y = YY_axis[1, 0] - YY_axis[0, 0]
    Minor_disp_x = (-(dyy * dx - dxy * dy) * det) * pixel_res_x
    Minor_disp_y = (-(dxx * dy - dxy * dx) * det) * pixel_res_y

    if sub_resolution:
        disp_x = Minor_disp_x + XX_axis[pos[0], pos[1]]
        disp_y = Minor_disp_y + YY_axis[pos[0], pos[1]]
    else:
        disp_x = XX_axis[pos[0], pos[1]]
        disp_y = YY_axis[pos[0], pos[1]]

    max_x = XX_axis[0, -1]
    min_x = XX_axis[0, 0]
    max_y = YY_axis[-1, 0]
    min_y = YY_axis[0, 0]

    if disp_x > max_x:
        disp_x = max_x
    elif disp_x < min_x:
        disp_x = min_x

    if disp_y > max_y:
        disp_y = max_y
    elif disp_y < min_y:
        disp_y = min_y

    return disp_y, disp_x


def Wavelet_transform(img, wavelet_method='db2', w_level=1, return_level=1):
    '''
        do the wavelet transfrom for the 3D image data 
    '''
    coeffs = pywt.wavedec(img,
                          wavelet_method,
                          level=w_level,
                          mode='zero',
                          axis=0)

    coeffs_filter = np.concatenate(coeffs[0:return_level], axis=0)
    coeffs_filter = np.moveaxis(coeffs_filter, 0, -1)

    level_name = []
    for kk in range(w_level):
        level_name.append('D{:d}'.format(kk + 1))
    level_name.append('A{:d}'.format(w_level))
    level_name = level_name[-return_level:]

    return coeffs_filter, level_name


def write_h5(result_path, file_name, data_dict):
    ''' this function is used to save the variables in *args to hdf5 file
        args are in format: {'name': data}
    '''

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    with h5py.File(os.path.join(result_path, file_name + '.hdf5'), 'w') as f:
        for key_name in data_dict:
            f.create_dataset(key_name,
                             data=data_dict[key_name],
                             compression="gzip",
                             compression_opts=9)
    prColor('result hdf5 file : {} saved'.format(file_name + '.hdf5'), 'green')

def read_h5(file_path, key_name, print_key=False):
    '''
        read the data with the key_name in the h5 file
    '''

    if not os.path.exists(file_path):
        prColor('Wrong file path', 'red')
        sys.exit()

    with h5py.File(file_path, 'r') as f:
        # List all groups
        if print_key:
            prColor("Keys: {}".format(list(f.keys())), 'green')

        data = f[key_name][:]
    return data

def load_images(Folder_path, filename_format='*.tif'):
    """
    Load multiple images from a folder
    
    Args:
        Folder_path (str): Path to folder containing images
        filename_format (str): File pattern to match (default: '*.tif')
        
    Returns:
        numpy.ndarray: 3D array of images [num_images, height, width]
    """
    f_list = glob.glob(os.path.join(Folder_path, filename_format))
    f_list = sorted(f_list)
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

def load_image(file_path):
    """
    Load a single image from file path and convert to numpy array
    
    Args:
        file_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Image as a numpy array
    """
    if os.path.exists(file_path):
        img = np.array(Image.open(file_path))
    else:
        prColor('Error: wrong data path. No data is loaded.', 'red')
        sys.exit()
    return img


def select_circular_roi_on_image(image, title="Select circular ROI"):
    """
    Interactive circular ROI selection on an image.
    1. First click: defines the center.
    2. Second click: defines the radius.
    Returns center_x, center_y, radius, and a boolean mask.
    """
    prColor("请在图像上点击以选择圆形区域的中心...", 'magenta')
    
    # 确保使用交互式后端
    import matplotlib
    original_backend = matplotlib.get_backend()
    backend_ok = True
    try:
        matplotlib.use('Qt5Agg')  # 或者 'TkAgg'
    except ImportError:
        try:
            matplotlib.use('TkAgg')
        except ImportError:
            prColor("警告: 无法设置交互式后端，可能会出现问题", 'yellow')
            backend_ok = False
    
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='viridis')
    ax.set_title(title + " - Click center, then click for radius")
    plt.show(block=False)

    # 获取第一个点击（中心点）
    points = plt.ginput(1, timeout=-1)
    if not points:
        prColor("Selection cancelled.", 'yellow')
        plt.close(fig)
        # 恢复原始后端
        if backend_ok:
            try:
                matplotlib.use(original_backend)
            except:
                pass
        return None, None, None, None
    
    center_x, center_y = points[0]
    prColor(f"中心点已选择: ({center_x:.1f}, {center_y:.1f})", 'green')
    
    # 在图上标记中心点
    ax.plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=3)
    ax.set_title(title + " - Now click to set radius")
    plt.draw()
    
    # 获取第二个点击（半径点）
    prColor("现在点击以设置半径...", 'magenta')
    points = plt.ginput(1, timeout=-1)
    if not points:
        prColor("Selection cancelled.", 'yellow')
        plt.close(fig)
        # 恢复原始后端
        if backend_ok:
            try:
                matplotlib.use(original_backend)
            except:
                pass
        return None, None, None, None
    
    radius_x, radius_y = points[0]
    radius = np.sqrt((radius_x - center_x)**2 + (radius_y - center_y)**2)
    
    # 绘制选择的圆形ROI
    from matplotlib.patches import Circle
    circle = Circle((center_x, center_y), radius, fill=False, color='red', linewidth=2)
    ax.add_patch(circle)
    ax.set_title("ROI Selected")
    plt.draw()
    plt.pause(1)  # 显示1秒
    plt.close(fig)
    
    # 创建掩码
    H, W = image.shape
    y, x = np.ogrid[:H, :W]
    mask = np.sqrt((x - center_x)**2 + (y - center_y)**2) <= radius
    
    prColor(f"已选择圆形区域: 中心=({center_x:.1f}, {center_y:.1f}), 半径={radius:.1f}像素", 'green')
    
    # 恢复原始后端
    if backend_ok:
        try:
            matplotlib.use(original_backend)
        except:
            pass
    
    return center_x, center_y, radius, mask

def zernike_analysis_phase(phase_data, mask=None, high_order_terms=15, interactive=False, visualize=False, result_dir=None, run_name=None, low_order_terms_to_remove=6, pixel_size_um=None, wavelength_um=None):
    """
    对相位数据进行两步Zernike多项式拟合分析。
    第一步：可选地剥离低阶像差（如倾斜、离焦）作为背景。
    第二步：对剩余相位进行高阶像差拟合。
    
    Args:
        phase_data (numpy.ndarray): 2D相位数据。
        mask (numpy.ndarray, optional): 布尔掩码数组，True表示要拟合的区域。
        high_order_terms (int): 第二步高阶拟合的Zernike项数，默认为15项。
        interactive (bool): 是否交互式选择ROI。
        visualize (bool): 是否可视化结果。
        result_dir (str, optional): 结果保存目录。
        run_name (str, optional): 分析运行名称。
        low_order_terms_to_remove (int): 第一步要剥离的低阶Zernike项数，默认为6 (到2阶)。设置为0则跳过剥离。
        
    Returns:
        tuple: (phase_final_res, phase_high_order_fit, phase_low_order_bg, fit_info)
               - phase_final_res: 最终残差相位
               - phase_high_order_fit: 拟合出的高阶像差
               - phase_low_order_bg: 剥离的低阶背景
               - fit_info: 包含两次拟合详细信息的字典
    """
    if not isinstance(phase_data, np.ndarray) or phase_data.ndim != 2:
        prColor("错误: 相位数据必须是2D numpy数组", 'red')
        return None, None, None, None
    
    # 处理无效值
    if np.any(np.isnan(phase_data)) or np.any(np.isinf(phase_data)):
        prColor("警告: 相位数据包含NaN或无限值，将替换为0", 'yellow')
        phase_data = np.nan_to_num(phase_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 获取图像尺寸
    H, W = phase_data.shape
    
    # 如果需要交互式选择ROI且没有提供掩码
    if interactive and mask is None:
        center_x, center_y, radius_px, mask = select_circular_roi_on_image(phase_data, "Select circular ROI for fitting")
        if mask is None:
            prColor("未选择区域，拟合取消", 'yellow')
            return None, None, None, None
    elif mask is not None:
        # 如果提供了掩码，尝试从中计算半径
        coords = np.argwhere(mask)
        if coords.size > 0:
            center_y, center_x = np.mean(coords, axis=0)
            radius_px = np.max(np.sqrt((coords[:,0] - center_y)**2 + (coords[:,1] - center_x)**2))
        else:
            radius_px = min(H, W) / 2 # 备用值
    else:
        # 如果没有提供掩码且非交互，则使用图像的最小尺寸的一半作为半径
        radius_px = min(H, W) / 2
        prColor(f"未提供掩码，自动使用半径 {radius_px:.1f} 像素进行拟合", 'yellow')
        y, x = np.ogrid[:H, :W]
        center_y, center_x = H/2, W/2
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius_px**2
    
    # 创建归一化坐标
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    xx, yy = np.meshgrid(x, y)
    
    # 计算极坐标
    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    
    # 获取有效区域的数据
    r_valid = r[mask]
    theta_valid = theta[mask]
    phi_valid = phase_data[mask]
    
    # --- 第一步：低阶背景剥离 (如果需要) ---
    phase_to_fit = phase_data.copy()
    phi_low_order_bg = np.zeros_like(phase_data)
    low_order_fit_info = None

    if low_order_terms_to_remove > 0:
        prColor(f"--- 第一步：剥离 {low_order_terms_to_remove} 项低阶背景 ---", 'magenta')
        
        # 构建低阶Zernike基函数矩阵
        A_low = np.zeros((r_valid.size, low_order_terms_to_remove))
        for j in range(1, low_order_terms_to_remove + 1):
            n, m = noll_to_nm(j)
            A_low[:, j - 1] = zernike(r_valid, theta_valid, n, m)
        
        # 最小二乘拟合低阶系数
        c_low, _, _, _ = lstsq(A_low, phi_valid)
        
        # 重建低阶背景
        for j in range(1, low_order_terms_to_remove + 1):
            n, m = noll_to_nm(j)
            phi_low_order_bg += c_low[j - 1] * zernike(r, theta, n, m)
        
        # 计算“干净”的相位用于下一步拟合
        phase_clean = np.zeros_like(phase_data)
        phase_clean[mask] = phase_data[mask] - phi_low_order_bg[mask]
        phase_to_fit = phase_clean
        
        # 存储低阶拟合信息
        rms_low = np.sqrt(np.mean((phase_data[mask] - phi_low_order_bg[mask])**2))
        pv_low = np.max(phase_data[mask] - phi_low_order_bg[mask]) - np.min(phase_data[mask] - phi_low_order_bg[mask])
        low_order_fit_info = {
            "coefficients": c_low.tolist(),
            "num_terms": low_order_terms_to_remove,
            "rms_after_removal": float(rms_low),
            "pv_after_removal": float(pv_low),
            "fitted_background": phi_low_order_bg
        }
        prColor(f"低阶背景已剥离。残差 RMS: {rms_low:.4f}", 'green')
    else:
        prColor("--- 跳过低阶背景剥离 ---", 'yellow')
        phase_clean = phase_data # 如果不剥离，就直接用原始数据

    # --- 第二步：高阶精细拟合 ---
    prColor(f"--- 第二步：对净化后的相位进行 {high_order_terms} 项高阶拟合 ---", 'magenta')
    phi_valid_for_high_fit = phase_to_fit[mask]

    # 构建高阶Zernike基函数矩阵
    A_high = np.zeros((r_valid.size, high_order_terms))
    for j in range(1, high_order_terms + 1):
        n, m = noll_to_nm(j)
        A_high[:, j - 1] = zernike(r_valid, theta_valid, n, m)

    # 最小二乘拟合高阶系数
    c_high, _, _, _ = lstsq(A_high, phi_valid_for_high_fit)

    # 重建高阶拟合背景
    phi_high_order_fit = np.zeros_like(phase_data)
    for j in range(1, high_order_terms + 1):
        n, m = noll_to_nm(j)
        phi_high_order_fit += c_high[j - 1] * zernike(r, theta, n, m)

    # 计算最终残差
    phi_final_res = np.zeros_like(phase_data)
    phi_final_res[mask] = phase_to_fit[mask] - phi_high_order_fit[mask]
    
    # 计算最终误差指标
    rms_final = np.sqrt(np.mean(phi_final_res[mask]**2))
    pv_final = np.max(phi_final_res[mask]) - np.min(phi_final_res[mask])

    # --- 第三步：计算物理参数（如曲率半径） ---
    high_order_fit_results = {
        "coefficients": c_high.tolist(),
        "num_terms": high_order_terms,
        "rms_error": float(rms_final),
        "pv_error": float(pv_final),
        "zernike_terms": [],
        "fitted_aberration": phi_high_order_fit
    }

    # --- 新增：基于二阶相位和相位梯度的波前曲率计算 ---
    prColor("--- 计算波前相位曲率（基于二阶相位和梯度） ---", 'cyan')
    
    try:
        # 使用第一次拟合后的相位数据进行曲率计算
        phase_for_curvature = phase_to_fit if low_order_terms_to_remove > 0 else phase_data
        
        # 计算相位的一阶梯度（使用numpy的gradient函数）
        if pixel_size_um is not None:
            pixel_size_m = pixel_size_um * 1e-6
            # 计算x和y方向的梯度
            grad_y, grad_x = np.gradient(phase_for_curvature, pixel_size_m, pixel_size_m)
            
            # 计算二阶导数（曲率相关）
            # 计算Hessian矩阵的各个分量
            d2phi_dx2 = np.gradient(grad_x, pixel_size_m, axis=1)  # ∂²φ/∂x²
            d2phi_dy2 = np.gradient(grad_y, pixel_size_m, axis=0)  # ∂²φ/∂y²
            d2phi_dxdy = np.gradient(grad_x, pixel_size_m, axis=0)  # ∂²φ/∂x∂y
            
            # 计算主曲率（高斯曲率和平均曲率）
            # 高斯曲率 K = (∂²φ/∂x²)(∂²φ/∂y²) - (∂²φ/∂x∂y)²
            gaussian_curvature = d2phi_dx2 * d2phi_dy2 - d2phi_dxdy**2
            
            # 平均曲率 H = (∂²φ/∂x² + ∂²φ/∂y²) / 2
            mean_curvature = (d2phi_dx2 + d2phi_dy2) / 2
            
            # 在掩码区域内计算统计量
            if mask is not None:
                gaussian_curvature_roi = gaussian_curvature[mask]
                mean_curvature_roi = mean_curvature[mask]
                grad_magnitude_roi = np.sqrt(grad_x[mask]**2 + grad_y[mask]**2)
                
                # 计算曲率统计信息
                curvature_stats = {
                    'gaussian_curvature': {
                        'mean': float(np.mean(gaussian_curvature_roi)),
                        'std': float(np.std(gaussian_curvature_roi)),
                        'max': float(np.max(gaussian_curvature_roi)),
                        'min': float(np.min(gaussian_curvature_roi)),
                        'rms': float(np.sqrt(np.mean(gaussian_curvature_roi**2)))
                    },
                    'mean_curvature': {
                        'mean': float(np.mean(mean_curvature_roi)),
                        'std': float(np.std(mean_curvature_roi)),
                        'max': float(np.max(mean_curvature_roi)),
                        'min': float(np.min(mean_curvature_roi)),
                        'rms': float(np.sqrt(np.mean(mean_curvature_roi**2)))
                    },
                    'gradient_magnitude': {
                        'mean': float(np.mean(grad_magnitude_roi)),
                        'std': float(np.std(grad_magnitude_roi)),
                        'max': float(np.max(grad_magnitude_roi)),
                        'rms': float(np.sqrt(np.mean(grad_magnitude_roi**2)))
                    }
                }
                
                # 将曲率信息添加到结果中
                high_order_fit_results['wavefront_curvature'] = curvature_stats
                high_order_fit_results['curvature_maps'] = {
                    'gaussian_curvature': gaussian_curvature,
                    'mean_curvature': mean_curvature,
                    'gradient_x': grad_x,
                    'gradient_y': grad_y,
                    'gradient_magnitude': np.sqrt(grad_x**2 + grad_y**2)
                }
                
                prColor(f"波前曲率计算完成:", 'green')
                prColor(f"  高斯曲率 RMS: {curvature_stats['gaussian_curvature']['rms']:.6f} m⁻²", 'green')
                prColor(f"  平均曲率 RMS: {curvature_stats['mean_curvature']['rms']:.6f} m⁻¹", 'green')
                prColor(f"  梯度幅值 RMS: {curvature_stats['gradient_magnitude']['rms']:.6f} rad/m", 'green')
            else:
                prColor("警告: 无有效掩码，跳过曲率统计计算", 'yellow')
        else:
            prColor("警告: 未提供像素尺寸，跳过曲率计算", 'yellow')
            
    except Exception as e:
        prColor(f"计算波前曲率时出错: {e}", 'red')
    
    # --- 原有的基于Zernike系数的曲率半径计算 ---
    if pixel_size_um is not None and wavelength_um is not None and high_order_terms >= 4:
        try:
            # 物理单位转换
            pixel_size_m = pixel_size_um * 1e-6
            wavelength_m = wavelength_um * 1e-6
            rho_max_m = radius_px * pixel_size_m
            
            # 提取离焦系数 c4 (Noll index 4 -> list index 3)
            c4_rad = c_high[3]
            
            # 避免除以零
            if abs(c4_rad) > 1e-9: # 阈值避免数值不稳定
                # 计算曲率半径
                R_m = (np.pi * rho_max_m**2) / (2 * wavelength_m * c4_rad)
                high_order_fit_results['radius_of_curvature_m'] = float(R_m)
                prColor(f"基于Zernike离焦项的曲率半径: {R_m:.3f} m", 'green')
            else:
                prColor("离焦项系数 c4 过小，曲率半径视为无穷大（平面波）", 'yellow')
                high_order_fit_results['radius_of_curvature_m'] = float('inf')

        except Exception as e:
            prColor(f"计算Zernike曲率半径时出错: {e}", 'red')

    # 创建包含所有信息的字典
    fit_info = {
        "analysis_parameters": {
            "low_order_terms_to_remove": low_order_terms_to_remove,
            "high_order_terms": high_order_terms,
            "pixel_size_um": pixel_size_um,
            "wavelength_um": wavelength_um,
            "mask_radius_px": float(radius_px),
            "mask_ratio": float(np.sum(mask) / mask.size)
        },
        "low_order_fit": low_order_fit_info,
        "high_order_fit": high_order_fit_results,
        "original_phase": phase_data, # 保存原始相位
        "purified_phase": phase_clean, # 保存净化后的相位
        "final_residual_phase": phi_final_res # 保存最终残差
    }

    # 添加高阶Zernike项详细信息
    for j in range(1, high_order_terms + 1):
        n, m = noll_to_nm(j)
        term_info = {
            "noll_index": j,
            "n": int(n),
            "m": int(m),
            "coefficient": float(c_high[j-1])
        }
        fit_info["high_order_fit"]["zernike_terms"].append(term_info)

    # 为了向后兼容和简化访问，也把关键结果放在顶层
    fit_info['coefficients'] = c_high.tolist()
    fit_info['rms_error'] = float(rms_final)
    fit_info['pv_error'] = float(pv_final)

    # 可视化
    if visualize:
        _visualize_zernike_analysis(fit_info, result_dir, run_name)

    # 保存结果到JSON文件
    if result_dir and run_name:
        json_path = os.path.join(result_dir, f"{run_name}_fit_info.json")
        try:
            # 创建一个可序列化的版本
            serializable_fit_info = json.loads(json.dumps(fit_info, cls=NumpyArrayEncoder))
            with open(json_path, 'w') as f:
                json.dump(serializable_fit_info, f, indent=4)
            prColor(f"Zernike拟合信息已保存到 {json_path}", 'green')
        except Exception as e:
            prColor(f"无法保存Zernike拟合信息: {e}", 'red')

    return phi_final_res, phi_high_order_fit, phi_low_order_bg, fit_info

def _visualize_zernike_analysis(fit_info, result_dir=None, run_name=None):
    """Helper function to visualize the results of Zernike analysis."""
    import matplotlib
    original_backend = matplotlib.get_backend()
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        try:
            matplotlib.use('TkAgg')
        except ImportError:
            prColor("警告: 无法设置交互式后端进行可视化", 'yellow')

    try:
        # 从fit_info中提取数据
        phase_data = fit_info['original_phase']
        phase_clean = fit_info['purified_phase']
        phi_final_res = fit_info['final_residual_phase']
        low_order_terms_to_remove = fit_info['analysis_parameters']['low_order_terms_to_remove']
        high_order_terms = fit_info['analysis_parameters']['high_order_terms']
        c_high = [term['coefficient'] for term in fit_info['high_order_fit']['zernike_terms']]

        phi_low_order_bg = np.zeros_like(phase_data)
        if fit_info.get('low_order_fit'):
            phi_low_order_bg = fit_info['low_order_fit'].get('fitted_background', np.zeros_like(phase_data))

        phi_high_order_fit = np.zeros_like(phase_data)
        if fit_info.get('high_order_fit'):
            phi_high_order_fit = fit_info['high_order_fit'].get('fitted_aberration', np.zeros_like(phase_data))

        num_plots = 6
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Zernike Analysis Workflow', fontsize=16)
        axs = axs.ravel()

        plot_idx = 0

        # 1. 原始相位图
        im = axs[plot_idx].imshow(phase_data, cmap='jet')
        axs[plot_idx].set_title('1. Original Phase')
        fig.colorbar(im, ax=axs[plot_idx], fraction=0.046, pad=0.04)
        plot_idx += 1

        # 2. 剥离的低阶背景
        if low_order_terms_to_remove > 0:
            im = axs[plot_idx].imshow(phi_low_order_bg, cmap='jet')
            axs[plot_idx].set_title(f'2. Removed Low-Order BG ({low_order_terms_to_remove} terms)')
            fig.colorbar(im, ax=axs[plot_idx], fraction=0.046, pad=0.04)
        else:
            axs[plot_idx].text(0.5, 0.5, 'Low-Order Removal Skipped', ha='center', va='center', fontsize=12)
            axs[plot_idx].set_title('2. Low-Order Background')
            axs[plot_idx].axis('off')
        plot_idx += 1

        # 3. 净化后的相位
        im = axs[plot_idx].imshow(phase_clean, cmap='jet')
        axs[plot_idx].set_title('3. Cleaned Phase for Fitting')
        fig.colorbar(im, ax=axs[plot_idx], fraction=0.046, pad=0.04)
        plot_idx += 1

        # 4. 拟合的高阶像差
        im = axs[plot_idx].imshow(phi_high_order_fit, cmap='jet')
        axs[plot_idx].set_title(f'4. Fitted High-Order Aberration ({high_order_terms} terms)')
        fig.colorbar(im, ax=axs[plot_idx], fraction=0.046, pad=0.04)
        plot_idx += 1

        # 5. 最终残差
        im = axs[plot_idx].imshow(phi_final_res, cmap='jet')
        axs[plot_idx].set_title('5. Final Residual')
        fig.colorbar(im, ax=axs[plot_idx], fraction=0.046, pad=0.04)
        plot_idx += 1

        # 6. Zernike系数条形图
        zernike_indices = np.arange(1, high_order_terms + 1)
        axs[plot_idx].bar(zernike_indices, c_high, align='center', alpha=0.8)
        axs[plot_idx].set_title('6. High-Order Zernike Coefficients')
        axs[plot_idx].set_xlabel('Noll Index')
        axs[plot_idx].set_ylabel('Coefficient Amplitude')
        axs[plot_idx].set_xticks(zernike_indices)
        axs[plot_idx].tick_params(axis='x', rotation=45)

        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if result_dir and run_name:
            save_path = os.path.join(result_dir, f"{run_name}_visualization.png")
            plt.savefig(save_path, dpi=300)
            prColor(f"可视化结果已保存到 {save_path}", 'green')

        plt.show()
        
        # --- 新增：曲率可视化 ---
        if 'curvature_maps' in fit_info.get('high_order_fit', {}):
            try:
                curvature_maps = fit_info['high_order_fit']['curvature_maps']
                curvature_stats = fit_info['high_order_fit']['wavefront_curvature']
                
                # 创建曲率可视化窗口
                fig_curvature, axs_curv = plt.subplots(2, 3, figsize=(18, 12))
                fig_curvature.suptitle('Wavefront Curvature Analysis', fontsize=16)
                axs_curv = axs_curv.ravel()
                
                # 1. 高斯曲率
                im1 = axs_curv[0].imshow(curvature_maps['gaussian_curvature'], cmap='RdBu_r')
                axs_curv[0].set_title(f'Gaussian Curvature\nRMS: {curvature_stats["gaussian_curvature"]["rms"]:.2e} m⁻²')
                fig_curvature.colorbar(im1, ax=axs_curv[0], fraction=0.046, pad=0.04)
                
                # 2. 平均曲率
                im2 = axs_curv[1].imshow(curvature_maps['mean_curvature'], cmap='RdBu_r')
                axs_curv[1].set_title(f'Mean Curvature\nRMS: {curvature_stats["mean_curvature"]["rms"]:.2e} m⁻¹')
                fig_curvature.colorbar(im2, ax=axs_curv[1], fraction=0.046, pad=0.04)
                
                # 3. X方向梯度
                im3 = axs_curv[2].imshow(curvature_maps['gradient_x'], cmap='RdBu_r')
                axs_curv[2].set_title('Phase Gradient (X-direction)')
                fig_curvature.colorbar(im3, ax=axs_curv[2], fraction=0.046, pad=0.04)
                
                # 4. Y方向梯度
                im4 = axs_curv[3].imshow(curvature_maps['gradient_y'], cmap='RdBu_r')
                axs_curv[3].set_title('Phase Gradient (Y-direction)')
                fig_curvature.colorbar(im4, ax=axs_curv[3], fraction=0.046, pad=0.04)
                
                # 5. 梯度幅值
                im5 = axs_curv[4].imshow(curvature_maps['gradient_magnitude'], cmap='viridis')
                axs_curv[4].set_title(f'Gradient Magnitude\nRMS: {curvature_stats["gradient_magnitude"]["rms"]:.2e} rad/m')
                fig_curvature.colorbar(im5, ax=axs_curv[4], fraction=0.046, pad=0.04)
                
                # 6. 曲率统计直方图
                gaussian_curv_roi = curvature_maps['gaussian_curvature'][fit_info['original_phase'] != 0]
                axs_curv[5].hist(gaussian_curv_roi.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
                axs_curv[5].set_title('Gaussian Curvature Distribution')
                axs_curv[5].set_xlabel('Curvature (m⁻²)')
                axs_curv[5].set_ylabel('Frequency')
                axs_curv[5].grid(True, alpha=0.3)
                
                # 移除坐标轴刻度
                for i in range(5):  # 前5个子图
                    axs_curv[i].set_xticks([])
                    axs_curv[i].set_yticks([])
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                
                if result_dir and run_name:
                    curvature_save_path = os.path.join(result_dir, f"{run_name}_curvature_analysis.png")
                    plt.savefig(curvature_save_path, dpi=300)
                    prColor(f"曲率分析结果已保存到 {curvature_save_path}", 'green')
                
                plt.show()
                
            except Exception as e:
                prColor(f"曲率可视化时出错: {e}", 'red')

    except Exception as e:
        prColor(f"可视化Zernike分析时出错: {e}", 'red')
    finally:
        matplotlib.use(original_backend)

def convert_to_grayscale(image, method='weighted'):
    """
    Convert image to grayscale if needed
    
    Args:
        image (numpy.ndarray): Input image
        method (str): Conversion method ('weighted', 'average', 'luminance')
        
    Returns:
        numpy.ndarray: Grayscale image as float64
    """
    if image.ndim == 2:
        # Already grayscale
        return image.astype(np.float64)
    elif image.ndim == 3:
        if image.shape[2] == 1:
            # Single channel
            return image[:, :, 0].astype(np.float64)
        elif image.shape[2] == 3:
            # RGB
            if method == 'weighted':
                # Standard luminance weights
                gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            elif method == 'average':
                gray = np.mean(image, axis=2)
            elif method == 'luminance':
                # ITU-R BT.709 standard
                gray = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
            else:
                raise ValueError(f"Unknown conversion method: {method}")
            return gray.astype(np.float64)
        elif image.shape[2] == 4:
            # RGBA - ignore alpha channel
            if method == 'weighted':
                gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
            elif method == 'average':
                gray = np.mean(image[:, :, :3], axis=2)
            elif method == 'luminance':
                gray = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
            else:
                raise ValueError(f"Unknown conversion method: {method}")
            return gray.astype(np.float64)
        else:
            raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")

def match_image_sizes(ref_img, sample_img, method='crop_to_smaller'):
    """
    Match the sizes of reference and sample images
    
    Args:
        ref_img (numpy.ndarray): Reference image
        sample_img (numpy.ndarray): Sample image
        method (str): Matching method ('crop_to_smaller', 'crop_to_larger', 'pad_to_larger')
        
    Returns:
        tuple: (matched_ref_img, matched_sample_img)
    """
    ref_shape = ref_img.shape
    sample_shape = sample_img.shape
    
    if ref_shape == sample_shape:
        prColor('Images already have matching sizes', 'green')
        return ref_img, sample_img
    
    prColor(f'Reference image shape: {ref_shape}', 'yellow')
    prColor(f'Sample image shape: {sample_shape}', 'yellow')
    
    if method == 'crop_to_smaller':
        # Crop both images to the smaller dimensions
        target_height = min(ref_shape[0], sample_shape[0])
        target_width = min(ref_shape[1], sample_shape[1])
        
        # Center crop reference image
        ref_center_y, ref_center_x = ref_shape[0] // 2, ref_shape[1] // 2
        ref_start_y = ref_center_y - target_height // 2
        ref_end_y = ref_start_y + target_height
        ref_start_x = ref_center_x - target_width // 2
        ref_end_x = ref_start_x + target_width
        
        # Center crop sample image
        sample_center_y, sample_center_x = sample_shape[0] // 2, sample_shape[1] // 2
        sample_start_y = sample_center_y - target_height // 2
        sample_end_y = sample_start_y + target_height
        sample_start_x = sample_center_x - target_width // 2
        sample_end_x = sample_start_x + target_width
        
        matched_ref = ref_img[ref_start_y:ref_end_y, ref_start_x:ref_end_x]
        matched_sample = sample_img[sample_start_y:sample_end_y, sample_start_x:sample_end_x]
        
        prColor(f'Images cropped to size: {target_height}x{target_width}', 'green')
        
    elif method == 'crop_to_larger':
        # Crop to the larger dimensions (pad if needed)
        target_height = max(ref_shape[0], sample_shape[0])
        target_width = max(ref_shape[1], sample_shape[1])
        
        # This would require padding smaller images, which might introduce artifacts
        # For now, fall back to crop_to_smaller
        prColor('crop_to_larger not fully implemented, using crop_to_smaller', 'yellow')
        return match_image_sizes(ref_img, sample_img, method='crop_to_smaller')
        
    elif method == 'pad_to_larger':
        # Pad smaller images to match larger ones
        target_height = max(ref_shape[0], sample_shape[0])
        target_width = max(ref_shape[1], sample_shape[1])
        
        # Pad reference image if needed
        if ref_shape[0] < target_height or ref_shape[1] < target_width:
            pad_y = (target_height - ref_shape[0]) // 2
            pad_x = (target_width - ref_shape[1]) // 2
            matched_ref = np.pad(ref_img, ((pad_y, target_height - ref_shape[0] - pad_y),
                                          (pad_x, target_width - ref_shape[1] - pad_x)), 
                                mode='reflect')
        else:
            matched_ref = ref_img
            
        # Pad sample image if needed
        if sample_shape[0] < target_height or sample_shape[1] < target_width:
            pad_y = (target_height - sample_shape[0]) // 2
            pad_x = (target_width - sample_shape[1]) // 2
            matched_sample = np.pad(sample_img, ((pad_y, target_height - sample_shape[0] - pad_y),
                                               (pad_x, target_width - sample_shape[1] - pad_x)), 
                                   mode='reflect')
        else:
            matched_sample = sample_img
            
        prColor(f'Images padded to size: {target_height}x{target_width}', 'green')
        
    else:
        raise ValueError(f"Unknown matching method: {method}")
    
    return matched_ref, matched_sample

def apply_unified_roi(ref_img, sample_img, roi_size):
    """
    Apply ROI to both images using the same parameters
    
    Args:
        ref_img (numpy.ndarray): Reference image
        sample_img (numpy.ndarray): Sample image
        roi_size (int): ROI size (0 for no ROI)
        
    Returns:
        tuple: (ref_roi, sample_roi)
    """
    if roi_size == 0:
        prColor('No ROI applied (roi_size = 0)', 'yellow')
        return ref_img, sample_img
    
    # Ensure both images have the same shape
    if ref_img.shape != sample_img.shape:
        raise ValueError(f"Images must have the same shape before ROI application. "
                        f"Reference: {ref_img.shape}, Sample: {sample_img.shape}")
    
    # Apply ROI to both images
    ref_roi = image_roi(ref_img, roi_size)
    sample_roi = image_roi(sample_img, roi_size)
    
    prColor(f'ROI applied: {roi_size}x{roi_size} from center', 'green')
    prColor(f'Reference ROI shape: {ref_roi.shape}', 'cyan')
    prColor(f'Sample ROI shape: {sample_roi.shape}', 'cyan')
    
    return ref_roi, sample_roi

def validate_image_pair(ref_img, sample_img, tolerance=1e-10):
    """
    Validate that reference and sample images are properly matched
    
    Args:
        ref_img (numpy.ndarray): Reference image
        sample_img (numpy.ndarray): Sample image
        tolerance (float): Numerical tolerance for validation
        
    Returns:
        dict: Validation results
    """
    validation_results = {
        'shapes_match': False,
        'data_types_match': False,
        'values_in_range': False,
        'no_invalid_values': False,
        'all_tests_passed': False,
        'ref_shape': ref_img.shape,
        'sample_shape': sample_img.shape,
        'ref_dtype': ref_img.dtype,
        'sample_dtype': sample_img.dtype,
        'ref_value_range': (np.min(ref_img), np.max(ref_img)),
        'sample_value_range': (np.min(sample_img), np.max(sample_img))
    }
    
    # Check if shapes match
    validation_results['shapes_match'] = ref_img.shape == sample_img.shape
    
    # Check if data types are compatible
    validation_results['data_types_match'] = ref_img.dtype == sample_img.dtype
    
    # Check if values are in reasonable range
    ref_finite = np.isfinite(ref_img).all()
    sample_finite = np.isfinite(sample_img).all()
    validation_results['values_in_range'] = ref_finite and sample_finite
    
    # Check for invalid values (NaN, inf)
    ref_valid = not (np.isnan(ref_img).any() or np.isinf(ref_img).any())
    sample_valid = not (np.isnan(sample_img).any() or np.isinf(sample_img).any())
    validation_results['no_invalid_values'] = ref_valid and sample_valid
    
    # Overall validation
    validation_results['all_tests_passed'] = (
        validation_results['shapes_match'] and
        validation_results['data_types_match'] and
        validation_results['values_in_range'] and
        validation_results['no_invalid_values']
    )
    
    # Print validation report
    prColor('=== Image Pair Validation Report ===', 'cyan')
    prColor(f"Shapes match: {validation_results['shapes_match']}", 
           'green' if validation_results['shapes_match'] else 'red')
    prColor(f"Data types match: {validation_results['data_types_match']}", 
           'green' if validation_results['data_types_match'] else 'red')
    prColor(f"Values in range: {validation_results['values_in_range']}", 
           'green' if validation_results['values_in_range'] else 'red')
    prColor(f"No invalid values: {validation_results['no_invalid_values']}", 
           'green' if validation_results['no_invalid_values'] else 'red')
    
    if validation_results['all_tests_passed']:
        prColor('✓ All validation tests PASSED', 'green')
    else:
        prColor('✗ Some validation tests FAILED', 'red')
    
    prColor(f"Reference - Shape: {validation_results['ref_shape']}, "
           f"Type: {validation_results['ref_dtype']}, "
           f"Range: [{validation_results['ref_value_range'][0]:.3f}, "
           f"{validation_results['ref_value_range'][1]:.3f}]", 'light_gray')
    prColor(f"Sample - Shape: {validation_results['sample_shape']}, "
           f"Type: {validation_results['sample_dtype']}, "
           f"Range: [{validation_results['sample_value_range'][0]:.3f}, "
           f"{validation_results['sample_value_range'][1]:.3f}]", 'light_gray')
    
    return validation_results

def preprocess_image_pair(ref_path, sample_path, roi_size=0, 
                         grayscale_method='weighted', size_match_method='crop_to_smaller',
                         normalize=False, target_dtype=np.float64):
    """
    Complete preprocessing pipeline for reference and sample image pair
    
    Args:
        ref_path (str): Path to reference image
        sample_path (str): Path to sample image
        roi_size (int): ROI size (0 for no ROI)
        grayscale_method (str): Method for RGB to grayscale conversion
        size_match_method (str): Method for size matching
        normalize (bool): Whether to normalize images to [0, 1]
        target_dtype: Target data type for final images
        
    Returns:
        tuple: (processed_ref_img, processed_sample_img, processing_report)
    """
    processing_report = {
        'steps_completed': [],
        'warnings': [],
        'errors': [],
        'final_shape': None,
        'final_dtype': None
    }
    
    try:
        # Step 1: Load images
        prColor('Step 1: Loading images...', 'purple')
        ref_img = load_image(ref_path)
        sample_img = load_image(sample_path)
        processing_report['steps_completed'].append('images_loaded')
        prColor(f'Reference loaded: {ref_img.shape}, dtype: {ref_img.dtype}', 'light_gray')
        prColor(f'Sample loaded: {sample_img.shape}, dtype: {sample_img.dtype}', 'light_gray')
        
        # Step 2: Convert to grayscale if needed
        prColor('Step 2: Converting to grayscale...', 'purple')
        ref_gray = convert_to_grayscale(ref_img, method=grayscale_method)
        sample_gray = convert_to_grayscale(sample_img, method=grayscale_method)
        processing_report['steps_completed'].append('converted_to_grayscale')
        prColor(f'Grayscale conversion completed using {grayscale_method} method', 'green')
        
        # Step 3: Match image sizes
        prColor('Step 3: Matching image sizes...', 'purple')
        ref_matched, sample_matched = match_image_sizes(ref_gray, sample_gray, method=size_match_method)
        processing_report['steps_completed'].append('sizes_matched')
        
        # Step 4: Apply ROI
        prColor('Step 4: Applying ROI...', 'purple')
        ref_roi, sample_roi = apply_unified_roi(ref_matched, sample_matched, roi_size)
        processing_report['steps_completed'].append('roi_applied')
        
        # Step 5: Data type conversion
        prColor('Step 5: Converting data types...', 'purple')
        ref_final = ref_roi.astype(target_dtype)
        sample_final = sample_roi.astype(target_dtype)
        processing_report['steps_completed'].append('dtype_converted')
        
        # Step 6: Normalization (optional)
        if normalize:
            prColor('Step 6: Normalizing images...', 'purple')
            ref_final = (ref_final - np.min(ref_final)) / (np.max(ref_final) - np.min(ref_final))
            sample_final = (sample_final - np.min(sample_final)) / (np.max(sample_final) - np.min(sample_final))
            processing_report['steps_completed'].append('normalized')
            prColor('Images normalized to [0, 1] range', 'green')
        
        # Step 7: Final validation
        prColor('Step 7: Final validation...', 'purple')
        validation_results = validate_image_pair(ref_final, sample_final)
        processing_report['steps_completed'].append('validated')
        
        # Update final report
        processing_report['final_shape'] = ref_final.shape
        processing_report['final_dtype'] = ref_final.dtype
        processing_report['validation_passed'] = validation_results['all_tests_passed']
        
        if validation_results['all_tests_passed']:
            prColor('✓ Image preprocessing completed successfully!', 'green')
        else:
            processing_report['errors'].append('Final validation failed')
            prColor('✗ Image preprocessing completed with validation errors', 'red')
        
        return ref_final, sample_final, processing_report
        
    except Exception as e:
        error_msg = f"Error during preprocessing: {str(e)}"
        processing_report['errors'].append(error_msg)
        prColor(error_msg, 'red')
        raise e

def load_and_preprocess_image(file_path, grayscale_method='weighted', target_dtype=np.float64):
    """
    Load and preprocess a single image
    
    Args:
        file_path (str): Path to image file
        grayscale_method (str): Method for RGB to grayscale conversion
        target_dtype: Target data type
        
    Returns:
        numpy.ndarray: Preprocessed image
    """
    img = load_image(file_path)
    img_gray = convert_to_grayscale(img, method=grayscale_method)
    img_final = img_gray.astype(target_dtype)
    
    prColor(f'Image loaded and preprocessed: {img_final.shape}, dtype: {img_final.dtype}', 'green')
    return img_final

def plot_3d_surface_from_image(data, title="3D Surface", xlabel="X [pixels]", ylabel="Y [pixels]", 
                              zlabel="Value", height_scale=1.0, cmap='viridis', elev=30, azim=30, 
                              rstride=1, cstride=1, save_path_prefix=None):
    """
    Create 3D surface plot from 2D data array
    
    Args:
        data (numpy.ndarray): 2D array of data to plot
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
        zlabel (str): Z-axis label
        height_scale (float): Scale factor for Z values
        cmap (str): Colormap name
        elev (int): Elevation angle for 3D view
        azim (int): Azimuth angle for 3D view
        rstride (int): Row stride for surface sampling
        cstride (int): Column stride for surface sampling
        save_path_prefix (str): If provided, save plot to this path prefix
        
    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    if data is None or not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("Data must be a 2D numpy array")
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create coordinate grids
    rows, cols = data.shape
    X = np.arange(cols)
    Y = np.arange(rows)
    X, Y = np.meshgrid(X, Y)
    
    # Scale the Z data
    Z = data * height_scale
    
    # Create 3D surface plot
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, 
                          rstride=rstride, cstride=cstride,
                          linewidth=0, antialiased=True, alpha=0.8)
    
    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    
    # Set viewing angle
    ax.view_init(elev=elev, azim=azim)
    
    # Add colorbar
    colorbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    colorbar.set_label(zlabel)
    
    # Improve layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path_prefix:
        try:
            save_path = f"{save_path_prefix}_3D_surface.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            prColor(f'3D surface plot saved: {save_path}', 'green')
        except Exception as e:
            prColor(f'Error saving 3D plot: {e}', 'red')
    
    return fig


def write_json(result_path, file_name, data_dict):

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    file_name_para = os.path.join(result_path, file_name + '.json')
    with open(file_name_para, 'w') as fp:
        json.dump(data_dict, fp, indent=0)

    prColor('result json file : {} saved'.format(file_name + '.json'), 'green')


def read_json(filepath, print_para=False):

    if not os.path.exists(filepath):
        prColor('Wrong file path', 'red')
        sys.exit()
    # file_name_para = os.path.join(result_path, file_name+'.json')
    with open(filepath, 'r') as fp:
        data = json.load(fp)
        if print_para:
            prColor('parameters: {}'.format(data), 'green')

    return data

# --- Visualization and Auto-Save Functions ---

def create_standalone_figure(figsize=(10, 8), dpi=150):
    """
    Create a standalone matplotlib figure for saving
    
    Args:
        figsize (tuple): Figure size in inches
        dpi (int): DPI for the figure
        
    Returns:
        matplotlib.figure.Figure: Created figure object
    """
    return plt.figure(figsize=figsize, dpi=dpi)

def save_2d_figure_with_colorbar(data, result_type, output_dir, run_name, 
                                 figsize=(10, 8), dpi=150, cmap='gray'):
    """
    Save 2D figure with proper colorbar and annotations (saved as grayscale)
    
    Args:
        data (numpy.ndarray): 2D data array to visualize
        result_type (str): Type of result (e.g., 'phase', 'transmission')
        output_dir (str): Directory to save the figure
        run_name (str): Name of the analysis run
        figsize (tuple): Figure size in inches
        dpi (int): DPI for the saved figure
        cmap (str): Colormap to use (default: 'gray' for grayscale output)
        
    Returns:
        str: Path to saved file, or None if failed
    """
    try:
        # Create a new figure for 2D rendering
        fig = create_standalone_figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        
        # Create 2D image plot with good default settings
        im = ax.imshow(data, cmap=cmap, interpolation='nearest', 
                      aspect='auto', origin='upper')
        
        # Set comprehensive labels and title
        ax.set_xlabel('X [pixels]', fontsize=12)
        ax.set_ylabel('Y [pixels]', fontsize=12)
        
        # Set title with comprehensive information
        title = f"{run_name}: {result_type.replace('_', ' ').title()}\n"
        title += f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ax.set_title(title, fontsize=14, pad=20)
        
        # Add colorbar with proper labeling
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Set appropriate colorbar label based on result type
        if "phase" in result_type.lower():
            cbar_label = "Phase [rad]"
        elif "displace" in result_type.lower():
            cbar_label = "Displacement [pixels]"
        elif "dpc" in result_type.lower():
            cbar_label = "DPC [rad]"
        elif "transmission" in result_type.lower():
            cbar_label = "Transmission"
        else:
            cbar_label = "Value [arb. units]"
        
        cbar.set_label(cbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        # Adjust layout for better appearance
        fig.tight_layout()
        
        # Generate filename
        safe_result_type = "".join(c if c.isalnum() else "_" for c in result_type)
        safe_run_name = "".join(c if c.isalnum() else "_" for c in run_name)
        filename = f"{safe_run_name}_{safe_result_type}_2d.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save with high quality
        fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # Close the figure to prevent memory leaks
        plt.close(fig)
        
        prColor(f'2D visualization saved: {filename}', 'green')
        return filepath
        
    except Exception as e:
        prColor(f'Error saving 2D figure for {result_type}: {e}', 'red')
        return None

def save_3d_figure_with_colorbar(data, result_type, output_dir, run_name,
                                 figsize=(12, 9), dpi=150, cmap='viridis'):
    """
    Save 3D figure with proper colorbar and annotations
    
    Args:
        data (numpy.ndarray): 2D data array to visualize as 3D surface
        result_type (str): Type of result (e.g., 'phase', 'transmission')
        output_dir (str): Directory to save the figure
        run_name (str): Name of the analysis run
        figsize (tuple): Figure size in inches
        dpi (int): DPI for the saved figure
        cmap (str): Colormap to use
        
    Returns:
        str: Path to saved file, or None if failed
    """
    try:
        # Create a new figure for 3D rendering
        fig = create_standalone_figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mesh grid for 3D surface
        rows, cols = data.shape
        X = np.arange(cols)
        Y = np.arange(rows)
        X, Y = np.meshgrid(X, Y)
        Z = data
        
        # Create surface plot with good default settings
        surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.9, 
                             antialiased=True, linewidth=0, shade=True)
        
        # Set comprehensive labels and title
        ax.set_xlabel('X [pixels]', fontsize=12, labelpad=10)
        ax.set_ylabel('Y [pixels]', fontsize=12, labelpad=10)
        
        # Set appropriate Z-axis label based on result type
        if "phase" in result_type.lower():
            zlabel = "Phase [rad]"
        elif "displace" in result_type.lower():
            zlabel = "Displacement [pixels]"
        elif "dpc" in result_type.lower():
            zlabel = "DPC [rad]"
        elif "transmission" in result_type.lower():
            zlabel = "Transmission"
        else:
            zlabel = "Value"
        ax.set_zlabel(zlabel, fontsize=12, labelpad=10)
        
        # Set title with comprehensive information
        title = f"{run_name}: {result_type.replace('_', ' ').title()}\n"
        title += f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ax.set_title(title, fontsize=14, pad=20)
        
        # Add colorbar with proper labeling
        cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.1)
        cbar.set_label(zlabel, fontsize=12, labelpad=15)
        cbar.ax.tick_params(labelsize=10)
        
        # Set good viewing angle
        ax.view_init(elev=30, azim=45)
        
        # Adjust layout for better appearance
        fig.tight_layout()
        
        # Generate filename
        safe_result_type = "".join(c if c.isalnum() else "_" for c in result_type)
        safe_run_name = "".join(c if c.isalnum() else "_" for c in run_name)
        filename = f"{safe_run_name}_{safe_result_type}_3d.png"
        filepath = os.path.join(output_dir, filename)
        
        # Save with high quality
        fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        # Close the figure to prevent memory leaks
        plt.close(fig)
        
        prColor(f'3D visualization saved: {filename}', 'green')
        return filepath
        
    except Exception as e:
        prColor(f'Error saving 3D figure for {result_type}: {e}', 'red')
        return None

def auto_save_all_results(results_dict, output_dir, run_name):
    """
    Automatically save visualizations: phase gets 2D+3D, others get 2D only
    
    Args:
        results_dict (dict): Dictionary containing analysis results
        output_dir (str): Directory to save visualizations
        run_name (str): Name of the analysis run
        
    Returns:
        list: List of successfully saved file paths
    """
    saved_files = []
    
    # Define which result types should get which visualization
    PHASE_RESULTS = ['phase']  # Gets both 2D and 3D
    ALL_2D_RESULTS = ['phase', 'transmission', 'displace_x', 'displace_y', 'DPC_x', 'DPC_y']  # All get 2D
    
    # Find all valid results to save
    valid_results = []
    for key, data in results_dict.items():
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.size > 0:
            valid_results.append(key)
    
    if not valid_results:
        prColor("No valid results found for visualization", 'yellow')
        return saved_files

    total_files_to_save = len([k for k in valid_results if k in ALL_2D_RESULTS]) + len([k for k in valid_results if k in PHASE_RESULTS])
    prColor(f"Auto-saving {total_files_to_save} visualization files...", 'cyan')
    
    # Save 2D visualizations for all valid results
    for result_type in valid_results:
        if result_type in ALL_2D_RESULTS:
            data = results_dict[result_type]
            filepath_2d = save_2d_figure_with_colorbar(data, result_type, output_dir, run_name)
            if filepath_2d:
                saved_files.append(filepath_2d)
    
    # Save 3D visualizations for phase results only
    for result_type in valid_results:
        if result_type in PHASE_RESULTS:
            data = results_dict[result_type]
            filepath_3d = save_3d_figure_with_colorbar(data, result_type, output_dir, run_name)
            if filepath_3d:
                saved_files.append(filepath_3d)
    
    if saved_files:
        saved_2d_count = len([f for f in saved_files if '_2d.png' in f])
        saved_3d_count = len([f for f in saved_files if '_3d.png' in f])
        
        message_parts = []
        if saved_2d_count > 0:
            message_parts.append(f"{saved_2d_count} 2D")
        if saved_3d_count > 0:
            message_parts.append(f"{saved_3d_count} 3D")
        
        prColor(f"✓ Auto-save completed! Saved {' and '.join(message_parts)} visualization(s)", 'green')
    else:
        prColor("No files were saved during auto-save", 'yellow')
    
    return saved_files

def save_16bit_tiff_image(data, filename, result_type=""):
    """
    Save 2D numpy array as 16-bit TIFF image (data only, no visualization)
    
    Args:
        data (numpy.ndarray): 2D data array to save
        filename (str): Output filename (should end with .tiff or .tif)
        result_type (str): Type of result for metadata (optional)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            prColor("Error: Data must be a 2D numpy array", 'red')
            return False
        
        if data.size == 0:
            prColor("Error: Data array is empty", 'red')
            return False
        
        # Handle invalid values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            prColor("Warning: Data contains NaN or infinite values, replacing with 0", 'yellow')
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize data to 16-bit range (0-65535)
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max == data_min:
            prColor("Warning: Data has uniform values, creating flat 16-bit image", 'yellow')
            data_16bit = np.zeros_like(data, dtype=np.uint16)
        else:
            # Scale to full 16-bit range
            data_normalized = (data - data_min) / (data_max - data_min)
            data_16bit = (data_normalized * 65535).astype(np.uint16)
        
        # Ensure filename has correct extension
        if not filename.lower().endswith(('.tiff', '.tif')):
            filename += '.tiff'
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save as 16-bit TIFF using PIL - avoid problematic tiffinfo
        image = Image.fromarray(data_16bit, mode='I;16')
        
        # Save without metadata to avoid type comparison errors
        image.save(filename, format='TIFF')
        
        prColor(f'16-bit TIFF data image saved: {os.path.basename(filename)}', 'green')
        prColor(f'Original data range: [{data_min:.6f}, {data_max:.6f}]', 'light_gray')
        prColor(f'16-bit range: [0, 65535]', 'light_gray')
        
        return True
        
    except Exception as e:
        prColor(f'Error saving 16-bit TIFF image: {e}', 'red')
        return False

def select_rectangle_roi_on_image(image, window_title="Select Rectangle ROI"):
    """
    Display an image and allow the user to select a rectangular ROI using matplotlib,
    ensuring compatibility with a running PyQt5 event loop.

    Args:
        image (numpy.ndarray): The image to display.
        window_title (str): The title of the selection window.

    Returns:
        tuple: (x, y, w, h) of the selected rectangle, or None if not selected.
    """
    import numpy as _np
    import matplotlib.pyplot as _plt
    import matplotlib
    from matplotlib.widgets import RectangleSelector as _RectangleSelector
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QApplication
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont

    # Ensure a QApplication instance exists, but do not create one.
    if QApplication.instance() is None:
        # This function should be called from a PyQt application.
        # If you are running this standalone, you need to create a QApplication first.
        # For example:
        # if __name__ == '__main__':
        #     app = QApplication([])
        #     # ... call the function ...
        #     app.exec_()
        raise RuntimeError("A QApplication instance must be created before calling this function.")

    # Set non-interactive backend to avoid event loop conflicts
    current_backend = matplotlib.get_backend()
    matplotlib.use('Qt5Agg')

    try:
        img = _np.asarray(image)
        if img.ndim == 3 and img.shape[2] > 1:
            img_disp = img.mean(axis=2)
        else:
            img_disp = img

        img_disp = img_disp.astype(float)
        finite_mask = _np.isfinite(img_disp)
        if finite_mask.any():
            vmin = _np.nanmin(img_disp)
            vmax = _np.nanmax(img_disp)
            if vmax > vmin:
                img_disp = (img_disp - vmin) / (vmax - vmin)
            else:
                img_disp = _np.zeros_like(img_disp)
        else:
            img_disp = _np.zeros_like(img_disp)

        # Create a custom dialog to embed matplotlib
        class ROISelectionDialog(QDialog):
            def __init__(self):
                super().__init__()
                self.setWindowTitle(window_title)
                self.setModal(True)
                self.resize(800, 600)
                
                # Layout
                layout = QVBoxLayout()
                
                # Instructions
                instruction_label = QLabel("Drag mouse to select rectangle area, then click Confirm or Cancel")
                instruction_label.setFont(QFont("Arial", 10))
                instruction_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(instruction_label)
                
                # Matplotlib figure
                self.figure = _plt.figure(figsize=(8, 6))
                self.canvas = FigureCanvas(self.figure)
                layout.addWidget(self.canvas)
                
                # Buttons
                button_layout = QHBoxLayout()
                self.confirm_button = QPushButton("Confirm Selection")
                self.cancel_button = QPushButton("Cancel")
                self.confirm_button.clicked.connect(self.accept)
                self.cancel_button.clicked.connect(self.reject)
                button_layout.addWidget(self.confirm_button)
                button_layout.addWidget(self.cancel_button)
                layout.addLayout(button_layout)
                
                self.setLayout(layout)
                
                # Setup matplotlib plot
                self.ax = self.figure.add_subplot(111)
                self.ax.imshow(img_disp, cmap='gray', origin='upper')
                self.ax.set_title('Select Rectangle ROI - Drag to select area')
                
                # ROI storage
                self.roi = {'x1': None, 'y1': None, 'x2': None, 'y2': None}
                
                # Rectangle selector
                def onselect(eclick, erelease):
                    if eclick.xdata is not None and eclick.ydata is not None and \
                       erelease.xdata is not None and erelease.ydata is not None:
                        x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
                        x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
                        self.roi['x1'], self.roi['y1'] = min(x1, x2), min(y1, y2)
                        self.roi['x2'], self.roi['y2'] = max(x1, x2), max(y1, y2)
                        # Update button state
                        self.confirm_button.setEnabled(True)
                
                try:
                    self.selector = _RectangleSelector(
                        self.ax,
                        onselect,
                        useblit=False,
                        minspanx=2,
                        minspany=2,
                        spancoords='pixels',
                        interactive=True,
                        props=dict(facecolor='none', edgecolor='red', alpha=1.0, linewidth=1.5)
                    )
                except TypeError:
                    # Fallback for older Matplotlib versions
                    self.selector = _RectangleSelector(
                        self.ax,
                        onselect,
                        useblit=False,
                        minspanx=2,
                        minspany=2,
                        button=[1],  # Left mouse button
                        spancoords='pixels',
                        interactive=True,
                        props=dict(facecolor='none', edgecolor='red', alpha=1.0, linewidth=1.5)
                    )
                
                self.confirm_button.setEnabled(False)

            def get_roi(self):
                if self.roi['x1'] is not None:
                    x = self.roi['x1']
                    y = self.roi['y1']
                    w = self.roi['x2'] - self.roi['x1']
                    h = self.roi['y2'] - self.roi['y1']
                    return (x, y, w, h)
                return None

        dialog = ROISelectionDialog()
        result = dialog.exec_()

        if result == QDialog.Accepted:
            return dialog.get_roi()
        else:
            return None

    finally:
        # Restore the original backend
        matplotlib.use(current_backend)

def save_16bit_visualization_improved(data, filename, result_type="", cmap='viridis', 
                                     figsize=(10, 8), dpi=300, include_title=False):
    """
    Improved 16-bit TIFF visualization saving with better error handling and robustness
    
    Args:
        data (numpy.ndarray): 2D data array to visualize
        filename (str): Output filename
        result_type (str): Type of result for labeling
        cmap (str): Colormap to use
        figsize (tuple): Figure size in inches
        dpi (int): DPI for the saved image
        include_title (bool): Whether to include title
        
    Returns:
        bool: True if successful, False otherwise
    """
    fig = None
    try:
        # Enhanced data validation
        if not isinstance(data, np.ndarray):
            prColor("Error: Input must be a numpy array", 'red')
            return False
            
        if data.ndim != 2:
            prColor(f"Error: Data must be 2D, got {data.ndim}D array", 'red')
            return False
        
        if data.size == 0:
            prColor("Error: Data array is empty", 'red')
            return False
        
        # Handle invalid values with better reporting
        nan_count = np.count_nonzero(np.isnan(data))
        inf_count = np.count_nonzero(np.isinf(data))
        
        if nan_count > 0 or inf_count > 0:
            prColor(f"Warning: Data contains {nan_count} NaN and {inf_count} infinite values, replacing with 0", 'yellow')
            data_clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            data_clean = data.copy()
        
        # Ensure filename has correct extension
        if not filename.lower().endswith(('.tiff', '.tif')):
            filename += '.tiff'
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as dir_error:
                prColor(f"Error creating output directory: {dir_error}", 'red')
                return False
        
        prColor(f"Creating 16-bit visualization for {result_type}...", 'cyan')
        
        # Create matplotlib figure with better resource management
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111)
        
        # Create image with colorbar and enhanced error handling
        try:
            im = ax.imshow(data_clean, cmap=cmap, interpolation='nearest', aspect='auto')
        except Exception as imshow_error:
            prColor(f"Error creating image plot: {imshow_error}", 'red')
            # Try with simplified parameters
            im = ax.imshow(data_clean, cmap='gray', interpolation='nearest')
        
        # Set labels with error handling
        try:
            ax.set_xlabel('X [pixels]', fontsize=12)
            ax.set_ylabel('Y [pixels]', fontsize=12)
            
            # Add title if requested
            if include_title and result_type:
                ax.set_title(f"{result_type.replace('_', ' ').title()}", fontsize=14)
            
            # Add colorbar with appropriate label
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Set colorbar label based on result type
            if "phase" in result_type.lower():
                cbar_label = "Phase [rad]"
            elif "displace" in result_type.lower():
                cbar_label = "Displacement [pixels]"
            elif "dpc" in result_type.lower():
                cbar_label = "DPC [rad]"
            elif "transmission" in result_type.lower():
                cbar_label = "Transmission"
            else:
                cbar_label = "Value [arb. units]"
            
            cbar.set_label(cbar_label, fontsize=12)
            cbar.ax.tick_params(labelsize=10)
            
        except Exception as label_error:
            prColor(f"Warning: Error setting labels: {label_error}", 'yellow')
        
        # Adjust layout
        try:
            fig.tight_layout()
        except Exception as layout_error:
            prColor(f"Warning: Layout adjustment failed: {layout_error}", 'yellow')
        
        # Render to array with improved error handling
        try:
            fig.canvas.draw()
            
            # Get canvas dimensions
            width, height = fig.canvas.get_width_height()
            
            # Render to RGB buffer - handle different matplotlib versions
            try:
                # Try newer matplotlib method first
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            except AttributeError:
                # Fall back to older method
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                # Convert RGBA to RGB
                buf = buf.reshape((height, width, 4))[:, :, :3].flatten()
            
            buf = buf.reshape((height, width, 3))
            
            # Convert RGB to grayscale using standard luminance weights
            gray_image = 0.299 * buf[:, :, 0] + 0.587 * buf[:, :, 1] + 0.114 * buf[:, :, 2]
            
            # Ensure data is in correct range and type
            gray_image = np.clip(gray_image, 0, 255)
            
            # Convert to 16-bit (scale 8-bit [0,255] to 16-bit [0,65535])
            gray_16bit = (gray_image * 257).astype(np.uint16)
            
        except Exception as render_error:
            prColor(f"Error during figure rendering: {render_error}", 'red')
            return False
        
        # Save as 16-bit TIFF with improved error handling
        try:
            # Use PIL with explicit mode setting
            image_16bit = Image.fromarray(gray_16bit, mode='I;16')
            
            # Save without problematic metadata
            image_16bit.save(filename, format='TIFF', compression=None)
            
            prColor(f'16-bit TIFF visualization saved: {os.path.basename(filename)}', 'green')
            prColor(f'Dimensions: {gray_16bit.shape}, Size: {os.path.getsize(filename)/1024:.1f} KB', 'light_gray')
            
            return True
            
        except Exception as save_error:
            prColor(f"Error saving TIFF file: {save_error}", 'red')
            
            # Try alternative saving method
            try:
                prColor("Attempting alternative save method...", 'yellow')
                import tifffile
                tifffile.imwrite(filename, gray_16bit, photometric='minisblack')
                prColor(f'16-bit TIFF saved using alternative method: {os.path.basename(filename)}', 'green')
                return True
            except Exception as alt_save_error:
                prColor(f"Alternative save method also failed: {alt_save_error}", 'red')
                return False
        
    except Exception as e:
        prColor(f'Unexpected error in save_16bit_visualization_improved: {e}', 'red')
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Always clean up the figure
        if fig is not None:
            try:
                plt.close(fig)
            except Exception as cleanup_error:
                prColor(f"Warning: Error cleaning up figure: {cleanup_error}", 'yellow')

def save_16bit_visualization_with_colorbar(data, filename, result_type="", cmap='viridis', 
                                          figsize=(10, 8), dpi=300, include_title=False):
    """
    Legacy function - redirects to improved version
    """
    return save_16bit_visualization_improved(data, filename, result_type, cmap, figsize, dpi, include_title)

def save_16bit_tiff_with_custom_range(data, filename, min_val=None, max_val=None, result_type=""):
    """
    Save 2D numpy array as 16-bit TIFF image with custom value range
    
    Args:
        data (numpy.ndarray): 2D data array to save
        filename (str): Output filename
        min_val (float): Custom minimum value for scaling (None for auto)
        max_val (float): Custom maximum value for scaling (None for auto)
        result_type (str): Type of result for metadata
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            prColor("Error: Data must be a 2D numpy array", 'red')
            return False
        
        # Handle invalid values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            prColor("Warning: Data contains NaN or infinite values, replacing with 0", 'yellow')
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Use custom range or auto-detect
        if min_val is None:
            min_val = np.min(data)
        if max_val is None:
            max_val = np.max(data)
        
        # Clip data to specified range
        data_clipped = np.clip(data, min_val, max_val)
        
        # Scale to 16-bit range
        if max_val == min_val:
            data_16bit = np.zeros_like(data_clipped, dtype=np.uint16)
        else:
            data_normalized = (data_clipped - min_val) / (max_val - min_val)
            data_16bit = (data_normalized * 65535).astype(np.uint16)
        
        # Ensure filename has correct extension
        if not filename.lower().endswith(('.tiff', '.tif')):
            filename += '.tiff'
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save as 16-bit TIFF
        image = Image.fromarray(data_16bit, mode='I;16')
        
        # Add metadata
        metadata = {
            'Software': 'Speckle Tracking Analysis Tool',
            'DateTime': datetime.datetime.now().strftime('%Y:%m:%d %H:%M:%S'),
            'ImageDescription': f'16-bit {result_type} data (custom range)' if result_type else '16-bit data (custom range)',
            'Custom_min_value': str(min_val),
            'Custom_max_value': str(max_val),
            'Data_type': result_type if result_type else 'unknown'
        }
        
        image.save(filename, format='TIFF', tiffinfo=metadata)
        
        prColor(f'16-bit TIFF image saved with custom range: {os.path.basename(filename)}', 'green')
        prColor(f'Custom range: [{min_val:.6f}, {max_val:.6f}]', 'light_gray')
        
        return True
        
    except Exception as e:
        prColor(f'Error saving 16-bit TIFF image with custom range: {e}', 'red')
        return False

def save_single_result_visualization(data, result_type, output_dir, run_name, 
                                   save_2d=True, save_3d=False, cmap_2d='gray', cmap_3d='viridis'):
    """
    Save visualization for a single result type
    
    Args:
        data (numpy.ndarray): 2D data array to visualize
        result_type (str): Type of result
        output_dir (str): Directory to save the figure
        run_name (str): Name of the analysis run
        save_2d (bool): Whether to save 2D visualization
        save_3d (bool): Whether to save 3D visualization
        cmap_2d (str): Colormap to use for 2D images (default: 'gray' for grayscale)
        cmap_3d (str): Colormap to use for 3D images (default: 'viridis' for color)
        
    Returns:
        list: List of successfully saved file paths
    """
    saved_files = []
    
    if not isinstance(data, np.ndarray) or data.ndim != 2 or data.size == 0:
        prColor(f"Invalid data for {result_type}: must be non-empty 2D array", 'red')
        return saved_files
    
    if save_2d:
        filepath_2d = save_2d_figure_with_colorbar(data, result_type, output_dir, run_name, cmap=cmap_2d)
        if filepath_2d:
            saved_files.append(filepath_2d)
    
    if save_3d:
        filepath_3d = save_3d_figure_with_colorbar(data, result_type, output_dir, run_name, cmap=cmap_3d)
        if filepath_3d:
            saved_files.append(filepath_3d)
    
    return saved_files

def image_preprocess(image, have_dark, dark_img, have_flat, flat_img):
    '''
        do the flat or dark correction for the images
        img:            image to be corrected
        have_dark:      if there is dark
        dark_img:           dark image
        have_flat:      if there is flat
        flat_img:           flat image        
    '''
    if (have_flat != 0 and have_dark != 0):
        numerator = (flat_img - dark_img).clip(0.00000001)
        # numerator = numerator / np.amax(numerator)
        image = ((image - dark_img) / numerator) * np.amax(image)
    elif (have_dark != 0):
        image = (image - dark_img).clip(0.00000001)
    elif (have_flat != 0):
        flat_img[flat_img == 0] = 0.00000001
        # flat_img = flat_img / np.amax(flat_img)
        image = (image / flat_img) * np.amax(image)

def zernike_radial(rho, n, m):
    """
    Calculates the radial part of Zernike polynomials.

    Parameter order is ``(rho, n, m)``. A shadowed earlier definition
    ``zernike_radial(n, m, r)`` was deleted in R1.4; in-tree call sites
    already used this later signature. See ``src/wsvt/reference/README.md``.
    """
    import math
    if (n - m) % 2 != 0:
        return np.zeros_like(rho)
    
    radial_poly = np.zeros_like(rho, dtype=float)
    # Correct the range to be inclusive of n - 2*k
    for k in range((n - m) // 2 + 1):
        num = (-1)**k * math.factorial(n - k)
        den = math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k)
        radial_poly += num / den * rho**(n - 2*k)
        
    return radial_poly

def zernike(rho, phi, n, m):
    """
    Calculates the Zernike polynomial Z_n^m(rho, phi).
    """
    if m >= 0:
        return zernike_radial(rho, n, m) * np.cos(m * phi)
    else:
        return zernike_radial(rho, n, -m) * np.sin(-m * phi)

def noll_to_nm(j):
    """
    Converts a single Noll index to (n, m) pair.
    """
    if j < 1:
        raise ValueError("Noll index must be a positive integer.")
    n = int(np.floor((-1 + np.sqrt(8*j - 7)) / 2))
    p = j - n*(n+1)//2
    if n % 4 in (0, 1):
        m = -n + 2*p - 2
    else:
        m = n - 2*p + 2
    # Simplified logic for m sign
    if m % 2 == 0:
        m_abs = abs(m)
    else:
        m_abs = abs(m) + 1

    if (j - n*(n+1)//2) % 2 == 1:
        m = m_abs
    else:
        m = -m_abs
        
    # Correction for m=0 cases
    if n % 2 == 0 and m != 0:
        if (j % 2) == 1:
            m = abs(m)
        else:
            m = -abs(m)
    elif n % 2 != 0:
        if (j % 2) == 1:
            m = abs(m)
        else:
            m = -abs(m)

    # Final check for n, m parity
    if (n-m) % 2 != 0:
        # This case should ideally not be hit with correct logic
        pass

    # A more direct approach from Wikipedia might be better if issues persist
    n = 0
    while (n + 1) * (n + 2) / 2 < j:
        n += 1
    m_rem = j - n * (n + 1) / 2
    if n % 2 == 0:  # even n
        if m_rem % 2 == 1:  # odd remainder
            m = (m_rem + 1) / 2 * 2 - 1
        else:  # even remainder
            m = - (m_rem / 2 * 2)
    else:  # odd n
        if m_rem % 2 == 1:
            m = (m_rem + 1) / 2 * 2 - 1
        else:
            m = - (m_rem / 2 * 2)
    return n, int(m)
