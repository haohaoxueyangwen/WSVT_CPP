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

# Import necessary libraries
import numpy as np                     # For numerical operations
import pywt                           # PyWavelets library for wavelet transforms
import os                             # For file and directory operations
import sys                            # For system-specific parameters and functions
import time                           # For timing code execution
from PIL import Image                 # For image processing
import glob                           # For finding pathnames matching a pattern
import scipy.constants as sc          # For scientific constants
from .func import prColor, frankotchellappa, image_roi, Wavelet_transform, write_h5, write_json, find_disp  # Custom functions
from .euclidean_dist import dist_numba  # Distance calculation function with Numba acceleration
import matplotlib                     # For plotting
# matplotlib.use('Agg')               # Uncomment to use non-GUI backend
from matplotlib import pyplot as plt  # For plotting
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from matplotlib import cm             # For colormap
import multiprocessing as ms          # For parallel processing
import concurrent.futures             # For parallel execution
import scipy.interpolate as sfit      # For interpolation
import copy                           # For creating deep copies of objects


def load_image(file_path):#加载图像的函数
    """
    Load an image from file path and convert to numpy array
    
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
    return np.array(img)

def image_preprocess(image, have_dark, dark_img, have_flat, flat_img):# 图像预处理函数
    '''
    Perform flat field and dark field corrections on an image
    
    Args:
        image (numpy.ndarray): Raw image to be corrected # 确认原始图像
        have_dark (bool): Flag indicating if dark image is available # 确认是否存在暗场图像
        dark_img (numpy.ndarray): Dark image for correction # 暗场图像
        have_flat (bool): Flag indicating if flat image is available # 确认是否存在平场图像
        flat_img (numpy.ndarray): Flat image for correction # 平场图像
        暗场矫正：消除本底噪声和暗电流
        平场矫正：补偿各像素不均匀性和x射线强度空间分布不均匀性
        目的：消除系统误差
    Returns:
        numpy.ndarray: Corrected image # 返回校正后的图像
    '''
    if (have_flat != 0 and have_dark != 0): # 如果两个图像都存在
        # Apply both dark and flat field corrections
        numerator = (flat_img - dark_img).clip(0.00000001) # 分子
        # numerator = numerator / np.amax(numerator)
        image = ((image - dark_img) / numerator) * np.amax(image)
    elif (have_dark != 0): # 如果只有暗场图像存在
        # Apply only dark field correction
        image = (image - dark_img).clip(0.00000001)
    elif (have_flat != 0): # 如果只有平场图像存在
        # Apply only flat field correction
        flat_img[flat_img == 0] = 0.00000001 # 将平场图像中为0的像素值设置为0.00000001
        # flat_img = flat_img / np.amax(flat_img)
        image = (image / flat_img) * np.amax(image) # 校正图像

    return image

class WXST:
    """
    Wavelet-transform-based Single shot Speckle Tracking class #定义小波变换xst类
    
    This class implements the WXST algorithm for single-shot speckle tracking using
    wavelet transforms and multi-resolution analysis to accelerate processing and
    increase noise robustness.
    # 用小波变换和多分辨率分析来加速处理并增加噪声鲁棒性
    """
    def __init__(self,
                 img, # 样本图像
                 ref, # 参考图像
                 M_image=512, # 图像大小
                 N_s=5, # 模板窗口大小
                 cal_half_window=20, # 计算窗口大小
                 N_s_extend=4, # 高阶金字塔计算窗口大小
                 n_cores=4, # 核心数
                 n_group=4, # 组数
                 energy=14e3, # 能量
                 p_x=0.65e-6, # 像素大小
                 z=500e-3, # 距离
                 wavelet_level_cut=2, # 小波变换层数
                 pyramid_level=2, # 金字塔层数
                 n_iter=1, # 迭代次数
                 use_wavelet=True): # 是否使用小波变换  
        """
        Initialize the WXST solver with parameters
        
        Args:
            img (numpy.ndarray): Sample image
            ref (numpy.ndarray): Reference image
            M_image (int): ROI size of the images
            N_s (int): Template window size (2*N_s+1)
            cal_half_window (int): Half window size for calculation area (2*cal_half_window)
            N_s_extend (int): Calculation window for high order pyramid
            n_cores (int): Number of CPU cores for parallel processing
            n_group (int): Number to reduce memory usage per process
            energy (float): X-ray energy in eV
            p_x (float): Pixel size in meters
            z (float): Sample to detector distance in meters
            wavelet_level_cut (int): Wavelet coefficient cutting level
            pyramid_level (int): Pyramid level for multi-resolution analysis
            n_iter (int): Number of iterations for calculation
            use_wavelet (bool): Whether to use wavelet transform
        """
        self.img_data = img # 调用样本图像
        self.ref_data = ref # 调用参考图像
        # roi of the images
        self.M_image = M_image # 从字典中调用图像大小
        # template window, the N_s nearby pixels used to represent the local pixel, 2*N_s+1
        self.N_s = N_s # 调用模版窗口大小
        # the number of the area to calculate for each pixel, 2*cal_half_window X 2*cal_half_window
        self.cal_half_window = cal_half_window # 调用计算窗口大小
        # the calculation window for high order pyramid
        self.N_s_extend = N_s_extend # 调用高阶金字塔计算窗口大小

        # process number for parallel
        self.n_cores = n_cores # 调用核心数
        # number to reduce the each memory use
        self.n_group = n_group ##

        # energy, 10kev
        self.energy = energy # 调用能量
        self.wavelength = sc.value(
            'inverse meter-electron volt relationship') / energy # 调用波长
        # pixel size [m]
        self.p_x = p_x # 调用像素大小
        # distance [m]
        self.z = z # 调用距离
        self.wavelet_level_cut = wavelet_level_cut # 调用小波变换层数
        # pyramid level to wrap the images
        self.pyramid_level = pyramid_level # 调用金字塔层数
        # iterations for the calculation
        self.n_iter = n_iter # 调用迭代次数
        # if use wavelet transform or not
        self.use_wavelet = use_wavelet # 是否使用小波变换   

    def template_stack(self, img): # 模板堆叠函数
        '''
        Create a stack of shifted images for template matching # 创建一个用于模板匹配的移位图像堆栈
        
        This function creates a stack of shifted images by rolling the input image 
        # 通过在不同方向上滚动输入图像来创建一个移位图像堆栈，用于模板匹配
        in different directions, which is used for template matching.
        
        Args:
            img (numpy.ndarray): Input image # 加载图像
            
        Returns:
            numpy.ndarray: Stack of shifted images # 返回移位图像堆栈
        '''
        img_stack = [] # 创建一个空列表来存储移位图像
        axis_Nw = np.arange(-self.N_s, self.N_s + 1) # 创建一个包含从-N_s到N_s的数组
        for x in axis_Nw:
            for y in axis_Nw:
                img_stack.append(np.roll(np.roll(img, x, axis=0), y, axis=1))
                # 使用np.roll函数在x和y方向上滚动图像，并将结果添加到img_stack列表中
        return np.array(img_stack)
    ''' 移位操作原理：
    1. 创建一个包含从-N_s到N_s的数组 ，然后历遍xy。
    2. 使用np.roll函数在x和y方向上滚动图像，并将结果添加到img_stack列表中
    3. 返回移位图像堆栈
    这个c操作的作用：1.通过比较移位后的图像和参考图像的相似度，找到最佳匹配的位移量，从而找到最大相关性的位移值，从而找到最佳的
    匹配位置
    2.通过不同位移的相关性分析

    '''

    def pyramid_data(self):
        """
        Generate pyramid（金字塔） data for multi-resolution analysis #产生金字塔数据对多分辨率分析
        
        This method creates a pyramid of downsampled images using wavelet transform,
        and stacks the template windows for each pyramid level. # 这个方法通过使用小波变换创建一个降采样的金字塔模型
        然后为每一个金字塔层创建模版窗口堆叠
        
        
        Returns:
            tuple: (ref_pyramid, img_pyramid) - Lists of reference and sample image pyramids
        """
        # get the pyramid data
        # method 1, pyramid then stack the image, which means the template window size is increasing for different pyramid level
        # get the pyramid wrapping images
        ref_pyramid = [] # 创建一个空列表来存储参考图像金字塔
        img_pyramid = [] # 创建一个空列表来存储样本图像金字塔
        prColor(
            'obtain pyramid image and stack the window with pyramid level: {}'.
            format(self.pyramid_level), 'green')
        ref_pyramid.append(self.ref_data) # 将参考图像添加到参考图像金字塔中
        img_pyramid.append(self.img_data) # 将样本图像添加到样本图像金字塔中     

        # Generate pyramid levels using wavelet transform # 使用小波变换生成金字塔层
        for kk in range(self.pyramid_level): # 历遍金字塔层
            ref_pyramid.append(
                pywt.dwtn(ref_pyramid[kk], 'db3', mode='zero',
                          axes=(-2, -1))['aa']) # 将参考图像添加到参考图像金字塔中
            img_pyramid.append(
                pywt.dwtn(img_pyramid[kk], 'db3', mode='zero',
                          axes=(-2, -1))['aa']) # 将样本图像添加到样本图像金字塔中  

        # Normalize the data by standard deviation for better comparison
        normlize_std = lambda img: ( # 标准差归一化
            (img - np.ndarray.mean(img, axis=0)) / np.ndarray.std(img, axis=0))# 标准差归一化

        # Apply template stacking to each pyramid level # 对每个金字塔层应用模板堆叠
        ref_pyramid = [
            normlize_std(self.template_stack(img_data))# 
            for img_data in ref_pyramid # 对参考图像金字塔中的每个图像数据进行标准差归一化
        ]
        img_pyramid = [
            normlize_std(self.template_stack(img_data))
            for img_data in img_pyramid
        ]

        return ref_pyramid, img_pyramid

    def resampling_spline(self, img, s):
        """
        Resample an image to a new size using spline interpolation
        
        Args:
            img (numpy.ndarray): Original image
            s (tuple): Target size (rows, columns)
            
        Returns:
            numpy.ndarray: Resampled image
        """
        # img: original
        # s: size of the sampling, (row, col)
        m, n = img.shape
        x_axis = np.arange(n)
        y_axis = np.arange(m)
        fit = sfit.RectBivariateSpline(y_axis, x_axis, img)

        x_new = np.linspace(0, n - 1, s[1])
        y_new = np.linspace(0, m - 1, s[0])

        return fit(y_new, x_new)

    def wavelet_data(self):
        self.pyramid_time = 0.0
        self.wavelet_time = 0.0
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

            # Adjust wavelet level based on image size
            if ref_pyramid[0].shape[0] > 150:
                self.wavelet_add_list = [0, 0, 0, 0, 0, 0]
            elif ref_pyramid[0].shape[0] > 50:
                self.wavelet_add_list = [0, 0, 1, 2, 2, 2]
            else:
                self.wavelet_add_list = [2, 2, 2, 2, 2, 2]

            # Apply wavelet transform to each pyramid level
            start_time = time.time()
            for p_level in range(len(img_pyramid)):
                if p_level > len(self.wavelet_add_list):
                    wavelevel_add = 2
                else:
                    wavelevel_add = self.wavelet_add_list[p_level]

                # Transform the sample image
                img_wa, level_name = Wavelet_transform(
                    img_pyramid[p_level],
                    wavelet_method,
                    w_level=self.wavelet_level,
                    return_level=coefs_level + wavelevel_add)
                img_pyramid[p_level] = img_wa

                # Transform the reference image
                ref_wa, level_name = Wavelet_transform(
                    ref_pyramid[p_level],
                    wavelet_method,
                    w_level=self.wavelet_level,
                    return_level=coefs_level + wavelevel_add)
                ref_pyramid[p_level] = ref_wa

                prColor(
                    'pyramid level: {}\nvector length: {}\nUse wavelet coef: {}'
                    .format(p_level, ref_wa.shape[2], level_name), 'green')

            end_time = time.time()
            self.wavelet_time = end_time - start_time
            print('wavelet time: {}'.format(self.wavelet_time))
        else:
            # If not using wavelet transform, just rearrange axes
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
        '''
        Calculate the displacement for each pixel using wavelet coefficients
        
        Args:
            y_list (list): List of y-coordinates to process
            img_wa_stack (numpy.ndarray): Wavelet coefficients of sample image
            ref_wa_stack (numpy.ndarray): Wavelet coefficients of reference image
            displace_pyramid (tuple): Initial displacement from pyramid level
            cal_half_window (int): Half window size for calculation
            n_pad (int): Padding size
            
        Returns:
            tuple: (disp_add_y, disp_add_x, y_list) - Displacement in y and x directions, and y coordinates
        '''
        # dist_numba has an explicit float64 signature.  Keep this conversion
        # aligned with the original Python WSVT path and perform it once per
        # worker chunk rather than once per pixel.
        if img_wa_stack.dtype != np.float64:
            img_wa_stack = img_wa_stack.astype(np.float64)
        if ref_wa_stack.dtype != np.float64:
            ref_wa_stack = ref_wa_stack.astype(np.float64)

        dim = img_wa_stack.shape
        disp_x = np.zeros((dim[0], dim[1]))
        disp_y = np.zeros((dim[0], dim[1]))

        # the axis for the peak position finding
        window_size = 2 * cal_half_window + 1
        y_axis = np.arange(window_size) - cal_half_window
        x_axis = np.arange(window_size) - cal_half_window
        XX, YY = np.meshgrid(x_axis, y_axis)

        # Process each pixel in the region
        for yy in range(dim[0]):
            for xx in range(dim[1]):
                # Get wavelet coefficients for current pixel
                img_wa_line = img_wa_stack[yy, xx, :]
                # Get reference wavelet data within search window
                ref_wa_data = ref_wa_stack[
                    n_pad + yy + int(displace_pyramid[0][yy, xx]):n_pad + yy +
                    int(displace_pyramid[0][yy, xx]) + window_size,
                    n_pad + xx + int(displace_pyramid[1][yy, xx]):n_pad + xx +
                    int(displace_pyramid[1][yy, xx]) + window_size, :]

                # Calculate correlation using Euclidean distance
                Corr_img = dist_numba(img_wa_line, ref_wa_data)
                '''
                    use gradient to find the peak
                '''
                # Find displacement from correlation peak
                disp_y[yy, xx], disp_x[yy, xx] = find_disp(
                    Corr_img, XX, YY, sub_resolution=True)

        # Add the displacement to the pyramid displacement
        disp_add_y = displace_pyramid[0] + disp_y
        disp_add_x = displace_pyramid[1] + disp_x
        return disp_add_y, disp_add_x, y_list

    def solver(self):
        _t_solver_start = time.time()
        ref_pyramid, img_pyramid = self.wavelet_data()
        # Calculate transmission image with safety checks
        # Add small epsilon to avoid division by zero
        ref_safe = self.ref_data.astype(np.float64)
        img_safe = self.img_data.astype(np.float64)
        
        # Prevent division by zero by clipping very small values
        epsilon = 1e-10
        ref_safe = np.clip(ref_safe, epsilon, None)
        
        transmission = img_safe / ref_safe
        
        # Clip transmission to reasonable physical range (0.01 to 10)
        # Most X-ray transmission values should be between 0 and 2
        transmission = np.clip(transmission, 0.01, 10.0)
        
        # Free up memory
        for attr in ('img_data', 'ref_data'):
            self.__dict__.pop(attr, None)

        # Setup parallel processing
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
        # Setup pyramid search window sizes
        max_pyramid_searching_window = int(
            np.ceil(self.cal_half_window / 2**self.pyramid_level))
        searching_window_pyramid_list = [self.N_s_extend
                                         ] * self.pyramid_level + [
                                             int(max_pyramid_searching_window)
                                         ]
        m, n, c = img_pyramid[0].shape
        displace = [np.zeros((m, n)), np.zeros((m, n))]

        # Iterate to refine the displacement
        for k_iter in range(self.n_iter):
            # Downsample displacement for pyramid processing
            displace = [img / 2**self.pyramid_level for img in displace]

            m, n, c = img_pyramid[-1].shape
            # Resample displacement to match pyramid size
            displace[0] = self.resampling_spline(displace[0], (m, n))
            displace[1] = self.resampling_spline(displace[1], (m, n))

            prColor(
                'down sampling the dispalce to size: {}'.format(
                    displace[0].shape), 'green')

            # Clip displacement values to valid range
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

            # Process each pyramid level from coarsest to finest
            for p_level in range(self.pyramid_level, -1, -1):
                # Configure search window based on pyramid level
                if p_level == self.pyramid_level:
                    pyramid_seaching_window = searching_window_pyramid_list[
                        p_level]
                    m, n, c = img_pyramid[p_level].shape
                    displace_pyramid = [np.round(img) for img in displace]

                    n_pad = int(np.ceil(self.cal_half_window / 2**p_level))

                else:
                    pyramid_seaching_window = searching_window_pyramid_list[
                        p_level]

                    m, n, c = img_pyramid[p_level].shape
                    # Upsample displacement from previous level
                    displace_pyramid = [
                        np.round(self.resampling_spline(img * 2, (m, n)))
                        for img in displace
                    ]

                    # Clip displacement values to valid range
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

                # Split work into chunks for parallel processing
                y_axis = np.arange(ref_pyramid[p_level].shape[0])
                chunks_idx_y = np.array_split(y_axis, n_tasks)

                dim = img_pyramid[p_level].shape

                # Pad reference image for search window
                ref_wa_pad = np.pad(ref_pyramid[p_level],
                                    ((n_pad + pyramid_seaching_window,
                                      n_pad + pyramid_seaching_window),
                                     (n_pad + pyramid_seaching_window,
                                      n_pad + pyramid_seaching_window),
                                     (0, 0)),
                                    'constant',
                                    constant_values=(0, 0))

                # Use CPU parallel to calculate displacement
                result_list = []
                
                # Calculate the pixel displacement for the pyramid images using parallel processing
                with concurrent.futures.ProcessPoolExecutor(
                        max_workers=cores) as executor:

                    futures = []
                    # Submit jobs for each chunk
                    for y_list in chunks_idx_y:
                        # Get the stack data for this chunk
                        img_wa_stack = img_pyramid[p_level][y_list, :, :]
                        ref_wa_stack = ref_wa_pad[
                            y_list[0]:y_list[-1] + 2 *
                            (n_pad + pyramid_seaching_window) + 1, :, :]

                        # Start the jobs
                        futures.append(
                            executor.submit(self.displace_wavelet, y_list,
                                            img_wa_stack, ref_wa_stack,
                                            (displace_pyramid[0][y_list, :],
                                             displace_pyramid[1][y_list, :]),
                                            pyramid_seaching_window, n_pad))

                    # Collect results as jobs complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result_list.append(future.result())
                            # Display progress bar
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
                                'WXST parallel displacement task failed'
                            ) from exc

                # Extract results from parallel processing
                disp_y_list = [item[0] for item in result_list]
                disp_x_list = [item[1] for item in result_list]
                y_list = [item[2] for item in result_list]

                # Combine results from all chunks
                displace_y = np.zeros((dim[0], dim[1]))
                displace_x = np.zeros((dim[0], dim[1]))

                for y, disp_x, disp_y in zip(y_list, disp_x_list, disp_y_list):
                    displace_x[y, :] = disp_x
                    displace_y[y, :] = disp_y

                # Clip displacement values to valid range
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

        # Crop and negate displacement (convert to physical displacement)
        displace[0] = -displace[0][self.cal_half_window:-self.cal_half_window,
                                   self.cal_half_window:-self.cal_half_window]
        displace[1] = -displace[1][self.cal_half_window:-self.cal_half_window,
                                   self.cal_half_window:-self.cal_half_window]

        _t_post_start = time.time()
        # Calculate differential phase contrast
        DPC_y = (displace[0] - np.mean(displace[0])) * self.p_x / self.z
        DPC_x = (displace[1] - np.mean(displace[1])) * self.p_x / self.z

        # Reconstruct phase using Frankot-Chellappa algorithm
        phase = -frankotchellappa(
            DPC_x, DPC_y) * self.p_x * 2 * np.pi / self.wavelength
        _t_post_end = time.time()
        self.postprocess_time = _t_post_end - _t_post_start
        prColor('post-process time: {:.6f} s'.format(self.postprocess_time), 'light_purple')
        _t_solver_end = time.time()
        self.time_cost = _t_solver_end - _t_solver_start
        prColor('total time: {:.6f} s'.format(self.time_cost), 'light_purple')
        prColor('  pyramid:    {:.6f} s'.format(self.pyramid_time), 'light_purple')
        prColor('  wavelet:    {:.6f} s'.format(self.wavelet_time), 'light_purple')
        prColor('  displace:   {:.6f} s'.format(self.displace_time), 'light_purple')
        prColor('  post-proc:  {:.6f} s'.format(self.postprocess_time), 'light_purple')

        return displace, [DPC_y, DPC_x], phase, transmission

    def run(self, result_path=None):
        """
        Run the WXST algorithm and save results
        
        Args:
            result_path (str, optional): Path to save results. Defaults to None.
            
        Returns:
            dict: Dictionary containing analysis results
        """
        # Run the solver to get results
        self.displace, self.DPC, self.phase, self.transmission = self.solver()
        
        # Create results dictionary to return
        results_dict = {
            'displace_x': self.displace[1],
            'displace_y': self.displace[0],
            'DPC_x': self.DPC[1],
            'DPC_y': self.DPC[0],
            'phase': self.phase,
            'transmission': self.transmission
        }

        if result_path is not None:
            '''
            Save the calculation results
            '''
            # Create the results directory if it doesn't exist
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            # Create a unique filename for results
            self.result_filename = 'WXST' + str(self.M_image) + '_px_' + str(
                self.wavelet_level_cut) + 'wavelet_Cutlevel_' + str(
                    self.pyramid_level) + 'pyramid_level'

            # Ensure filename is unique by adding a number if needed
            kk = 1
            while os.path.exists(
                    os.path.join(
                        result_path,
                        self.result_filename + '.hdf5')) or os.path.exists(
                            os.path.join(result_path,
                                         self.result_filename + '.json')):
                self.result_filename = 'WXST' + str(
                    self.M_image) + '_px_' + str(
                        self.wavelet_level_cut) + 'wavelet_Cutlevel_' + str(
                            self.pyramid_level
                        ) + 'pyramid_level' + '_{}'.format(kk)
                kk += 1
            
            # Save results to HDF5 file
            write_h5(
                result_path, self.result_filename, {
                    'displace_x': self.displace[1],
                    'displace_y': self.displace[0],
                    'DPC_x': self.DPC[1],
                    'DPC_y': self.DPC[0],
                    'phase': self.phase,
                    'transmission_image': self.transmission
                })

            # Save parameters to JSON file
            parameter_dict = {
                'M_image': self.M_image,
                'template_window': self.N_s,
                'N_s extend': self.N_s_extend,
                'half_window': self.cal_half_window,
                'energy': self.energy,
                'wavelength': self.wavelength,
                'pixel_size': self.p_x,
                'z_distance': self.z,
                'cpu_cores': self.n_cores,
                'n_group': self.n_group,
                'wavelet_level': self.wavelet_level,
                'pyramid_level': self.pyramid_level,
                'n_iter': self.n_iter,
                'time_cost': self.time_cost,
                'pyramid_time': self.pyramid_time,
                'wavelet_time': self.wavelet_time,
                'displace_time': self.displace_time,
                'postprocess_time': self.postprocess_time,
                'use_wavelet': self.use_wavelet,
                'wavelet_level_cut': self.wavelet_level_cut,
                'wavelet_add': self.wavelet_add_list
            }

            write_json(result_path, self.result_filename, parameter_dict)
        
        return results_dict


def main():
    # Parse command line arguments based on number of arguments provided
    if len(sys.argv) == 1:
        # Default execution with no arguments - for testing
        Folder_path = 'Data_folder_path'
        File_ref = os.path.join(Folder_path, 'ref_001.tif')
        File_img = os.path.join(Folder_path, 'sample_001.tif')

        Folder_result = os.path.join(Folder_path, 'WXST_test')
        # Parameters: [image_size, template_window, cal_half_window, n_group, n_cores, energy, pixel_size, distance, use_wavelet, wavelet_ct, pyramid level, n_iteration]
        parameter_wavelet = [
            500, 5, 10, 4, 4, 14e3, 0.65e-6, 310e-3, 1, 2, 2, 1
        ]

    elif len(sys.argv) == 4:
        # Basic execution with sample, reference, and result path
        File_img = sys.argv[1]
        File_ref = sys.argv[2]
        Folder_result = sys.argv[3]
        # Default parameters
        parameter_wavelet = [
            1500, 5, 20, 4, 4, 14e3, 0.65e-6, 500e-3, 1, 2, 2, 1
        ]
    elif len(sys.argv) == 16:
        # Full parameter specification
        File_img = sys.argv[1]
        File_ref = sys.argv[2]
        Folder_result = sys.argv[3]
        parameter_wavelet = sys.argv[4:]
    else:
        # Invalid number of arguments
        prColor('Wrong parameters! should be: sample, ref, result', 'red')

    prColor('folder: {}'.format(Folder_result), 'green')
    
    # Parse parameters
    # ROI size of the images
    M_image = int(parameter_wavelet[0])
    # Template window size (2*N_s+1)
    N_s = int(parameter_wavelet[1])
    # Half window size for search area (2*cal_half_window)
    cal_half_window = int(parameter_wavelet[2])
    # Calculation window for high order pyramid
    N_s_extend = 4

    # Number of CPU cores for parallel processing
    n_cores = int(parameter_wavelet[3])
    # Number to reduce memory usage per process
    n_group = int(parameter_wavelet[4])

    # X-ray energy in eV
    energy = float(parameter_wavelet[5])
    # Calculate wavelength from energy
    wavelength = sc.value('inverse meter-electron volt relationship') / energy
    # Pixel size in meters
    p_x = float(parameter_wavelet[6])
    # Sample to detector distance in meters
    z = float(parameter_wavelet[7])
    # Whether to use wavelet transform
    use_wavelet = int(parameter_wavelet[8])
    # Wavelet coefficient cutting level
    wavelet_level_cut = int(parameter_wavelet[9])
    # Pyramid level for multi-resolution analysis
    pyramid_level = int(parameter_wavelet[10])
    # Number of iterations
    n_iter = int(parameter_wavelet[11])

    # Load reference and sample images
    ref_data = load_image(File_ref)
    img_data = load_image(File_img)

    # Examples of image preprocessing (commented out)
    # radio_flat = load_images(Folder_radio, 'flat*.tif')
    # radio_dark = load_images(Folder_radio, 'dark*.tif')
    # ref_data = image_preprocess(ref_data, have_dark=1, dark_img=radio_dark, have_flat=1, flat_img=radio_flat)
    # img_data = image_preprocess(img_data, have_dark=1, dark_img=radio_dark, have_flat=1, flat_img=radio_flat)

    # Extract region of interest
    ref_data = image_roi(ref_data, M_image)
    img_data = image_roi(img_data, M_image)

    # Create and initialize WXST solver
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
                       use_wavelet=use_wavelet)

    # Create results directory if needed
    if not os.path.exists(Folder_result):
        os.makedirs(Folder_result)
        
    # Calculate and save transmission image with safety checks
    ref_safe = ref_data.astype(np.float64)
    img_safe = img_data.astype(np.float64)
    epsilon = 1e-10
    ref_safe = np.clip(ref_safe, epsilon, None)
    sample_transmission = img_safe / ref_safe
    sample_transmission = np.clip(sample_transmission, 0.01, 10.0)
    plt.imsave(os.path.join(Folder_result, 'transmission.png'),
               sample_transmission)

    # Run the WXST algorithm
    WXST_solver.run(result_path=Folder_result)

    # Extract results for visualization
    displace = WXST_solver.displace
    DPC_x = WXST_solver.DPC[1]
    DPC_y = WXST_solver.DPC[0]
    phase = WXST_solver.phase
    result_filename = WXST_solver.result_filename

    # Save result images
    plt.imsave(os.path.join(Folder_result, 'displace_x.png'), displace[1])
    plt.imsave(os.path.join(Folder_result, 'displace_y.png'), displace[0])
    plt.imsave(os.path.join(Folder_result, 'dpc_x.png'), DPC_x)
    plt.imsave(os.path.join(Folder_result, 'dpc_y.png'), DPC_y)
    plt.imsave(os.path.join(Folder_result, 'phase.png'), phase)

    # Create and save figures with colorbars
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

    # Create and save 3D surface plot of phase
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    XX, YY = np.meshgrid(
        np.arange(phase.shape[1]) * p_x * 1e6,
        np.arange(phase.shape[0]) * p_x * 1e6)
    ax1.plot_surface(XX, YY, phase, cmap=cm.get_cmap('hot'))
    ax1.set_xlabel(r'x [$\mu m$]')
    ax1.set_ylabel(r'y [$\mu m$]')
    ax1.set_zlabel('phase [rad]')
    plt.savefig(os.path.join(Folder_result, 'Phase_3d.png'))

    # Close all plot windows
    plt.close()


if __name__ == "__main__":
    main()
