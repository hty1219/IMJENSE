#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:06:18 2023
@author: frm

This script is a demo for IMJENSE reconstruction  

"""
import sys

from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append("/data0/tianyu/code/IMJENSE")
import numpy as np
import h5py as h5
import os
from numpy import fft
import torch
from skimage.metrics import structural_similarity as compute_ssim
from skimage.io import imsave
from scipy.io import loadmat
import utils
import IMJENSE
import time
from datetime import datetime
import matplotlib.pyplot as plt
import math

os.environ["CUDA_VISIBLE_DEVICES"] = '6'
DEVICE = torch.device('cuda:{}'.format(str(0) if torch.cuda.is_available() else 'cpu'))

data_root = '/data0/shanghong/data/fastmri_knee_mc/multicoil_train/'
"""
file_list_knee = [
    'file1000003.h5', 'file1000005.h5', 'file1000010.h5', 'file1000012.h5',
    'file1000021.h5', 'file1000032.h5', 'file1000040.h5', 'file1000043.h5',
    'file1000045.h5', 'file1000048.h5', 'file1000058.h5', 'file1000061.h5',
    'file1000069.h5', 'file1000075.h5', 'file1000084.h5', 'file1000086.h5',
    'file1000089.h5', 'file1000098.h5', 'file1000101.h5', 'file1000109.h5'
]
"""
file_list_knee = [
    'file1000003.h5','file1000005.h5',
]

file_paths = [os.path.join(data_root, f) for f in file_list_knee]
print(f"Total files to process: {len(file_paths)}")
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = os.path.join('./results_experiments', f'exp_{current_time}')
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

print(f"Results will be saved to: {experiment_dir}")
print(f"Total files to process: {len(file_list_knee)}")

psnr_list = []
ssim_list = []
nmse_list = []
#%% load fully sampled k-space data
for idx, fname in enumerate(file_list_knee):
    fpath = os.path.join(data_root, fname)
    print(f"\nProcessing ({idx + 1}/{len(file_list_knee)}): {fname}")

    if not os.path.exists(fpath):
        print(f"Error: File not found {fpath}")
        continue

    try:
        # Load data
        f = h5.File(fpath, 'r')
        if 'KspOrg' in f:
            data_cpl = f['KspOrg'][:]
        elif 'kspace' in f:
            data_cpl = f['kspace'][:]
        else:
            print(f"Skipping {fname}: No 'KspOrg' or 'kspace' key found.")
            continue
        n_slices = data_cpl.shape[0]
        slice_idx = n_slices // 2
        data_cpl = data_cpl[slice_idx]
        print(f"selected middle slice index: {slice_idx}")

        #resample to 320x320
        img_temp = fft.fftshift(fft.ifft2(fft.fftshift(data_cpl, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))

        _, current_h, current_w = img_temp.shape
        target_size = 320

        if current_h >= target_size and current_w >= target_size:
            cx, cy = current_h // 2, current_w // 2
            start_x = cx - target_size // 2
            start_y = cy - target_size // 2
            img_crop = img_temp[:, start_x:start_x + target_size, start_y:start_y + target_size]

            data_cpl = fft.fftshift(fft.fft2(fft.fftshift(img_crop, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
            print(f"Cropped data to {target_size}x{target_size}. New shape: {data_cpl.shape}")
        else:
            print(f"Warning: Image size ({current_h}x{current_w}) is smaller than target 320x320. Skipping crop.")

        Nchl, Nrd, Npe = data_cpl.shape

        # Parameter settings
        Rx = 1
        Ry = 4
        num_ACSx = Nrd
        num_ACSy = 24
        w0 = 31
        lamda = 3.8
        fn = lambda x: utils.normalize01(np.abs(x))

        # Calculate GT
        img_all = fft.fftshift(fft.ifft2(fft.fftshift(data_cpl, axes=(-1, -2)), axes=(-1, -2)), axes=(-1, -2))
        gt = np.sqrt(np.sum(np.abs(img_all) ** 2, 0))

        # Undersampling
        tstKsp = data_cpl.transpose(1, 2, 0)
        SamMask = utils.KspaceUnd(Nrd, Npe, Rx, Ry, num_ACSx, num_ACSy)
        SamMask = np.tile(np.expand_dims(SamMask, axis=-1), (1, 1, Nchl))
        tstDsKsp = tstKsp * SamMask

        # Normalize
        zf_coil_img = fft.fftshift(fft.ifft2(fft.fftshift(tstDsKsp, axes=(0, 1)), axes=(0, 1)), axes=(0, 1))
        NormFactor = np.max(np.sqrt(np.sum(np.abs(zf_coil_img) ** 2, axis=2)))
        tstDsKsp = tstDsKsp / NormFactor

        # Reconstruction
        time_start = time.time()
        pre_img, pre_tstCsm, pre_img_dc, pre_img_sos, pre_ksp = IMJENSE.IMJENSE_Recon(
            tstDsKsp, SamMask, DEVICE, w0=w0, TV_weight=lamda, PolyOrder=15,
            MaxIter=1500, LrImg=1e-4, LrCsm=0.1
        )
        time_end = time.time()
        print(f'Recon time: {(time_end - time_start) / 60:.2f} mins')

        # Metrics
        normOrg = fn(gt)
        normRec = fn(pre_img_sos)

        psnrRec = utils.myPSNR(normOrg, normRec)
        ssimRec = compute_ssim(normRec, normOrg, data_range=1, gaussian_weights=True)
        nmseRec = np.sum((normOrg - normRec) ** 2) / np.sum(normOrg ** 2)

        psnr_list.append(psnrRec)
        ssim_list.append(ssimRec)
        nmse_list.append(nmseRec)

        print(f'{fname} -> PSNR: {psnrRec:.4f}, SSIM: {ssimRec:.4f}, NMSE: {nmseRec:.4f}')

        # Save results
        base_name = os.path.splitext(fname)[0]
        fig_comp, axes_comp = plt.subplots(3, 1, figsize=(6, 15))
        im_rec = axes_comp[0].imshow(normRec, cmap='gray', vmin=0, vmax=1)
        axes_comp[0].set_title(f'Predict\nPSNR: {psnrRec:.2f}, SSIM: {ssimRec:.4f}')
        axes_comp[0].axis('off')

        divider0 = make_axes_locatable(axes_comp[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        fig_comp.colorbar(im_rec, cax=cax0)

        im_gt = axes_comp[1].imshow(normOrg, cmap='gray', vmin=0, vmax=1)
        axes_comp[1].set_title('True')
        axes_comp[1].axis('off')

        divider1 = make_axes_locatable(axes_comp[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        fig_comp.colorbar(im_gt, cax=cax1)

        error_map = np.abs(normOrg - normRec)
        im_err = axes_comp[2].imshow(error_map, cmap='jet', vmin=0, vmax=0.1)
        axes_comp[2].set_title('Error')
        axes_comp[2].axis('off')

        divider2 = make_axes_locatable(axes_comp[2])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        fig_comp.colorbar(im_err, cax=cax2)

        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, f'{base_name}_comparison.png'), dpi=300)
        plt.close(fig_comp)

        n_channels = pre_tstCsm.shape[2]
        grid_side = math.ceil(math.sqrt(n_channels))

        fig_csm, axes_csm = plt.subplots(grid_side, grid_side, figsize=(12, 12))
        axes_csm = axes_csm.flatten()

        for i in range(grid_side * grid_side):
            if i < n_channels:
                coil_img = np.abs(pre_tstCsm[:, :, i])

                if np.max(coil_img) > 0:
                    coil_img = coil_img / np.max(coil_img)

                axes_csm[i].imshow(coil_img, cmap='jet')
                axes_csm[i].set_title(f'Coil {i + 1}', fontsize=8)

            axes_csm[i].axis('off')

        fig_csm.suptitle(f'Sensitivity Maps ({n_channels} Coils)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(experiment_dir, f'{base_name}_coils_csm.png'), dpi=300)
        plt.close(fig_csm)
    except Exception as e:
        print(f"Error processing {fname}: {e}")
        continue
        

print("\n" + "="*30)
print("Batch Processing Complete")
if len(psnr_list) > 0:
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_nmse = np.mean(nmse_list)

    std_psnr = np.std(psnr_list)
    std_ssim = np.std(ssim_list)
    std_nmse = np.std(nmse_list)
    print(f"Processed {len(psnr_list)} files successfully.")
    print(f"Average PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
    print(f"Average NMSE: {avg_nmse:.4f} ± {std_nmse:.4f}")
else:
    print("No files were processed successfully.")
print("="*30)