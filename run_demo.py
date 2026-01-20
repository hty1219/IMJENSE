#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modified for batch processing all slices in all h5 files.
"""
import sys
import mridataset
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob

sys.path.append("/data0/tianyu/code/IMJENSE")
import numpy as np
import h5py as h5
import os
from numpy import fft
import torch
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.io import imsave
from scipy.io import loadmat
import utils
import IMJENSE
import time
from datetime import datetime
import matplotlib.pyplot as plt
import math

DEVICE = torch.device('cuda:{}'.format(str(0) if torch.cuda.is_available() else 'cpu'))

#data_root = '/data0/shanghong/data/fastmri_knee_mc/multicoil_train/'
#data_root = '/data1/zijian/data/stanford3dcor_tiny/test'
data_root = '/data1/zijian/data/aheadax_e5_tiny/test'


print("正在读取文件列表...")
file_list_paths = glob.glob(os.path.join(data_root, '*.h5'))
#file_list_knee = [os.path.basename(p) for p in file_list_paths[:1]]
file_list_knee = [os.path.basename(p) for p in file_list_paths]

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = os.path.join('./results_experiments', f'exp_FULL_{current_time}')
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)

print(f"Results will be saved to: {experiment_dir}")
print(f"Total files found: {len(file_list_knee)}")

psnr_list = []
ssim_list = []
nmse_list = []


total_slices_processed = 0
file_slice_stats = {}

for file_idx, fname in enumerate(file_list_knee):
    fpath = os.path.join(data_root, fname)
    print(f"\n[{file_idx + 1}/{len(file_list_knee)}] Reading File: {fname}")

    if not os.path.exists(fpath):
        print(f"Error: File not found {fpath}")
        continue

    try:
        # Load data
        f = h5.File(fpath, 'r')

        vol_max = f.attrs.get('max')
        if vol_max is None:
            vol_max = f['max'][()] if 'max' in f else None

        if 'KspOrg' in f:
            kspace_vol = f['KspOrg'][:]
        elif 'kspace' in f:
            kspace_vol = f['kspace'][:]
        else:
            print(f"Skipping {fname}: No 'KspOrg' or 'kspace' key found.")
            f.close()
            continue

        n_slices = kspace_vol.shape[0]
        file_slice_stats[fname] = n_slices  # 记录统计信息
        print(f"  -> Found {n_slices} slices in {fname}")

        for slice_idx in range(n_slices):
            print(f"    Processing Slice {slice_idx + 1}/{n_slices} (Total Processed: {total_slices_processed + 1})")

            data_cpl = kspace_vol[slice_idx]


            # resample to 320x320
            img_temp = fft.fftshift(fft.ifft2(fft.fftshift(data_cpl, axes=(-1, -2)), axes=(-1, -2), norm='ortho'),
                                    axes=(-1, -2))

            _, current_h, current_w = img_temp.shape
            orig_h, orig_w = current_h, current_w
            target_size = 320

            pad_h = max(0, target_size - current_h)
            pad_w = max(0, target_size - current_w)

            if pad_h > 0 or pad_w > 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left

                img_temp = np.pad(img_temp,
                                  ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
                                  mode='reflect')
                # print(f"      Padded to {img_temp.shape[1]}x{img_temp.shape[2]}")
                _, current_h, current_w = img_temp.shape

            if current_h >= target_size and current_w >= target_size:
                cx, cy = current_h // 2, current_w // 2
                start_x = cx - target_size // 2
                start_y = cy - target_size // 2
                img_crop = img_temp[:, start_x:start_x + target_size, start_y:start_y + target_size]

                # Transform back to K-space
                data_cpl = fft.fftshift(fft.fft2(fft.fftshift(img_crop, axes=(-1, -2)), axes=(-1, -2), norm='ortho'),
                                        axes=(-1, -2))
            else:
                print(f"      Warning: Image size handling skipped for slice {slice_idx}. Shape: {img_temp.shape}")

            Nchl, Nrd, Npe = data_cpl.shape

            # Parameter settings
            Rx = 1
            Ry = 4
            num_ACSx = Nrd
            num_ACSy = 26
            #w0 = 31
            #lamda = 3.8
            w0 = 22
            lamda = 4.2
            fn = lambda x: utils.normalize01(np.abs(x))

            # Calculate GT
            img_all = fft.fftshift(fft.ifft2(fft.fftshift(data_cpl, axes=(-1, -2)), axes=(-1, -2), norm='ortho'),
                                   axes=(-1, -2))
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
            # time_start = time.time()
            pre_img, pre_tstCsm, pre_img_dc, pre_img_sos, pre_ksp = IMJENSE.IMJENSE_Recon(
                tstDsKsp, SamMask, DEVICE, w0=w0, TV_weight=lamda, PolyOrder=15,
                MaxIter=1500, LrImg=1e-4, LrCsm=0.1
            )
            # time_end = time.time()
            # print(f'      Recon time: {(time_end - time_start) / 60:.2f} mins')

            if vol_max is not None:
                # normOrg = np.abs(gt) / vol_max
                # normRec = np.abs(pre_img_sos * NormFactor) / vol_max
                pass
            else:
                # print("      Warning: Volume Max not found, using slice-wise normalization.")
                pass

            fft_scale_correction = np.sqrt(Nrd * Npe)
            recon_restored = pre_img_sos * NormFactor * fft_scale_correction
            gt_restored = gt

            print(f"Shape before cropping black border: {gt_restored.shape, recon_restored.shape}")
            metric_h = min(orig_h, target_size)
            metric_w = min(orig_w, target_size)
            gt_restored = mridataset.center_crop(gt_restored, (metric_h, metric_w))
            recon_restored = mridataset.center_crop(recon_restored, (metric_h, metric_w))
            print(f"Shape after cropping black border: {gt_restored.shape, recon_restored.shape}")
            # Compute Metrics
            psnrRec = compute_psnr(gt_restored, recon_restored, data_range=vol_max)
            ssimRec = compute_ssim(gt_restored, recon_restored, data_range=vol_max)
            nmseRec = np.linalg.norm(gt_restored - recon_restored) ** 2 / np.linalg.norm(gt_restored) ** 2

            psnr_list.append(psnrRec)
            ssim_list.append(ssimRec)
            nmse_list.append(nmseRec)

            print(f'      Slice {slice_idx} -> PSNR: {psnrRec:.4f}, SSIM: {ssimRec:.4f}, NMSE: {nmseRec:.4f}')

            # --- Save Results (Modified Naming) ---
            base_name = os.path.splitext(fname)[0]
            folder_name = f"{base_name}_slice{slice_idx:03d}_PSNR{psnrRec:.2f}_SSIM{ssimRec:.4f}_NMSE{nmseRec:.4f}"
            current_slice_dir = os.path.join(experiment_dir, folder_name)

            if not os.path.exists(current_slice_dir):
                os.makedirs(current_slice_dir)

            # 1. Comparison Plot
            fig_comp, axes_comp = plt.subplots(3, 1, figsize=(6, 15))
            im_rec = axes_comp[1].imshow(recon_restored, cmap='gray', vmin=0,
                                         vmax=1 if vol_max is None else gt_restored.max())
            axes_comp[1].set_title(f'Predict (Slice {slice_idx})')
            axes_comp[1].axis('off')

            divider0 = make_axes_locatable(axes_comp[1])
            cax0 = divider0.append_axes("right", size="5%", pad=0.05)
            fig_comp.colorbar(im_rec, cax=cax0)

            im_gt = axes_comp[0].imshow(gt_restored, cmap='gray', vmin=0,
                                        vmax=1 if vol_max is None else gt_restored.max())
            axes_comp[0].set_title(f'True (Slice {slice_idx})')
            axes_comp[0].axis('off')

            divider1 = make_axes_locatable(axes_comp[0])
            cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            fig_comp.colorbar(im_gt, cax=cax1)

            error_map = np.abs(gt_restored - recon_restored)
            im_err = axes_comp[2].imshow(error_map, cmap='jet', vmin=0,
                                         vmax=0.1 if vol_max is None else 0.1 * gt_restored.max())
            axes_comp[2].set_title('Error')
            axes_comp[2].axis('off')

            divider2 = make_axes_locatable(axes_comp[2])
            cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            fig_comp.colorbar(im_err, cax=cax2)

            plt.tight_layout()
            plt.savefig(os.path.join(current_slice_dir, f'{base_name}_slice{slice_idx:03d}_comparison.png'), dpi=300)
            plt.close(fig_comp)

            # 2. Sensitivity Maps Plot (Only process if needed, skip to save time if many coils)
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

            fig_csm.suptitle(f'Sensitivity Maps (Slice {slice_idx})', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(current_slice_dir, f'{base_name}_slice{slice_idx:03d}_coils_csm.png'), dpi=300)
            plt.close(fig_csm)

            # 3. Mask Plot
            fig_mask = plt.figure(figsize=(5, 5))
            plt.imshow(SamMask[:, :, 0], cmap='gray', vmin=0, vmax=1)
            plt.title('Sampling Mask')
            plt.axis('off')
            plt.savefig(os.path.join(current_slice_dir, f'{base_name}_slice{slice_idx:03d}_mask.png'), dpi=300,
                        bbox_inches='tight', pad_inches=0.05)
            plt.close(fig_mask)

            total_slices_processed += 1

        # Close file after processing all slices
        f.close()

    except Exception as e:
        print(f"Error processing {fname}: {e}")
        import traceback

        traceback.print_exc()
        continue

# --- 修改 4: 最终统计报告 ---
print("\n" + "=" * 50)
print("EXPERIMENT COMPLETE")
print(f"Total Files Scanned: {len(file_list_knee)}")
print(f"Total Slices Processed: {total_slices_processed}")
print("-" * 50)
print("Slice Counts Per File:")
for name, count in file_slice_stats.items():
    print(f"  {name}: {count} slices")
print("-" * 50)

if len(psnr_list) > 0:
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_nmse = np.mean(nmse_list)

    std_psnr = np.std(psnr_list)
    std_ssim = np.std(ssim_list)
    std_nmse = np.std(nmse_list)

    print(f"Metrics (Average over {len(psnr_list)} slices):")
    print(f"  Average PSNR: {avg_psnr:.5f} ± {std_psnr:.5f}")
    print(f"  Average SSIM: {avg_ssim:.5f} ± {std_ssim:.5f}")
    print(f"  Average NMSE: {avg_nmse:.5f} ± {std_nmse:.5f}")
else:
    print("No files were processed successfully.")
print("=" * 50)