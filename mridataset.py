import numpy as np
import torch

def ifft2_np(x: np.ndarray) -> np.ndarray:
    """Centered 2D IFFT (ortho)."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))

def fft2_np(x: np.ndarray) -> np.ndarray:
    """Centered 2D FFT (ortho)."""
    return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x, axes=(-2, -1)), norm="ortho"), axes=(-2, -1))

def to_tensor(data: np.ndarray) -> torch.Tensor:
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)

def kspace_to_target(x: np.ndarray) -> np.ndarray:
    """Generate RSS reconstruction target from k-space data.

    Args:
        x: K-space data

    Returns:
        RSS reconstruction target
    """
    return np.sqrt(np.sum(np.square(np.abs(ifft2_np(x))), axis=-3)).astype(np.float32)
def ifft2c_tensor(x):
    """ifft2c with tensor as input/output, used for k-space to image.

    Shape:
        Input: tensor, shape=[Nbs, Npe, Nread, 2], dtype=float
        Output: tensor, shape=[Nbs, Npe, Nread, 2], dtype=float
    """
    x = torch.complex(x[...,0], x[...,1]).numpy()
    x = ifft2_np(x)
    x = torch.from_numpy(x)
    x = torch.stack((x.real, x.imag), dim=-1)
    return x

def fft2c_tensor(x):
    """fft2c with tensor as input/output, used for image to k-space.

    Shape:
        Input: tensor, shape=[Nbs, Ny, Nx, 2], dtype=float
        Output: tensor, shape=[Nbs, Ny, Nx, 2], dtype=float
    """
    x = torch.complex(x[...,0], x[...,1]).numpy()
    x = fft2_np(x)
    x = torch.from_numpy(x)
    x = torch.stack((x.real, x.imag), dim=-1)
    return x

def center_crop(data, shape):
    """
    center crop a tensor (or numpy ndarray) along the last two dimensions.

    Args:
        data: tensor/np.ndarray [..., H, W]
        shape: tuple (h, w)

    Returns:
        center cropped tensor.
    """
    H, W = data.shape[-2], data.shape[-1]
    h, w = shape
    assert 0 < h <= H
    assert 0 < w <= W

    w_from = (W - w) // 2
    h_from = (H - h) // 2
    w_to = w_from + w
    h_to = h_from + h
    data = data[..., h_from:h_to, w_from:w_to]
    return data

def img_rm_black_border(image, thre=1e-10):
    """cut image black border if exists

    Parameters:
        image: np array [H, W], non-negative
        thre: float, relative threshold for zero border

    Output:
        cropped_image: np array [h, w]
        border_idx: list of int, [top, bottom, left, right]
        border_exist: bool
    """
    assert len(image.shape) == 2
    assert np.min(image) >= 0
    assert np.max(image) > 0
    H, W = image.shape
    black_threshold = np.max(image) * thre

    # scan row
    row_means = np.mean(image, axis=1)
    non_black_rows = np.where(row_means > black_threshold)[0]
    top = non_black_rows[0]
    bottom = non_black_rows[-1]

    # scan column
    col_means = np.mean(image, axis=0)
    non_black_cols = np.where(col_means > black_threshold)[0]
    left = non_black_cols[0]
    right = non_black_cols[-1]

    # output
    cropped_image = image[top:bottom+1, left:right+1]
    border_idx = [top, bottom, left, right]

    if (top==0) and (bottom==(H-1)) and (left==0) and (right==(W-1)):
        border_exist = False
    else:
        border_exist = True

    return cropped_image, border_idx, border_exist

def img3_rm_black_border(A, B, C=None):
    """cut image black border if exists, cut B and C based on A

    Parameters:
        A: np array [H, W], non-negative
        B: np array [H, W]
        C: np array [H, W]

    Output:
        A: np array [h, w]
        B: np array [h, w]
        C: np array [h, w]
    """
    assert A.shape == B.shape
    assert len(A.shape) == 2

    if C is not None:
        assert A.shape == C.shape

    _, border_idx, border_exist = img_rm_black_border(A)

    if border_exist:
        top, bottom, left, right = border_idx
        A = A[top:bottom+1, left:right+1]
        B = B[top:bottom+1, left:right+1]
        C = C[top:bottom+1, left:right+1] if C is not None else C

    return A, B, C


