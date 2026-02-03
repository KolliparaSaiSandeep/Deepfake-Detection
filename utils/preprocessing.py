import torch
import numpy as np
from scipy.fftpack import dct

def get_dct_map(img_tensor):
    # 1. Convert to Grayscale (Luminance)
    gray = 0.299*img_tensor[:,0,:,:] + 0.587*img_tensor[:,1,:,:] + 0.114*img_tensor[:,2,:,:]
    
    # 2. Apply 2D DCT
    def apply_2d_dct(a):
        import scipy.fftpack as fft
        return fft.dct(fft.dct(a.detach().cpu().numpy(), axis=1, norm='ortho'), axis=2, norm='ortho')
    
    # 3. LOG SCALING: This is the secret! 
    # It pulls the hidden "checkerboard" details out of the darkness.
    dct_np = apply_2d_dct(gray)
    dct_log = np.log(np.abs(dct_np) + 1e-6)
    
    # 4. NORMALIZATION: Scale values between 0 and 255 for display
    dct_min, dct_max = dct_log.min(), dct_log.max()
    dct_scaled = 255 * (dct_log - dct_min) / (dct_max - dct_min)
    dct_uint8 = dct_scaled.astype(np.uint8)

    return torch.tensor(dct_log).float().unsqueeze(1), dct_uint8






    