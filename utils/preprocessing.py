import torch
import numpy as np
from scipy.fftpack import dct

def get_dct_map(img_tensor):
    # Convert to grayscale
    gray = 0.299*img_tensor[:,0,:,:] + 0.587*img_tensor[:,1,:,:] + 0.114*img_tensor[:,2,:,:]
    def apply_dct(a):
        return dct(dct(a.cpu().numpy(), axis=1, norm='ortho'), axis=2, norm='ortho')
    
    res = np.log(np.abs(apply_dct(gray)) + 1e-6)
    return torch.tensor(res).float().unsqueeze(1)




    