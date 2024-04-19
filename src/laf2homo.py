import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def laf2homo(kps):
    c = kps[:, :, 2]
    s = torch.sqrt(torch.abs(kps[:, 0, 0] * kps[:, 1, 1] - kps[:, 0, 1] * kps[:, 1, 0]))   
    
    Hi = torch.zeros((kps.shape[0], 3, 3), device=device)
    Hi[:, :2, :] = kps / s.reshape(-1, 1, 1)
    Hi[:, 2, 2] = 1 

    H = torch.linalg.inv(Hi)
    
    return c, H, s
