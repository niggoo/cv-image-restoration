import torch

def threshold_func(img, threshold=176):
    '''
    Takes a torch tensor and casts all values to 0 or 255 depending on if they pass the threshold.
    '''
    output = torch.clone(img)
    output[output < threshold] = 0
    output[output >= threshold] = 255
    
    return output