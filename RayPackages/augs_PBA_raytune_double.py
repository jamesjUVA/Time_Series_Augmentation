### Augmentations imported from https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py


import numpy as np
from tqdm import tqdm
from scipy.interpolate import CubicSpline



def jitter(x, mag=.5):
    #range for PBA .005:.055
    aug_range=(.05,.505)
    sigma=aug_range[0]+(aug_range[1]-aug_range[0])*mag
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)



#def scaling(x, sigma=0.1):
#    # https://arxiv.org/pdf/1706.00527.pdf
#    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0],x.shape[2]))
#    return np.multiply(x, factor[:,np.newaxis,:])

def scaling(x, mag=.5):
    aug_range=(.05,1.05)
    sigma=aug_range[0]+(aug_range[1]-aug_range[0])*mag
    #range for PBA .05:.15
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma)
    return x*factor

#def rotation(x):
#    flip = np.random.choice([-1, 1], size=(x.shape[0],x.shape[2]))
#    rotate_axis = np.arange(x.shape[2])
#    np.random.shuffle(rotate_axis)    
#    return flip[:,np.newaxis,:] * x[:,:,rotate_axis]

def rotation(x,mag=0):
    ## Constant for PBA
    return x[::-1] 


def permutation(x, mag=.5):
    ## Range for PBA 2:6
    
    max_segments=np.round(mag*4)+2

    orig_steps = np.arange(x.shape[0])
    
    num_segs = np.random.randint(1, max_segments+1)
    
    ret = np.zeros_like(x)
    
    if np.random.random()>.5:
        seg_mode="random"
    else:
        seg_mode='equal'
    
    
    if num_segs > 1:
        if seg_mode == "random":
            split_points = np.random.choice(x.shape[0]-2, num_segs-1, replace=False)
            split_points.sort()
            splits = np.split(orig_steps, split_points)
        else:
            splits = np.array_split(orig_steps, num_segs)
        warp = np.concatenate(np.random.permutation(splits)).ravel()
        ret = x[warp]
    else:
        ret = x
    return ret



def time_warp(x, mag=.5):
    ## Range for PBA .15:.35
    aug_range=(.05,.15)
    sigma=aug_range[0]+(aug_range[1]-aug_range[0])*mag
    orig_steps = np.arange(x.shape[0])
    
    knot=np.random.randint(1,4)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2))
    warp_steps = (np.ones((1))*(np.linspace(0, x.shape[0]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    
    
    time_warp = CubicSpline(warp_steps[:], warp_steps[:] * random_warps[:])(orig_steps)
    scale = (x.shape[0]-1)/time_warp[-1]
    ret[:] = np.interp(orig_steps, np.clip(scale*time_warp, 0, x.shape[0]-1), x[:]).T
    
    return ret


def window_slice(x, mag=.5):
    ## changed to make random UP to mah with min of .05
    ## Range for PBA [.05:.15]
    aug_range=(.025,.1525)
    reduce_ratio=aug_range[0]+(aug_range[1]-aug_range[0])*mag
    reduce_ratio = np.random.uniform(aug_range[0],reduce_ratio)
    reduce_ratio=1-reduce_ratio  
    # this stretches back to original size and thus becomes like a time_warp
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[0]).astype(int)
    if target_len >= x.shape[0]:
        return x
    starts = np.random.randint(low=0, high=x.shape[0]-target_len, size=(1)).astype(int)
    ends = (target_len + starts).astype(int)
    
    ret = np.zeros_like(x)
    ret = np.interp(np.linspace(0, target_len, num=x.shape[0]), np.arange(target_len), x[starts[0]:ends[0]]).T
    return ret

def crop(x, mag=.5):
    ## Range for PBA [.05:.15]
    aug_range=(.025,.1525)
    reduce_ratio=aug_range[0]+(aug_range[1]-aug_range[0])*mag
    reduce_ration = np.random.uniform(.05,reduce_ratio)
    reduce_ratio=1-reduce_ratio  
    target_len = np.ceil(reduce_ratio*x.shape[0]).astype(int)
    if target_len >= x.shape[0]:
        return x
    ret = np.zeros_like(x)
    if np.random.random()>.5: #flip LR to shift left OR right.
        starts = np.random.randint(low=0, high=x.shape[0]-target_len, size=(1)).astype(int)
        ret[0:(ret.shape[0]-int(starts))]=x[starts[0]:x.shape[0]]
    else:    
        ends = np.random.randint(low=target_len, high=x.shape[0], size=(1)).astype(int)
        ret[ret.shape[0]-int(ends):]=x[0:ends[0]]
    return ret


def magnitude_warp(x, mag=.5):
    ## Range for PBA [.1:.3]
    knot = np.random.randint(1,4)
    
    aug_range=(.1,.4)
    sigma=aug_range[0]+(aug_range[1]-aug_range[0])*mag
    
    orig_steps = np.arange(x.shape[0])
    
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(knot+2))
    warp_steps = (np.ones((1))*(np.linspace(0, x.shape[0]-1., num=knot+2))).T
    
    ret = np.zeros_like(x)
    
    warper = np.array([CubicSpline(warp_steps[:], random_warps[:])(orig_steps)]).T
    ret = x * warper[:,0]

    return ret


def window_warp(x, mag=.5):
    ## Range for PBA [.1:.5]

    aug_range=(.2,.5)
    scale=aug_range[0]+(aug_range[1]-aug_range[0])*mag
    
    window_ratio=np.random.uniform(low=.05,high=.25)
    #modified to range of .1 - .5 yeilding max intensity of [.5,2] and min of [.9,1.1111]
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    low=1-scale
    high=1/(1-scale)
    scales=[low,high]
    warp_scale = np.random.choice(scales)
    warp_size = np.ceil(window_ratio*x.shape[0]).astype(int)
    window_steps = np.arange(warp_size)
        
    window_start = np.random.randint(low=1, high=x.shape[0]-warp_size-1) #.astype(int)
    window_end = (window_start + warp_size).astype(int)
            
    ret = np.zeros_like(x)
   
    start_seg = x[:window_start]
    window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scale)), window_steps, x[window_start:window_end])
    end_seg = x[window_end:]
    warped = np.concatenate((start_seg, window_seg, end_seg))                
    ret = np.interp(np.arange(x.shape[0]), np.linspace(0, x.shape[0]-1, num=warped.size), warped).T
    return ret
