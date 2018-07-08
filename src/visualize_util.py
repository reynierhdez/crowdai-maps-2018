#==========Import Section==========
import numpy as np
from skimage.measure import label
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import itertools

#==========Helpers===============
def label_baseline(msk1 =None,
                   threshold = 0.5):
    """Transform and label mask so that objects are separated
    
    Keyword arguments:
    msk1 -- mask
    threshold -- labelling threshold (default 0.5)    
    """
    
    labels = (np.copy(msk1)>255*threshold)*1
    labels = label(labels)
    
    return labels

def binarize_mask(mask):
    '''Binarize mask
    
    Keyword arguments:
    mask -- any mask to binarize
    '''
    
    return (mask != 0) * 1

def filter_mask(mask, filter_by = 0):
    '''Filter mask so that everything but specified number is zero
    
    Keyword arguments:
    mask -- any mask to filter
    filter_by -- specified number to leave after filtering (default 0)
    '''
    return (mask == filter_by) * 1

def compute_label_center(mask, label = 0):
    '''Compute lebel center in order to mark it on a visualization
    
    Keyword arguments:
    mask -- some mask
    label -- some label for filtering (default 0)
    '''
    
    itemindex = np.where(mask == label)
    centerx = int(np.round((np.max(itemindex[1]) + np.min(itemindex[1])) / 2, 0))
    centery = int(np.round((np.max(itemindex[0]) + np.min(itemindex[0])) / 2, 0))
    return centerx, centery

#==========Metrics===============
def metrics_single_pair(gtmask, genmask, lbl_treshold = 0.5):
    '''Compute Intersection Over Union and return set of true positives, false positive and false negatives 
    for a pair of ground truth mask and generated mask
    
    Keyword arguments:
    gtmask -- ground truth mask
    genmask -- generated mask
    lbl_threshold -- labeling threshold for mask (default 0.5)
    '''
    gtmask_lbl = label_baseline(gtmask, lbl_treshold)
    genmask_lbl = label_baseline(genmask, lbl_treshold)
    
    max_lbl_gt = np.max(gtmask_lbl)
    max_lbl_gen = np.max(genmask_lbl)
    
    combs = itertools.product(np.arange(0, max_lbl_gt, 1), np.arange(0, max_lbl_gen, 1))
    
    ious = []
    dious = {}
    TPs = 0
    FPs = 0
    FNs = 0
    
    for a in np.arange(1, max_lbl_gt + 1, 1):
        iou_a = 0
        for b in np.arange(1, max_lbl_gen + 1, 1):
            gt_filtered = binarize_mask(filter_mask(gtmask_lbl, a))
            gen_filtered = binarize_mask(filter_mask(genmask_lbl, b))

            union = np.sum(binarize_mask(gt_filtered + gen_filtered))
            intersect = np.sum(gt_filtered * gen_filtered)

            iou = 0
            if union > 0:
                iou = intersect / union
                
            if iou > iou_a: 
                iou_a = iou
                
        ious.append(iou_a)
        dious[a] = iou_a
    TPs = np.sum([x >= 0.5 for x in ious])
    FNs = np.sum([x < 0.5 for x in ious])
    if max_lbl_gen > max_lbl_gt:
        FPs = max_lbl_gen - max_lbl_gt # false positives
    return TPs, FPs, FNs, dious

def compute_precision_recall(TPs, FPs, FNs):
    '''Compute Precision-Recall for a set of true positives, false positives and false negatives for a single pair of masks
    
    Keyword arguments:
    TPs -- number of true positives
    FPs -- number of false positives
    FNs -- number of false negatives
    '''
    if (TPs + FPs + FNs) == 0:
        precision = 1
        recall = 1
    elif (TPs + FPs) == 0:
        recall = 0
        precision = 0
    elif (TPs + FNs) == 0:
        recall = 0
        precision = 0
    else:
        precision = TPs/(TPs + FPs)
        recall = TPs/(TPs + FNs)
    
    return precision, recall

#==========Visualizer=============
def mix_vis_masks(gtmask, 
                  genmask, 
                  orig, 
                  lbl_treshold = 0.5, 
                  save_path = '', 
                  save_title = 'example.png', 
                  do_vis = True,
                  do_save = True):
    '''Visualize and save ground truth and generated masks 
    mixed together to compare alongside with original image.
    
    Keyword arguments:
    gtmask -- ground truth mask
    genmask -- generated mask
    orig -- original image
    lbl_threshold -- labeling threshold for mask (default 0.5)
    save_path -- path to save (default is cwd)
    save_title -- title for saved image (default "example.png")
    do_vis -- whether to visualize results or not
    do_save -- whether to save image or not
    '''
    
    TPs, FPs, FNs, dious = metrics_single_pair(gtmask, genmask, lbl_treshold)
    
    precision, recall = compute_precision_recall(TPs, FPs, FNs)
    
    if do_vis:
        gtmask_lbl = label_baseline(gtmask, lbl_treshold)
        genmask_lbl = label_baseline(genmask, lbl_treshold)
    
        f, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(gtmask_lbl, alpha = 0.5)
        ax[0].imshow(genmask_lbl, alpha = 0.5)

        for key, value in dious.items():
            centerx, centery = compute_label_center(gtmask_lbl, key)
            ax[0].text(centerx, centery, '{:.2f}'.format(value), weight = 'bold', color = 'black', size = 14)

        ax[1].imshow(orig)
        ax[0].set_title('Pr: {:.2f}, Re: {:.2f}, TPs: {:.2f}, FPs: {:.2f}, FNs: {:.2f}'.format(precision, recall, TPs, FPs, FNs))
        ax[1].set_title('Original Img')
        ax[0].axis('off')
        ax[1].axis('off')
        plt.savefig(os.path.join(save_path, save_title))
    
    return precision, recall, TPs, FPs, FNs