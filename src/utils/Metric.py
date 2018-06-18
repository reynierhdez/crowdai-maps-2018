import numpy as np

def calculate_ap(y_pred, gt_masks):
    # control for images with only one object
    if len(gt_masks.shape)>2:
        height, width, num_masks = gt_masks.shape[1],gt_masks.shape[2], gt_masks.shape[0]
    else:
        print(gt_masks.shape)
        height, width, num_masks = gt_masks.shape[0],gt_masks.shape[1], 1

    # pred labels and gt masks should have the same dimensions
    assert y_pred.shape[0]==height
    assert y_pred.shape[1]==width
    
    # Make a ground truth label image (pixel value is index of object label)
    # Note that labels will contain the background label
    labels = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labels[gt_masks[index] > 0] = index + 1    
        
    # y_pred should also contain background labels
    # y_pred should contain it if it is taken from wt transform
        
    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred)) 
    
    # Compute intersection between all objects
    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection 
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union   

    # Loop over IoU thresholds
    prec = []
    rec = []
    # print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in [0.5]:
        tp, fp, fn = precision_at(t, iou)
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        # print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
        rec.append(r)
    # print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    
    return prec, np.mean(prec), rec, np.mean(rec)

# Precision helper function
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn    