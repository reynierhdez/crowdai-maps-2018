import cv2
import numpy as np
from scipy import ndimage as ndi

import skimage
from skimage.morphology import square
from skimage.morphology import dilation
from skimage.morphology import watershed
from skimage.morphology import binary_erosion
from skimage.feature import peak_local_max
from skimage.io import imread,imread_collection
from skimage.segmentation import find_boundaries
from skimage.filters import sobel
from skimage.measure import label,regionprops
from skimage import exposure
from skimage.morphology import thin
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation
from skimage.morphology import disk
from skimage.feature import peak_local_max

cv2.setNumThreads(0)

def wt_baseline(img = None,
              threshold = 0.5):

    # Make segmentation using edge-detection and watershed.
    edges = sobel(img)

    # Identify some background and foreground pixels from the intensity values.
    # These pixels are used as seeds for watershed.
    markers = np.zeros_like(img)
    foreground, background = 1, 2
    markers[img < threshold * 255 // 2] = background
    markers[img > threshold * 255] = foreground

    ws = watershed(edges, markers)
    labels = label(ws == foreground)
    
    return labels

def label_baseline(msk1 =None,
                   threshold = 0.5):
    
    labels = (np.copy(msk1)>255*threshold)*1
    labels = label(labels)
    
    return labels

def energy_baseline(msk = None,
                    energy = None,
                    threshold = 0.5,
                    thin_labels = False):

    msk_ths = (np.copy(msk)>255*threshold)*1
    energy_ths = (np.copy(energy)>255*0.4)*1

    distance = ndi.distance_transform_edt(msk_ths)
    
    # Marker labelling
    markers = label(energy_ths)    

    labels = watershed(-distance,
                       markers,
                       mask=msk_ths)

    if thin_labels == True:
        for i,lbl in enumerate(np.unique(labels)):
            if i == 0:
                # pass the background
                pass
            else:
                current_label = (labels==lbl) * 1
                thinned_label = thin(current_label,max_iter=1)
                labels[labels==lbl] = 0
                labels[thinned_label] = lbl

        return labels