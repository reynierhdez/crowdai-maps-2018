import cv2
import os
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
from skimage.color import gray2rgb
from pipeline2d3d.src.util.bad_series_uid import messed_up_uids
cv2.setNumThreads(0)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

def to_np(x):
    x = x.cpu().numpy()
    if len(x.shape)>3:
        return x[:,0:3,:,:]
    else:
        return x

def plot_projections(img=None,
                     clahe=0):
    fr,up,le,ri = project_dicom(img)
    fr,up,le,ri = map(lambda x: min_max(x),[fr,up,le,ri])
    
    if clahe>0:
        fr,up,le,ri = map(lambda x: equalize_adapthist(x,clip_limit=clahe),[fr,up,le,ri])
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True, sharey=True)
    ax = axes.ravel()    
    
    ax[0].imshow(fr, cmap=plt.cm.gray, interpolation='nearest')
    ax[0].set_title('Frontal')
    ax[0].axis('off')

    ax[1].imshow(up, cmap=plt.cm.gray, interpolation='nearest')
    ax[1].set_title('From above')
    ax[1].axis('off')

    ax[2].imshow(le, cmap=plt.cm.gray, interpolation='nearest')
    ax[2].set_title('Left')
    ax[2].axis('off')

    ax[3].imshow(ri, cmap=plt.cm.gray, interpolation='nearest')
    ax[3].set_title('Right')
    ax[3].axis('off')

    plt.show()    
    
def project_dicom(img):
    fr = img.sum(axis=1)
    up = img.sum(axis=0)
    le = img[:,:,:img.shape[2]//2].sum(axis=2)
    ri = img[:,:,img.shape[2]//2:].sum(axis=2)
    fr,up,le,ri = map(lambda x: min_max(x),[fr,up,le,ri])

    return fr,up,le,ri    

def min_max(img):
    img = img.astype('float64')
    img += -img.min()
    if img.max()>0:
        img *= (1.0/img.max())
    return img

def min_max_int(img=None,d_min=None,d_max=None):
    # print('New img')
    # print(img.min(),img.max())
    
    img_max = img.max()
    
    img = img.astype('float64')
    # support whole dicom normalization
    if d_min:
        img += -d_min
    else:
        img += -img.min()
    
    # print(img.min(),img.max())    
    
    if d_max:
        if img_max>0:
            img *= (1.0/(d_max-d_min)) * 255
            # print(img.min(),img.max())
    else:
        if img_max>0:
            img *= (1.0/img.max()) * 255
    img = img.astype('uint8')
    return img 

def binarize_mask(mask):
    return ((mask>0)*255).astype('uint8')

def uid2path(series_uid):
    return os.path.join(os.getenv('NUMPY_CACHE_FOLDER'),series_uid)+'.npy'

def get_max_tooth_height(bboxes):
    teeth = []
    
    for bbox in bboxes:
        teeth.append(bbox.tooth_number)
    
    teeth = list(set(teeth))
    
    teeth_sizes = []
    for tooth in teeth:
        tooth_bboxes = list(filter(lambda x: x.tooth_number == tooth, bboxes))
        tooth_slices = list(map(lambda x: x.slice_number, tooth_bboxes))
        teeth_sizes.append(max(tooth_slices)-min(tooth_slices))
    return max(teeth_sizes)

def reduce_dicom(dicom,slices):
    dicom = np.copy(dicom[np.asarray(slices),:,:])
    new_idx = list(range(0,len(slices)))
    return dicom, new_idx

def load_meta_data_df(g):
    # refactor here
    cached_series_uids = [_.split('/')[-1].split('.npy')[0] for _ in g]

    # read cached meta-data
    df_meta = pd.read_feather('../utils/2018_04_19_meta_feather')
    df_meta['is_cached'] = df_meta.series_uid.apply(lambda x: True if x in cached_series_uids else False)
    # drop extra columns
    cols = ['avg_slices_per_tooth', 'tooth_count', 'series_uid', 'birth_date','img_rows', 'img_cols','age','is_cached','manufacturer']
    df_meta = df_meta[cols]
    le, u = df_meta['manufacturer'].factorize()
    df_meta['manufacturer'] = le
    
    # remove bad series uid
    
    print('Total studies BEFORE removing broken studies {}'.format(df_meta.shape[0]))
    df_meta = df_meta[~df_meta.series_uid.isin(messed_up_uids)]
    print('Total studies AFTER removing broken studies {}'.format(df_meta.shape[0]))    
    df_meta = df_meta.reset_index()
    
    return df_meta
    
def red_only(img):
    img[:,:,1:] = 0
    return img    

def green_only(img):
    img[:,:,0] = 0
    img[:,:,2] = 0
    return img

def project_dicom_for_tb(img,msk):
    # reduce additional dimensions (i.e. context or energy levels) if any
    # treat imgs as if they are plain 3D DICOMs
    if len(img.shape)>3:
        img = img.sum(axis=1)
        msk = msk.sum(axis=1)

    fr = img.sum(axis=1)
    up = img.sum(axis=0)
    le = img[:,:,:img.shape[2]//2].sum(axis=2)
    ri = img[:,:,img.shape[2]//2:].sum(axis=2)
    
    fr_msk = msk.sum(axis=1)
    up_msk = msk.sum(axis=0)
    le_msk = msk[:,:,:msk.shape[2]//2].sum(axis=2)
    ri_msk = msk[:,:,msk.shape[2]//2:].sum(axis=2)    
    
    fr,up,le,ri = map(lambda x: min_max(x),[fr,up,le,ri])
    fr_msk,up_msk,le_msk,ri_msk = map(lambda x: min_max(x),[fr_msk,up_msk,le_msk,ri_msk])    
    
    # convert to RGB gray images
    # mask mask as red
    fr,up,le,ri = map(lambda x: gray2rgb(x),[fr,up,le,ri])
    fr_msk,up_msk,le_msk,ri_msk = map(lambda x: red_only(gray2rgb(x)),[fr_msk,up_msk,le_msk,ri_msk])
    
    fr = fr + fr_msk
    up = up + up_msk
    le = le + le_msk
    ri = ri + ri_msk
    
    fr,up,le,ri = map(lambda x: min_max(x),[fr,up,le,ri])    
    
    return fr,up,le,ri    

def project_dicom_for_tb_panno(img,msk,gt_msk):
    
    num_slices = img.shape[0]
    
    # reduce additional dimensions (i.e. context or energy levels) if any
    # treat imgs as if they are plain 3D DICOMs
    if len(img.shape)>3:
        img = img.sum(axis=1)
        msk = msk.sum(axis=1)
        if gt_msk:
            gt_msk = gt_msk.sum(axis=1)

    fr = img.sum(axis=1)
    up = img.sum(axis=0)
    le = img[:,:,:img.shape[2]//2].sum(axis=2)
    ri = img[:,:,img.shape[2]//2:].sum(axis=2)
    
    fr_msk = msk.sum(axis=1)
    up_msk = msk.sum(axis=0)
    le_msk = msk[:,:,:msk.shape[2]//2].sum(axis=2)
    ri_msk = msk[:,:,msk.shape[2]//2:].sum(axis=2)    
    
    # check that gt mask is not null
    if gt_msk:
        fr_gt_msk = gt_msk.sum(axis=1)
        up_gt_msk = gt_msk.sum(axis=0)
        le_gt_msk = gt_msk[:,:,:gt_msk.shape[2]//2].sum(axis=2)
        ri_gt_msk = gt_msk[:,:,gt_msk.shape[2]//2:].sum(axis=2)
    else:
        fr_gt_msk = np.zeros_like(fr_msk)
        up_gt_msk = np.zeros_like(up_msk)
        le_gt_msk = np.zeros_like(le_msk)
        ri_gt_msk = np.zeros_like(ri_msk)
        
    fr,up,le,ri = map(lambda x: min_max(x),[fr,up,le,ri])
    fr_msk,up_msk,le_msk,ri_msk = map(lambda x: min_max(x),[fr_msk,up_msk,le_msk,ri_msk])
    fr_gt_msk,up_gt_msk,le_gt_msk,ri_gt_msk = map(lambda x: min_max(x),[fr_gt_msk,up_gt_msk,le_gt_msk,ri_gt_msk])       
    
    # convert to RGB gray images
    # mask mask as red
    fr,up,le,ri = map(lambda x: gray2rgb(x),[fr,up,le,ri])
    fr_msk,up_msk,le_msk,ri_msk = map(lambda x: red_only(gray2rgb(x)),[fr_msk,up_msk,le_msk,ri_msk])
    fr_gt_msk,up_gt_msk,le_gt_msk,ri_gt_msk = map(lambda x: green_only(gray2rgb(x)),[fr_gt_msk,up_gt_msk,le_gt_msk,ri_gt_msk])

    fr = fr + fr_msk + fr_gt_msk
    up = up + up_msk + up_gt_msk
    le = le + le_msk + le_gt_msk
    ri = ri + ri_msk + ri_gt_msk
    
    tile_size = max(num_slices,512)

    # pad imgs to the same sizes
    fr = cv2.copyMakeBorder(fr,0,tile_size-fr.shape[0],0,tile_size-fr.shape[1],cv2.BORDER_CONSTANT,value=0) 
    up = cv2.copyMakeBorder(up,0,tile_size-up.shape[0],0,tile_size-up.shape[1],cv2.BORDER_CONSTANT,value=0) 
    le = cv2.copyMakeBorder(le,0,tile_size-le.shape[0],0,tile_size-le.shape[1],cv2.BORDER_CONSTANT,value=0) 
    ri = cv2.copyMakeBorder(ri,0,tile_size-ri.shape[0],0,tile_size-ri.shape[1],cv2.BORDER_CONSTANT,value=0)     
    
    fr,up,le,ri = map(lambda x: min_max(x),[fr,up,le,ri])
    
    panno = np.hstack((fr,up,le,ri))
    
    return panno  

class JawCutter(object):
    def __init__(self,
                 window_size=10,
                 threshold=0.5,
                 max_gap=2,
                 debug=False,
                 add_frame=5):
        
        self.window_size = window_size
        self.threshold = threshold
        self.max_gap = max_gap
        self.debug = debug
        self.add_frame = add_frame
        
    def __call__(self,
                 pred):

        pred_ths = np.copy(pred)
        pred_ths[pred_ths<self.threshold]=0
        pred_ths[pred_ths>self.threshold]=1
        pred_ths = pred_ths.astype('uint8')

        # reduce to 3 dimensions
        if len(pred.shape)>3:
            pred_ths = pred_ths.sum(axis=1)
            
        z_min_max = self.process_projection(pred_ths,axes=(1,2),dimension=0)
        x_min_max = self.process_projection(pred_ths,axes=(0,2),dimension=1)
        y_min_max = self.process_projection(pred_ths,axes=(0,1),dimension=2)
        
        return z_min_max,x_min_max,y_min_max

    def process_projection(self,
                           pred_ths=None,
                           axes=(1,2),
                           dimension=0):
        
        if self.debug:
            print('Processing dimension {}, reduction axes {}'.format(dimension,axes))
            
        # project values to one dimension
        pred_ths_projection = pred_ths.sum(axis=axes)

        axis_projections_window = []
        batches = []

        # create a histogram 
        for batch in chunker(range(0,pred_ths.shape[dimension]), self.window_size):
            batches.append(list(batch))            
            axis_projections_window.append(pred_ths_projection[batch].sum())    

        sequences = []
        current_sequence = []
        last_value = 0

        for i,value in enumerate(axis_projections_window):
            # finish sequence if it ended
            if last_value>0 and value==0 and current_sequence:
                # print('Finishing sequence on {}'.format(i))
                sequences.append(current_sequence)
                current_sequence = []
            # add to sequence if it continues or just started
            elif value>0:
                # print('Adding to sequence on {}'.format(i))        
                current_sequence.append(i)

            last_value = value

            
        sequences_lengths = [len(_) for _ in sequences]
        sequences_energy = [np.asarray(axis_projections_window)[_].sum() for _ in sequences]
        
        if self.debug:
            print('Sequences list {}'.format(sequences))
            print('Sequences lengths {}'.format(sequences_lengths))
            print('Sequences energy {}'.format(sequences_energy))                   
        
        # naive greedy algorithm - add +/- 1 sequence on both sides
        largest_sequence = sequences_energy.index(max(sequences_energy))

        # try adding +/- 1 sequence
        # if there is only one sequence - do not merge anything
        if len(sequences)>1:
            try:

                prev_sequence = largest_sequence - 1

                # if the biggest sequence is the first one - do nothing
                if prev_sequence<0:
                    merge_previous = False
                else:
                    distance = min(sequences[largest_sequence]) - max(sequences[prev_sequence])

                    if distance<(self.max_gap+1):
                        merge_previous = True
                    else:
                        merge_previous = False
            except:
                merge_previous = False        

            try:
                # if the largest_sequence is the last one - exception will be thrown
                next_sequence = largest_sequence + 1
                distance = min(sequences[next_sequence]) - max(sequences[largest_sequence])

                if distance<(max_window_len+1):
                    merge_next = True
                else:
                    merge_next = False

            except:
                merge_next = False
        else:
            merge_previous = False
            merge_next = False

        final_sequence = sequences[largest_sequence]
        
        if self.debug:
            print('Merge prev {} / merge next {}'.format(merge_previous,merge_next))
        
        if merge_previous:
            final_sequence.extend(sequences[largest_sequence-1])

        if merge_next:
            final_sequence.extend(sequences[largest_sequence+1])

        if self.debug:            
            print(final_sequence)
        min_dimension = min(batches[min(final_sequence)])
        max_dimension = max(batches[max(final_sequence)])
        
        if self.add_frame>0:
            min_dimension = min_dimension - self.add_frame
            max_dimension = max_dimension + self.add_frame
        
        return (min_dimension,max_dimension)

def max_box(img_3d):
    # reduce to 3 dimensions if necessary
    if len(img_3d.shape)>3:
        img_3d = img_3d.sum(axis=1)
    
    non_zero_idx = np.where(img_3d>0)
    
    min_x = min(non_zero_idx[0])
    max_x = max(non_zero_idx[0])
    
    min_y = min(non_zero_idx[1])
    max_y = max(non_zero_idx[1])
    
    min_z = min(non_zero_idx[2])
    max_z = max(non_zero_idx[2])
    
    return ((min_x,max_x),(min_y,max_y),(min_z,max_z))

def chunker(seq, size):
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))    

def bb_tuple2dict_3d(bbox):
    bbox_dict = {
        'z1': bbox[0][0],
        'z2': bbox[0][1],       
        'x1': bbox[1][0],
        'x2': bbox[1][1],
        'y1': bbox[2][0],
        'y2': bbox[2][1]
    }
    return bbox_dict

def bb_tuple2dict_2d(bbox):
    bbox_dict = {
        'x1': bbox[0][0],
        'x2': bbox[0][1],
        'y1': bbox[1][0],
        'y2': bbox[1][1]
    }
    return bbox_dict

def iou_3d(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2', 'z1','z2'}
        The (z1, x1, y1) position is at the top left corner,
        the (z2, x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2', 'z1','z2'}
        The (z1, x1, y2) position is at the top left corner,
        the (z2, x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb1['z1'] < bb1['z2'] 
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    assert bb2['z1'] < bb2['z2']     

    # determine the coordinates of the intersection rectangle
    z_up = max(bb1['z1'], bb2['z1'])
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])

    z_down = min(bb1['z2'], bb2['z2'])                               
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top or z_down < z_up:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top) * (z_down - z_up)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1']) * (bb1['z2'] - bb1['z1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1']) * (bb2['z2'] - bb2['z1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def iou_2d(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def bbox_inside_3d(bb1,bb2):
    """
    Calculate whether bb1 is contained within bb2

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2', 'z1','z2'}
        The (z1, x1, y1) position is at the top left corner,
        the (z2, x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2', 'z1','z2'}
        The (z1, x1, y2) position is at the top left corner,
        the (z2, x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb1['z1'] < bb1['z2'] 
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    assert bb2['z1'] < bb2['z2']   
    
    if (bb2['x1']<bb1['x1']\
        and bb2['x2']>bb1['x2']\
        and bb2['y1']<bb1['y1']\
        and bb2['y2']>bb1['y2']\
        and bb2['z1']<bb1['z1']\
        and bb2['z2']>bb1['z2']):
        return 1
    else:
        return 0

def bbox_inside_2d(bb1,bb2):
    """
    Calculate whether bb1 is contained within bb2

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y2) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    
    if (bb2['x1']<bb1['x1']\
        and bb2['x2']>bb1['x2']\
        and bb2['y1']<bb1['y1']\
        and bb2['y2']>bb1['y2']):
        return 1
    else:
        return 0    
