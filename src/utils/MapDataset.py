import gc
import os
import cv2
import time
import glob
import random
import pickle

import collections
import numpy as np
import pandas as pd

from scipy import ndimage
import torch.utils.data as data
from skimage.draw import circle
from skimage.morphology import thin
from skimage.io import imread,imread_collection
from skimage.segmentation import find_boundaries

from skimage.morphology import binary_dilation
from skimage.morphology import disk

from pycocotools.coco import COCO
from pycocotools import mask as cocomask

cv2.setNumThreads(0)

def thin_region_fast(mask,
                    iterations):
    empty = np.zeros_like(mask)
    try:
        min_x, max_x = np.argwhere(mask > 0)[:,0].min(),np.argwhere(mask > 0)[:,0].max()
        min_y, max_y = np.argwhere(mask > 0)[:,1].min(),np.argwhere(mask > 0)[:,1].max()
    
        empty[min_x:max_x,min_y:max_y] = thin(mask[min_x:max_x,min_y:max_y],max_iter=iterations)
        return empty
    except:
        return empty
    
def distance_transform_fast(mask,
                           return_indices=False):
    
    min_x, max_x = np.argwhere(mask > 0)[:,0].min(),np.argwhere(mask > 0)[:,0].max()
    min_y, max_y = np.argwhere(mask > 0)[:,1].min(),np.argwhere(mask > 0)[:,1].max()
    
    if return_indices == False:
        empty = np.zeros_like(mask)
        try:
            empty[min_x:max_x,min_y:max_y] = ndimage.distance_transform_edt(mask[min_x:max_x,min_y:max_y])
            return empty
        except:
            return empty
    else:
        min_x = max(min_x-5,0)
        min_y = max(min_y-5,0)
        max_x = max_x+5
        max_y = max_y+5
        
        empty = np.zeros_like(mask)
        indices = np.zeros_like(np.vstack([[mask]*2])) 
        
        try:
            empty[min_x:max_x,min_y:max_y],indices[:,min_x:max_x,min_y:max_y] = ndimage.distance_transform_edt(mask[min_x:max_x,min_y:max_y],return_indices=True)
            indices[0] = indices[0] + min_x
            indices[1] = indices[1] + min_y
            return empty,indices
        except:
            return empty,indices        
    
def mask2vectors(mask):
    distances, indices = distance_transform_fast(mask,return_indices=True)
    # avoid division by zero for blank areas  when normalizing
    grid_indices = np.indices((mask.shape[0],mask.shape[1]))
    distances[distances==0]=1
    return (indices*(mask>255//2) - grid_indices*(mask>255//2)) / np.asarray([distances,distances])

class MapDataset(data.Dataset):
    def __init__(self,
                 transforms = None,
                 mode = 'train', # 'train', 'val' or 'test'
                 target_resl = [299,299],
                 
                 do_energy_levels = True,
                 energy_levels = [1,5,9],
                 do_boundaries = False,
                 do_size_jitter = False,
                 
                 train_images_directory = "../data/train/images",
                 train_annotations_path = "../data/train/annotation.json",
                 pickled_train_annotations = "pickled_train_annotations",
                 
                 val_images_directory = "../data/val/images",
                 val_annotations_path = "../data/val/annotation.json",
                 pickled_val_annotations = "pickled_val_annotations",
                 
                 test_images_directory = "../data/test_images/"
                ):

        TRAIN_ANNOTATIONS_PATH = train_annotations_path
        PICKLED_TRAIN_ANNOTATIONS = pickled_train_annotations

        VAL_ANNOTATIONS_PATH = val_annotations_path
        PICKLED_VAL_ANNOTATIONS = pickled_val_annotations

        TEST_IMAGES_DIRECTORY = test_images_directory
        
        self.transforms = transforms
        self.mode = mode
        self.target_resl = target_resl
        self.energy_levels = energy_levels
        self.do_energy_levels = do_energy_levels
        self.do_boundaries = do_boundaries
        self.do_size_jitter = do_size_jitter   
        self.do_energy_levels = do_energy_levels
                
        if self.mode in ['train']:
            if os.path.isfile(PICKLED_TRAIN_ANNOTATIONS):
                with open(PICKLED_TRAIN_ANNOTATIONS, 'rb') as handle:
                    gc.disable()
                    coco_train = pickle.load(handle)
                    print('Train annotations loaded from pickle')
                    gc.enable()
            else:
                coco_train = COCO(TRAIN_ANNOTATIONS_PATH)
                print('Train annotations loaded from json')
                with open(PICKLED_TRAIN_ANNOTATIONS, 'wb') as handle:
                    pickle.dump(coco_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    print('Train annotations pickled')   

            self.coco = coco_train
            self.train_img_ids = coco_train.getImgIds()
            self.img_directory = train_images_directory
        
        elif self.mode in ['val']:   
            if os.path.isfile(PICKLED_VAL_ANNOTATIONS):
                with open(PICKLED_VAL_ANNOTATIONS, 'rb') as handle:
                    gc.disable()
                    coco_val = pickle.load(handle)
                    print('Val annotations loaded from pickle') 
                    gc.enable()
            else:
                coco_val = COCO(VAL_ANNOTATIONS_PATH)
                print('Val annotations loaded from json')    
                with open(PICKLED_VAL_ANNOTATIONS, 'wb') as handle:
                    pickle.dump(coco_val, handle, protocol=pickle.HIGHEST_PROTOCOL)        
                    print('Val annotations pickled')            
                
            self.coco = coco_val
            self.val_img_ids = coco_val.getImgIds()
            self.img_directory = val_images_directory  
                
        elif self.mode in ['test']: 
            self.test_paths = sorted(glob.glob('../data/test_images/*.jpg'))
            self.test_img_ids = [int(_.split('/')[-1].split('.')[0]) for _ in self.test_paths]
            
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_img_ids)
        elif self.mode == 'val':
            return len(self.val_img_ids)          
        elif self.mode == 'test':  
            return len(self.test_paths) 

    def get_img_masks(self, img_id):
        img_meta = self.coco.loadImgs(int(img_id))[0]
        image_path = os.path.join(self.img_directory, img_meta["file_name"])
        annotation_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(annotation_ids)

        img = imread(image_path)
        masks = []         

        for _, annotation in enumerate(annotations):
            rle = cocomask.frPyObjects(annotation['segmentation'], img_meta['height'], img_meta['width'])
            m = cocomask.decode(rle)
            # m.shape has a shape of (300, 300, 1)
            # so we first convert it to a shape of (300, 300)
            m = m.reshape((img_meta['height'], img_meta['width']))
            masks.append(m)
        
        return img,masks
        
    def __getitem__(self, idx):
        
        if self.mode == 'train':
            img_id = self.train_img_ids[idx]
            img,masks = self.get_img_masks(img_id)
        elif self.mode == 'val':
            img_id = self.self.val_img_ids[idx]
            img,masks = self.get_img_masks(img_id)
        
        if self.mode in ['train','val']:
            masks = [mask.astype('uint8') for mask in masks]
            mask = np.sum(np.stack(masks, 0), 0)

            if self.do_energy_levels:
                masks_thin = []
                for energy_level in self.energy_levels:
                    masks_thin.append(np.asarray([(thin_region_fast(_,energy_level)) for _ in masks]))
            
            if self.do_boundaries == True:
                gt_labels = np.zeros((mask.shape[0], mask.shape[1]), np.uint16)
                for index in range(0, len(masks)):
                    gt_labels[masks[index] > 0] = index + 1
                boundaries = find_boundaries(gt_labels, connectivity=1, mode='thick', background=0)
                boundaries = (boundaries * 1).astype('uint8')
            
            # do not forget to binarize the masks
            if self.do_energy_levels:    
                masks_thin = [np.sum(np.stack(_, 0), 0).astype('uint8') for _ in masks_thin]

            mask = mask.astype('uint8')

            lst = []
            lst.append(mask)

            if self.do_energy_levels:
                lst.extend(masks_thin)

            if self.do_boundaries:                
                lst.extend([boundaries])     

            # for _ in lst:
            #    print(_.shape)
                
            msk = np.stack(lst,axis=0)
            img, msk = self.transforms(img, msk.transpose((1,2,0)), self.target_resl)

            # conform to the PyTorch tensor format
            img = img.transpose((2,0,1))
            
            if len(msk.shape)>2:
                msk = msk.transpose((2,0,1))
            else:
                msk = msk[np.newaxis,:,:]
                
            return img,msk,img_id

        elif self.mode == 'test':
            img = imread(self.test_paths[idx])
            img_id = self.test_img_ids[idx]
            
            img, _ = self.transforms(img, None, self.target_resl)
            
            # conform to the PyTorch tensor format
            img = img.transpose((2,0,1))            

            # return img id as well as path for easier submissions
            return img,img_id,self.test_paths[idx]