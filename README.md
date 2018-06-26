# crowdai-maps-2018
CrowdAI mapping challenge 2018 solution

So far just the command for training the best model
First
```
cd src
```
Then
```
# 3 GPU training, ~10 images per GPU
python3 train_weighted.py \
	--arch linknet152 --img_size 288 --lr 1e-3 \
	--batch-size 30 --workers 8 \
	--epochs 100 --start-epoch 0 --epoch_fraction 0.1 \
	--m0 5 --m1 5 --m2 20 \
	--ths 0.5 --tensorboard True  --tensorboard_images True \
	--do_augs True --do_more_augs False --aug_prob 0.1 \
	--do_energy_levels False --do_boundaries False \
	--optimizer adam \
	--do_running_mean False --bce_weight 0.5 --dice_weight 0.5 \
	--print-freq 20 \
	--lognumber ln152_wboth_small_augs_boost_dice \
	--do_remove_small_on_borders True \
	--do_produce_sizes_mask True --do_produce_distances_mask True \
	--w0 5.0 --sigma 10.0 \
  ```
  
  Also make sure that your `data` directory looks like this
  ```
  data/
├── test_images
├── train
│   └── images
└── val
    └── images
    ```
