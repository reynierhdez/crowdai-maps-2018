# cd /home/keras/notebook/nvme/mapsai/repo/crowdai-maps-2018/src
python3 train_weighted_val2.py \
    --resume weights/ln152_resume_ft_high_bce_best.pth.tar \
    --arch linknet152 --img_size 288 --lr 1e-5 \
    --batch-size 30 --workers 8 \
    --epoch_fraction 0.1 \
    --ths 0.5 --tensorboard True  --tensorboard_images True \
    --do_energy_levels False --do_boundaries False \
    --print-freq 20 \
    --lognumber ln152_resume_ft_high_bce_100more_val2 \
    --do_remove_small_on_borders False \
    --do_produce_sizes_mask True --do_produce_distances_mask True \
    --w0 5.0 --sigma 10.0 \
    --evaluate