# 0
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_feat.py --backbone resnet_feat --lr 0.007 --workers 4 --epochs 100 --batch-size 16 --gpu-ids 0  --eval-interval 1 --dataset crack --start_epoch 0 --checkname raw_feature_focal_loss

# 1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_feat.py --backbone resnet_feat --lr 0.007 --workers 4 --epochs 100 --batch-size 16 --gpu-ids 0  --eval-interval 1 --dataset crack --start_epoch 0 --loss-type mixed --checkname raw_feature_mixed_loss

# 2
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_feat.py --backbone resnet_feat --lr 0.007 --workers 4 --epochs 100 --batch-size 16 --gpu-ids 0  --eval-interval 1 --dataset crack --start_epoch 0  --features False --featuresnew True --loss-type mixed --checkname new_feature_mixed_loss

# 3
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_feat_dense.py --backbone resnet_feat --lr 0.007 --workers 4 --epochs 100 --batch-size 16 --gpu-ids 0  --eval-interval 1 --dataset crack --start_epoch 0 --loss-type mixed --checkname raw_feature_mixed_loss_ASPP

# 4
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_feat_dense.py --backbone resnet_feat --lr 0.007 --workers 4 --epochs 100 --batch-size 16 --gpu-ids 0  --eval-interval 1 --dataset crack --start_epoch 0  --features False --featuresnew True --loss-type mixed --checkname new_feature_mixed_loss_ASPP