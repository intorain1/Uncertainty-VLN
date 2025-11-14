torchrun --nproc_per_node=1 \
    -m src.train.main \
    --train-data /home/user/intorains/Matterport3D_O \
    --train-num-samples 8000000 \
    --aug-cfg scale='(0.8, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --beta1 0.9 \
    --beta2 0.95 \
    --lr "1e-6" \
    --epochs 32 \
    --batch-size 16 \
    --warmup 200 \
    --wd 0.01 \
    --zeroshot-frequency 1 \
    --dataset-resampled \
    --dataset-type seq \
    --model ViT-B-16-ProLIP-long \
    --precision amp_bf16 \
    --workers 4 \
    --prolip-logs /home/user/intorains/Uncertainty-VLN/log \
    --accum-freq 1 \
    --vib-beta 0.0001 \
    --inclusion-alpha 0.0000001 \
    --inclusion-alpha-occ 0.001 \
    --inclusion-scale 10 \
    --inclusion-eps -100 \
    --delete-previous-checkpoint \
    --prolip \
    --force-resize-pos-emb \

        # --drop-ratio 0.125 \
    # --drop-prob 0.75 \
    # --torchcompile
        # --pretrained $PATH_TO_PRETRAINED \

        # --imagenet-val $IMAGENET_PATH \
    # --coco-test $COCO_PATH \
        # --nnodes=1 \
    # --node_rank=$RANK \
    # --master_addr=$MASTER_ADDR \
    # --master_port=$MASTER_PORT \
