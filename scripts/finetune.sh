torchrun --nproc_per_node=4 \
    -m src.train.main \
    --train-data /home/user/intorains/Matterport3D_O \
    --train-num-samples 8000000 \
    --aug-cfg scale='(0.8, 1.0)' color_jitter='(0.32, 0.32, 0.32, 0.08)' color_jitter_prob=0.8 gray_scale_prob=0.2 \
    --beta1 0.9 \
    --beta2 0.95 \
    --lr "1e-5" \
    --epochs 1280 \
    --batch-size 128 \
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
    --lock-image \
    --pretrained /root/.cache/huggingface/hub/models--SanghyukChun--ProLIP-ViT-B-16-DC-1B-12_8B/snapshots/0c0b4d6b5e29f8f3b5e5f3a01a2b04409e060d0d \
    --torchcompile 
        # --drop-ratio 0.125 \
    # --drop-prob 0.75 \

        

        # --imagenet-val $IMAGENET_PATH \
    # --coco-test $COCO_PATH \
        # --nnodes=1 \
    # --node_rank=$RANK \
    # --master_addr=$MASTER_ADDR \
    # --master_port=$MASTER_PORT \
