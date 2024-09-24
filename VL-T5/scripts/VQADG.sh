# The name of experiment
name=VLT5

output=snap/vqadg/$name

PYTHONPATH=$PYTHONPATH:./src \
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/vqadg2.py \
        --distributed --multiGPU \
        --train vqadg_train \
        --valid vqadg_val \
        --test vqadg_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 50 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output $output ${@:2} \
        --load snap/pretrain/VLT5/Epoch30 \
        --num_beams 5 \
        --batch_size 16 \
        --valid_batch_size 16 \