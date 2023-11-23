export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="../data/Brain"
export OUTPUT_DIR="models"

accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$TRAIN_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=$TRAIN_DIR \
  --with_prior_preservation --real_prior --prior_loss_weight=1.0 \
  --class_prompt="Brain" --num_class_images=200 \
  --instance_prompt="XXXX of Brain"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --scale_lr --hflip  \
  --modifier_token "XXXX" \
  --validation_prompt="XXXX of Brain" \