#!/usr/bin/bash
#SBATCH -N 1
#SBATCH -p main
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH --exclude=worker-3

#echo 'activating virtual environment'
#eval "$(conda shell.bash hook)"
#conda activate kerui
which python

LM_PATH="/home/wiss/zhang/code/open_flamingo/ckpt/llama/7B"
LM_TOKENIZER_PATH="/home/wiss/zhang/code/open_flamingo/ckpt/llama/7B"
VISION_ENCODER_NAME="ViT-L-14"
VISION_ENCODER_PRETRAINED="openai"
CKPT_PATH="/home/wiss/zhang/code/open_flamingo/ckpt/OpenFlamingo-9B/checkpoint.pt"

TYPE='exist'
RANDOM_ID=$$
PROMPT='wo'
SHOT=32
COMMENT="new04_${SHOT}_${PROMPT}"
RESULTS_FILE="results/gqa_results/${TYPE}/${SHOT}/simplified_exist/${COMMENT}_scores_${RANDOM_ID}"

python eval/evaluate_gqa.py \
    --lm_path $LM_PATH \
    --lm_tokenizer_path $LM_TOKENIZER_PATH \
    --vision_encoder_path $VISION_ENCODER_NAME \
    --vision_encoder_pretrained $VISION_ENCODER_PRETRAINED \
    --checkpoint_path $CKPT_PATH \
    --cross_attn_every_n_layers 4 \
    --device "0" \
    --results_file $RESULTS_FILE \
    --batch_size 1 \
    --eval_gqa \
    --visdial_test_image_path "/home/wiss/zhang/nfs/multiinstruct/MultiInstruct/raw_datasets/visdial/VisualDialog_test2018" \
    --visdial_test_json_path "/home/wiss/zhang/nfs/multiinstruct/MultiInstruct/raw_datasets/visdial/visdial_1.0_test.json" \
    --visdial_val_image_path "/home/wiss/zhang/nfs/multiinstruct/MultiInstruct/raw_datasets/visdial/VisualDialog_val2018" \
    --visdial_val_json_path "/home/wiss/zhang/nfs/multiinstruct/MultiInstruct/raw_datasets/visdial/visdial_1.0_val.json" \
    --visdial_train_image_path "/home/wiss/zhang/nfs/multiinstruct/MultiInstruct/raw_datasets/visdial/VisualDialog_train2018" \
    --visdial_train_json_path "/home/wiss/zhang/nfs/multiinstruct/MultiInstruct/raw_datasets/visdial/visdial_1.0_train.json" \
    --gqa_image_folder_path "/home/wiss/zhang/nfs/multiinstruct/MultiInstruct/raw_datasets/GQA/images" \
    --gqa_train_questions_json_path "/home/wiss/zhang/nfs/multiinstruct/MultiInstruct/raw_datasets/GQA/train_balanced_questions.json" \
    --gqa_val_questions_json_path "/home/wiss/zhang/nfs/multiinstruct/MultiInstruct/raw_datasets/GQA/val_balanced_questions.json" \
    --gqa_test_questions_json_path "/home/wiss/zhang/nfs/multiinstruct/MultiInstruct/raw_datasets/GQA/gqa_exist_new1k.json" \
    --num_samples 5 \
    --query_set_size 100 \
    --shots $SHOT \
    --num_trials 4 \
    --trial_seeds 180 8 50 98 \
    --comment $COMMENT \
    --prompt $PROMPT \
    --type $TYPE


echo "evaluation complete! results written to ${RESULTS_FILE}.json"
#    --gqa_test_questions_json_path "/home/wiss/zhang/nfs/multiinstruct/MultiInstruct/raw_datasets/GQA/testdev_all_questions.json" \
