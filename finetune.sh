#1/bin/sh
:'
Modify these paths as needed.
'
export PROJ_DIR=/nlp/u/kawin/snli
export MODEL_DIR=/nlp/scr/kawin
export TRANSFORMERS_CACHE="/nlp/scr/kawin/huggingface/"
export HF_DATASETS_CACHE="/nlp/scr/kawin/huggingface/datasets/"

# for large models (e.g., roberta-large), use batch size 64
BATCH_SIZE=32
MODEL=$1

:'
Train the model specified by the first argument on every
(un)transformed dataset. Training a model is our way of finding
the function in the function family that minimizes the conditional
entropy of Y given X (or the null variable). For every dataset
we consider (SNLI, MNLI, etc.), one of the datasets we train on 
will have X replaced with a null variable.
'

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/snli_train_std.csv \
  --validation_file $PROJ_DIR/data/snli_train_std.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 1 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_snli_std


python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/snli_train_null.csv \
  --validation_file $PROJ_DIR/data/snli_train_std.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 1 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_snli_null


# train over time

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL_DIR/finetuned/${MODEL////-}_snli_std \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/snli_train_std.csv \
  --validation_file $PROJ_DIR/data/snli_train_std.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 1 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_snli_std2

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL_DIR/finetuned/${MODEL////-}_snli_std2 \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/snli_train_std.csv \
  --validation_file $PROJ_DIR/data/snli_train_std.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 1 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_snli_std3

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL_DIR/finetuned/${MODEL////-}_snli_std3 \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/snli_train_std.csv \
  --validation_file $PROJ_DIR/data/snli_train_std.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_snli_std5

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL_DIR/finetuned/${MODEL////-}_snli_std5 \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/snli_train_std.csv \
  --validation_file $PROJ_DIR/data/snli_train_std.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 5 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_snli_std10


# train fractionally

for seed in b c d e
do
for fraction in 0.05 0.2 0.4 0.6 0.8 0.99
do
python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/frac/snli_train_null_${seed}_${fraction}.csv \
  --validation_file $PROJ_DIR/data/frac/snli_test_null_${seed}.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_snli_null_${seed}_${fraction}

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/frac/snli_train_std_${seed}_${fraction}.csv \
  --validation_file $PROJ_DIR/data/frac/snli_test_std_${seed}.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_snli_std_${seed}_${fraction}
done
done


# special cases

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/snli_train_hypothesis.csv \
  --validation_file $PROJ_DIR/data/snli_train_hypothesis.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_snli_hypothesis

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/snli_train_premise.csv \
  --validation_file $PROJ_DIR/data/snli_train_premise.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_snli_premise

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/snli_train_raw_overlap.csv \
  --validation_file $PROJ_DIR/data/snli_train_raw_overlap.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_snli_raw_overlap

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/snli_train_length.csv \
  --validation_file $PROJ_DIR/data/snli_train_length.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_snli_length

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/snli_train_shuffled.csv \
  --validation_file $PROJ_DIR/data/snli_train_shuffled.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_snli_shuffled

# COLA

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/cola_train_null.csv \
  --validation_file $PROJ_DIR/data/cola_train_null.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_cola_null

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/cola_train_std.csv \
  --validation_file $PROJ_DIR/data/cola_train_std.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 1 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_cola_std1

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL_DIR/finetuned/${MODEL////-}_cola_std1 \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/cola_train_std.csv \
  --validation_file $PROJ_DIR/data/cola_train_std.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 1 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_cola_std2

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL_DIR/finetuned/${MODEL////-}_cola_std2 \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/cola_train_std.csv \
  --validation_file $PROJ_DIR/data/cola_train_std.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 1 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_cola_std3

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL_DIR/finetuned/${MODEL////-}_cola_std3 \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/cola_train_std.csv \
  --validation_file $PROJ_DIR/data/cola_train_std.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_cola_std5

# DWMW 

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/dwmw_std.csv \
  --validation_file $PROJ_DIR/data/dwmw_std.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_dwmw_std

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/dwmw_null.csv \
  --validation_file $PROJ_DIR/data/dwmw_null.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_dwmw_null

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/dwmw_bad_vocab.csv \
  --validation_file $PROJ_DIR/data/dwmw_bad_vocab.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_dwmw_bad_vocab

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/dwmw_sentiment_vocab.csv \
  --validation_file $PROJ_DIR/data/dwmw_sentiment_vocab.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_dwmw_sentiment_vocab

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/dwmw_sentiment.csv \
  --validation_file $PROJ_DIR/data/dwmw_sentiment.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_dwmw_sentiment

# MultiNLI 

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/multinli_train_std.csv \
  --validation_file $PROJ_DIR/data/multinli_validation_std.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 2 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_multinli_std

python run_glue_no_trainer.py \
  --model_name_or_path $MODEL \
  --tokenizer_name $MODEL \
  --train_file $PROJ_DIR/data/multinli_train_null.csv \
  --validation_file $PROJ_DIR/data/multinli_validation_null.csv \
  --per_device_train_batch_size $BATCH_SIZE \
  --per_device_eval_batch_size $BATCH_SIZE \
  --num_train_epochs 1 \
  --seed 1 \
  --output_dir $MODEL_DIR/finetuned/${MODEL////-}_multinli_null

