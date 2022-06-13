# Understanding Dataset Difficulty with V-Usable Information (ICML 2022, long talk)

The high-level idea is that dataset difficulty corresponds to the lack of *V-usable information*, a generalization of Shannon Information proposed by [Xu et al. (2019)](https://arxiv.org/abs/2002.10689).
*V* refers to a specific function family (e.g., BERT).

Here's how to use this code to estimate the BERT-usable information in the SNL dataset:

1. Install the packages in requirements.txt: `pip install -r requirements.txt`

2. In augment.py, instantiate the transformation functions and call .transform(). 
   The base class will download the data from [HuggingFace](https://huggingface.co/datasets) if necessary.
   This will create CSV files with two columns: 'sentence1' (containing the input) and 'label' (containing the label).
   `SNLIStandardTransformation` is a pass-through transformation; it does not change the input text in any way.
   `SNLINullTransformation` replaces the input with the null variable (i.e., empty string).
   
    ```
    import augment
    SNLIStandardTransformation('./data').transform()  
    SNLINullTransformation('./data').transform()
    ```
   
3. Use run_glue_no_trainer.py to train two BERT models: one on snli_train_std.csv; one on snli_train_null.csv.
   You can run the commands in finetune.sh (with bert-base-cased as the value for $MODEL).
   Change the values for $PROJ_DIR and $MODEL_DIR as needed.
   
    ```
    python run_glue_no_trainer.py \
    --model_name_or_path bert-base-cased \
    --tokenizer_name bert-base-cased \
    --train_file ./data/snli_train_std.csv \
    --validation_file ./data/snli_train_std.csv \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 2 \
    --seed 1 \
    --output_dir $MODEL_DIR/finetuned/bert-base-cased_snli_std2


    python run_glue_no_trainer.py \
    --model_name_or_path bert-base-cased \
    --tokenizer_name bert-base-cased \
    --train_file ./data/snli_train_null.csv \
    --validation_file ./data/snli_train_std.csv \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 1 \
    --seed 1 \
    --output_dir $MODEL_DIR/finetuned/bert-base-cased_snli_null
    ```
  
    
4. Then estimate the V-usable information using v_info.py. This will write a CSV file of the pointwise V-information (PVI) values -- for every example in the SNLI test -- which you can average to estimate the V-usable information.
    
    ```
    from v_info import v_info

    v_info(
      f"./data/snli_test_std.csv",
      f"{MODEL_DIR}/finetuned/bert-base-cased_snli_std2",
      f"./data/snli_test_null.csv", 
      f"{MODEL_DIR/finetuned/bert-base-cased_snli_null",
      'bert-base-cased',
      out_fn=f"PVI/bert-base-cased_std2_test.csv"
    )
    ```

## Use Cases

Our framework can be used for multiple purposes:

1. Compare different models by replacing bert-base-cased with other models (you can specify a model architecture on Huggingface or one of your own).

2. Compare different datasets by writing transformations for them in augment.py and specifying those files in steps 3 and 4.

3. Compare the usefulness of different attributes of the input. For example, to understand the importance of word order in SNLI, use `SNLIShuffleTransformation` to create transformed SNLI dataset `snli_train_shuffled.csv` and `snli_test_shuffled.csv`. Then repeat steps 3-4 with the new files in place of `snli_train_std.csv` and `snli_test_std.csv`.

4. Compare the difficulty of individual examples in the dataset by looking at the PVI values in the file generated in step 5.

5. Compare the difficulty of subsets of the data by averaging the PVI values of the examples that belong to that subset, which you can do with simple `pandas` operations.


## Citation 

```
@article{ethayarajh2021information,
  title={Information-theoretic measures of dataset difficulty},
  author={Ethayarajh, Kawin and Choi, Yejin and Swayamdipta, Swabha},
  journal={arXiv preprint arXiv:2110.08420},
  year={2021}
}
```


