# Understanding Dataset Difficulty with V-Usable Information (ICML 2022, long talk)

Link to Paper: https://arxiv.org/abs/2110.08420

The high-level idea is that dataset difficulty corresponds to the lack of *V-usable information*, a generalization of Shannon Information proposed by [Xu et al. (2019)](https://arxiv.org/abs/2002.10689).
*V* refers to a specific function family (e.g., BERT).

## Quick Start

Here's how to use this package to estimate the BERT-usable information in the [SNLI dataset](https://arxiv.org/abs/1508.05326):

1. Install the packages in requirements.txt: `pip install -r requirements.txt`

2. In `augment.py`, instantiate the transformations and call .transform(). 
   The base class will download the data from [HuggingFace](https://huggingface.co/datasets).
   This will create CSV files with two columns: 'sentence1' (containing the input) and 'label' (containing the label).
   `SNLIStandardTransformation` is a pass-through transformation that produces `snli_train_std.csv` and `snli_test_std.csv`; it does not change the input text in any way.
   `SNLINullTransformation` replaces the input with the null variable (i.e., empty string) to produce `snli_train_null.csv` and `snli_test_null.csv`
   
    ```
    import augment
    SNLIStandardTransformation('./data').transform()  
    SNLINullTransformation('./data').transform()
    ```
   
3. Use `run_glue_no_trainer.py` to train two BERT models: one on `snli_train_std.csv`; one on `snli_train_null.csv`.
   You can run the commands in `finetune.sh` (with bert-base-cased as the value for $MODEL).
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
  
    
4. Then estimate the V-usable information using `v_info.py`. This will write a CSV file of the pointwise V-information (PVI) values -- for every example in the SNLI test data -- which you can average to estimate the V-usable information.
    
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

2. Compare different datasets by writing transformations for them in `augment.py` and specifying those files in steps 3-4.

3. Compare the usefulness of different attributes of the input. For example, to understand the importance of word order in SNLI, use `augment.SNLIShuffleTransformation` to create transformed SNLI datasets `snli_train_shuffled.csv` and `snli_test_shuffled.csv`. Then repeat steps 3-4 with the new files in place of `snli_train_std.csv` and `snli_test_std.csv`.

4. Compare the difficulty of individual examples in the dataset by looking at the PVI values in the file generated in step 5.

5. Compare the difficulty of subsets of the data by averaging the PVI values of the examples that belong to that subset, which you can do with simple `pandas` operations.

## Reported Results

All the results reported in the paper can be found in the PVI directory. Each file is named with the format `{model}_{dataset}_{transformation}_{train/test}.csv`, though there are some files where the transformation and train/test split were swapped (or omitted, if there was no train/test split).

In addition to the columns present in the original dataset, each results CSV has the following columns:
- `H_yb`: the pointwise label entropy
- `H_yx`: the pointwise conditional entropy
- `predicted_label`: the label predicted by the model for that example
- `correct_yx`: whether the predicted label is correct or not
- `PVI`: the pointwise V-usable information (i.e., `H_yb - H_yx`)

You can average `correct_yx` to get the model accuracy and average `PVI` to estimate V-usable information with respect to the model V.

The `data` directory contains the MultiNLI, CoLA, and DWMW17 datasets with various transformations, such as shuffled word order. SNLI is not included because it is too large, though it can easily be downloaded from Huggingface. The `artefacts` directory contains the token-level annotation artefacts that were discovered in each dataset (see section 4.3 of the paper for details).

## Citation 

```
@InProceedings{pmlr-v162-ethayarajh22a,
  title = 	 {Understanding Dataset Difficulty with $\mathcal{V}$-Usable Information},
  author =       {Ethayarajh, Kawin and Choi, Yejin and Swayamdipta, Swabha},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {5988--6008},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/ethayarajh22a/ethayarajh22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/ethayarajh22a.html}
}
```


