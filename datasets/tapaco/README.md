# Original dataset credits
Original TaPaCo dataset is available at [huggingface](https://huggingface.co/datasets/tapaco).

# About preprocessed dataset
This dataset is a subset of the original dataset. The aim of processing this dataset is to make it usable to train models to generate paraphrase. It has been converted to `csv` format for ease of use. The original dataset consists of `73 languages`, but this version of the data consists of data in only `English` language.

The processed dataset is available as:
Dataset Type| File Name
--- | ---
Original dataset as csv|  tapaco_huggingface.csv
Only Input text and target text|  tapaco_paraphrases_dataset.csv

# Steps to reproduce the dataset

## Installing dependencies
```pip install datasets tqdm pandas```

## Processing dataset
### Storing original dataset as `csv`:
```python
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

#Loading dataset
dataset = load_dataset('tapaco', 'en')

def process_tapaco_dataset(dataset, out_file):
    tapaco = []
    # The dataset has only train split.
    for data in tqdm(dataset["train"]):
        keys = data.keys()
        tapaco.append([data[key] for key in keys])
    tapaco_df = pd.DataFrame(
        data=tapaco,
        columns=[
            "language",
            "lists",
            "paraphrase",
            "paraphrase_set_id",
            "sentence_id",
            "tags",
        ],
    )
    tapaco_df.to_csv(out_file, sep="\t", index=None)
    return tapaco_df

tapaco_df = process_tapaco_dataset(dataset,"tapaco_huggingface.csv")
```

### Preprocessing the converted csv dataset to an easier format:
```python
def generate_tapaco_paraphrase_dataset(dataset, out_file):
    dataset_df = dataset[["paraphrase", "paraphrase_set_id"]]
    non_single_labels = (
        dataset_df["paraphrase_set_id"]
        .value_counts()[dataset_df["paraphrase_set_id"].value_counts() > 1]
        .index.tolist()
    )
    tapaco_df_sorted = dataset_df.loc[
        dataset_df["paraphrase_set_id"].isin(non_single_labels)
    ]
    tapaco_paraphrases_dataset = []

    for paraphrase_set_id in tqdm(tapaco_df_sorted["paraphrase_set_id"].unique()):
        id_wise_paraphrases = tapaco_df_sorted[
            tapaco_df_sorted["paraphrase_set_id"] == paraphrase_set_id
        ]
        len_id_wise_paraphrases = (
            id_wise_paraphrases.shape[0]
            if id_wise_paraphrases.shape[0] % 2 == 0
            else id_wise_paraphrases.shape[0] - 1
        )
        for ix in range(0, len_id_wise_paraphrases, 2):
            current_phrase = id_wise_paraphrases.iloc[ix][0]
            for count_ix in range(ix + 1, ix + 2):
                next_phrase = id_wise_paraphrases.iloc[ix + 1][0]
                tapaco_paraphrases_dataset.append([current_phrase, next_phrase])
    tapaco_paraphrases_dataset_df = pd.DataFrame(
        tapaco_paraphrases_dataset, columns=["Text", "Paraphrase"]
    )
    tapaco_paraphrases_dataset_df.to_csv(out_file, sep="\t", index=None)
    return tapaco_paraphrases_dataset_df

tapaco_paraphrases_dataset_df = generate_tapaco_paraphrase_dataset(tapaco_df,"tapaco_paraphrases_dataset.csv")
```
