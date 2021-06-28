# Paraphrase datasets and pretrained models
This repository consists of a collection of preprocessed datasets and pretrained models for training models to generate paraphrases.

## Datasets
Each dataset has a `README` that describes the dataset, its source and preprocessed format. The datasets are stored in the [datasets](https://github.com/hetpandya/paraphrase-datasets-pretrained-models/tree/main/datasets) directory.

Dataset Type| File Name
--- | ---
TaPaCo Original|  [tapaco_huggingface.csv](https://github.com/hetpandya/paraphrase-datasets-pretrained-models/blob/main/datasets/tapaco/tapaco_huggingface.csv)
TaPaCo Preprocessed|  [tapaco_paraphrases_dataset.csv](https://github.com/hetpandya/paraphrase-datasets-pretrained-models/blob/main/datasets/tapaco/tapaco_paraphrases_dataset.csv)

## Pretrained models
List of models trained on various datasets for paraphrase generation.

Model| Dataset | Location
--- | --- | --- |
t5-small| [tapaco](https://github.com/hetpandya/paraphrase-datasets-pretrained-models/tree/main/datasets/tapaco) | [huggingface](https://huggingface.co/hetpandya/t5-small-tapaco)
t5-base| [tapaco](https://github.com/hetpandya/paraphrase-datasets-pretrained-models/tree/main/datasets/tapaco) | [huggingface](https://huggingface.co/hetpandya/t5-base-tapaco)
