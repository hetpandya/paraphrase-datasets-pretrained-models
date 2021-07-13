# Paraphrase datasets and pretrained models
This repository consists of a collection of preprocessed datasets for training models and pretrained models to generate paraphrases.

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
t5-small| [Quora Question Pairs](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) | [huggingface](https://huggingface.co/hetpandya/t5-small-quora)
t5-base| [tapaco](https://github.com/hetpandya/paraphrase-datasets-pretrained-models/tree/main/datasets/tapaco) | [huggingface](https://huggingface.co/hetpandya/t5-base-tapaco)

### Model Training
Examples for training models on the datasets can be found in the [examples](https://github.com/hetpandya/paraphrase-datasets-pretrained-models/tree/main/examples) directory.

### T5 model usage example

**Install dependencies using:**

```
pip install transformers sentencepiece
```

**Usage**

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("hetpandya/t5-small-tapaco")
model = T5ForConditionalGeneration.from_pretrained("hetpandya/t5-small-tapaco")

def get_paraphrases(sentence, prefix="paraphrase: ", n_predictions=5, top_k=120, max_length=256,device="cpu"):
        text = prefix + sentence + " </s>"
        encoding = tokenizer.encode_plus(
            text, pad_to_max_length=True, return_tensors="pt"
        )
        input_ids, attention_masks = encoding["input_ids"].to(device), encoding[
            "attention_mask"
        ].to(device)

        model_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_masks,
            do_sample=True,
            max_length=max_length,
            top_k=top_k,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=n_predictions,
        )

        outputs = []
        for output in model_output:
            generated_sent = tokenizer.decode(
                output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            if (
                generated_sent.lower() != sentence.lower()
                and generated_sent not in outputs
            ):
                outputs.append(generated_sent)
        return outputs

paraphrases = get_paraphrases("The house will be cleaned by me every Saturday.")

for sent in paraphrases:
  print(sent)
```

**Output**
```
The house is cleaned every Saturday by me.
The house will be cleaned on Saturday.
I will clean the house every Saturday.
I get the house cleaned every Saturday.
I will clean this house every Saturday.
```
