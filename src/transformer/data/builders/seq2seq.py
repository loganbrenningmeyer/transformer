import datasets
from omegaconf import DictConfig


def load_ted_talks(data_config: DictConfig) -> dict:
    source_key = data_config.source_key
    target_key = data_config.target_key

    split_years = {
        "train": ["2014", "2015"], 
        "valid": ["2016"]
    }

    splits = {"train": [], "valid": []}

    for split, years in split_years.items():

        for year in years:
            dataset = datasets.load_dataset(
                "IWSLT/ted_talks_iwslt",
                language_pair=(source_key, target_key),
                year=year
            )

            source_texts = dataset["train"]["translation"][source_key]
            target_texts = dataset["train"]["translation"][target_key]

            for source_text, target_text in zip(source_texts, target_texts):
                splits[split].append((source_text, target_text))

    return splits


