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

            for sample in dataset["train"]["translation"]:
                splits[split].append((sample[source_key], sample[target_key]))

    return splits


