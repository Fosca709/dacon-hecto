import random
from collections import defaultdict
from pathlib import Path

import polars as pl
from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Sampler

overlapped_categories = {
    "K5_3세대_하이브리드_2020_2022": "K5_하이브리드_3세대_2020_2023",
    "디_올뉴니로_2022_2025": "디_올_뉴_니로_2022_2025",
    "718_박스터_2017_2024": "박스터_718_2017_2024",
    "RAV4_2016_2018": "라브4_4세대_2013_2018",
    "RAV4_5세대_2019_2024": "라브4_5세대_2019_2024",
}


def get_class_names(data_path: Path) -> list[str]:
    train_path = data_path / "train"
    class_path = list(train_path.iterdir())
    class_names = sorted([c.name for c in class_path])
    for c in overlapped_categories.keys():
        class_names.remove(c)
    return class_names


def substitute_categories(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("class").map_elements(lambda x: overlapped_categories.get(x, x), return_dtype=pl.String)
    )


def get_train_dataframe(data_path: Path) -> pl.DataFrame:
    train_path = data_path / "train"
    class_folders = list(train_path.iterdir())
    df = pl.DataFrame({"class": [c.name for c in class_folders], "folder_path": class_folders})

    def _get_image_path(folder_path: Path):
        return [p.as_posix() for p in folder_path.iterdir()]

    df = df.select(
        pl.col("class"),
        pl.col("folder_path").alias("image_path").map_elements(_get_image_path, return_dtype=pl.List(pl.String)),
    )
    df = df.explode(pl.col("image_path"))
    df = substitute_categories(df)
    return df


def train_val_split(
    df: pl.DataFrame, val_ratio: float = 0.1, random_state: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    val_size = int(len(df) * val_ratio)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    split = list(splitter.split(X=df, y=df["class"]))
    train_indices = split[0][0]
    val_indices = split[0][1]

    df_train = df[train_indices]
    df_val = df[val_indices]
    return df_train, df_val


def get_debug_dataframes(
    df: pl.DataFrame, train_size: int = 512, val_size: int = 128, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame]:
    total_size = train_size + val_size
    df_debug = df.sample(n=total_size, shuffle=True, seed=seed)
    df_debug_train = df_debug[:train_size]
    df_debug_val = df_debug[train_size:]
    return df_debug_train, df_debug_val


def get_dataloader(df: pl.DataFrame, num_data_per_batch: int, shuffle: bool = True) -> DataLoader:
    dataset = Dataset.from_polars(df)
    return DataLoader(dataset=dataset, batch_size=num_data_per_batch, shuffle=shuffle)


def get_dataloader_from_config(
    data_path: Path,
    num_data_per_batch: int,
    eval_batch_size: int = 32,
    use_val: bool = True,
    use_confused_pairs: bool = False,
    num_pairs_per_batch: int = 1,
    confusion_pairs: list[tuple[str, str]] | None = None,
    debug: bool = False,
) -> tuple[DataLoader, DataLoader | None]:
    df = get_train_dataframe(data_path)

    if debug:
        df_train, df_val = get_debug_dataframes(df)
        if not use_val:
            df_val = None

    else:
        if use_val:
            df_train, df_val = train_val_split(df)

        else:
            df_train = df
            df_val = None

    if use_confused_pairs:
        train_dataloader = get_dataloader(df=df_train, num_data_per_batch=num_data_per_batch, shuffle=True)
    else:
        train_dataloader = get_confused_pair_dataloader(
            df=df,
            confusion_pairs=confusion_pairs,
            num_data_per_batch=num_data_per_batch,
            num_pairs_per_batch=num_pairs_per_batch,
        )

    if df_val is None:
        val_dataloader = None
    else:
        val_dataloader = get_dataloader(df=df_val, num_data_per_batch=eval_batch_size, shuffle=False)
    return train_dataloader, val_dataloader


def get_test_dataframe(data_path: Path) -> pl.DataFrame:
    test_path = data_path / "test"
    image_paths = [p.as_posix() for p in test_path.iterdir()]
    ids = [p.name for p in test_path.iterdir()]
    ids = [i.split(".")[0] for i in ids]
    return pl.DataFrame({"ID": ids, "image_path": image_paths})


class SetWithChoice:
    def __init__(self, data: list | None = None):
        assert len(data) == len(set(data))
        self.data = [] if data is None else data.copy()
        self.index_map = {k: i for i, k in enumerate(self.data)}

    def add(self, val):
        if val in self.index_map:
            return

        self.index_map[val] = len(self.data)
        self.data.append(val)

    def remove(self, val):
        idx = self.index_map[val]
        last_element = self.data[-1]

        self.data[idx] = last_element
        self.index_map[last_element] = idx

        self.data.pop()
        del self.index_map[val]

    def contains(self, val):
        return val in self.index_map

    def choice(self):
        return random.choice(self.data)

    def choice_and_remove(self):
        val = self.choice()
        self.remove(val)
        return val


class ConfusedPairSampler(Sampler):
    def __init__(
        self, dataset: Dataset, confusion_pairs: list[tuple[str, str]], batch_size: int, num_pairs_per_batch: int
    ):
        self.dataset = dataset
        self.confusion_pairs = confusion_pairs
        self.batch_size = batch_size
        self.num_pairs_per_batch = num_pairs_per_batch

        assert batch_size >= 2 * num_pairs_per_batch

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        remain_pairs = SetWithChoice(data=self.confusion_pairs)
        total_indices = SetWithChoice(data=list(range(len(self.dataset))))
        class_indices = defaultdict(list)
        for i, d in enumerate(self.dataset):
            class_indices[d["class"]].append(i)
        class_indices = {k: SetWithChoice(data=v) for k, v in class_indices.items()}

        i = 0
        while i < len(self.dataset):
            use_random_sample = True
            if (i % self.batch_size) < 2 * self.num_pairs_per_batch:
                while remain_pairs.data:
                    pair = remain_pairs.choice()
                    if class_indices[pair[0]].data and class_indices[pair[1]].data:
                        idx_0 = class_indices[pair[0]].choice_and_remove()
                        total_indices.remove(idx_0)
                        yield idx_0

                        idx_1 = class_indices[pair[1]].choice_and_remove()
                        total_indices.remove(idx_1)
                        yield idx_1

                        i += 2
                        use_random_sample = False
                        break

                    else:
                        remain_pairs.remove(pair)

            if use_random_sample:
                idx = total_indices.choice_and_remove()
                chosen_class = self.dataset[idx]["class"]
                class_indices[chosen_class].remove(idx)
                i += 1
                yield idx


def get_confused_pair_dataloader(
    df: pl.DataFrame, confusion_pairs: list[tuple[str, str]], num_data_per_batch: int, num_pairs_per_batch: int
) -> DataLoader:
    dataset = Dataset.from_polars(df)
    sampler = ConfusedPairSampler(
        dataset=dataset,
        confusion_pairs=confusion_pairs,
        batch_size=num_data_per_batch,
        num_pairs_per_batch=num_pairs_per_batch,
    )
    return DataLoader(dataset=dataset, batch_size=num_data_per_batch, sampler=sampler)
