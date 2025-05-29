from pathlib import Path

import polars as pl
from datasets import Dataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

overlapped_categories = {
    "K5_3세대_하이브리드_2020_2022": "K5_하이브리드_3세대_2020_2023",
    "디_올뉴니로_2022_2025": "디_올_뉴_니로_2022_2025",
    "718_박스터_2017_2024": "박스터_718_2017_2024",
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


def train_val_split(df: pl.DataFrame, val_ratio: float = 0.1, random_state: int = 42) -> tuple[pl.DataFrame]:
    val_size = int(len(df) * val_ratio)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    split = list(splitter.split(X=df, y=df["class"]))
    train_indices = split[0][0]
    val_indices = split[0][1]

    df_train = df[train_indices]
    df_val = df[val_indices]
    return df_train, df_val


def get_debug_dataframes(
    df: pl.DataFrame, train_size: int = 500, val_size: int = 100, seed: int = 42
) -> tuple[pl.DataFrame]:
    total_size = train_size + val_size
    df_debug = df.sample(n=total_size, shuffle=True, seed=seed)
    df_debug_train = df_debug[:train_size]
    df_debug_val = df_debug[train_size:]
    return df_debug_train, df_debug_val


def get_dataloader(df: pl.DataFrame, num_data_per_batch: int, shuffle: bool = True) -> DataLoader:
    dataset = Dataset.from_polars(df)
    return DataLoader(dataset=dataset, batch_size=num_data_per_batch, shuffle=shuffle)
