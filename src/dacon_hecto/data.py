from pathlib import Path

import polars as pl
from sklearn.model_selection import StratifiedShuffleSplit


def get_class_names(data_path: Path) -> list[str]:
    train_path = data_path / "train"
    class_path = list(train_path.iterdir())
    class_names = sorted([c.name for c in class_path])
    return class_names


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
