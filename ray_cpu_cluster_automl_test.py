from typing import List, Union, Callable, Dict, Type, Tuple
import ray
import itertools
import pandas as pd
import numpy as np
import os
from collections import defaultdict
from statsforecast import StatsForecast
from statsforecast.models import ETS, AutoARIMA, _TS
from pyarrow import parquet as pq
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error


@ray.remote
def train_and_evaluate_fold(
    model: _TS,
    df: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    label_column: str,
    metrics: Dict[str, Callable[[pd.Series, pd.Series], float]],
    freq: str = "D",
) -> Dict[str, float]:
    try:
        # Create the StatsForecast object with train data & model.
        statsforecast = StatsForecast(df=df.iloc[train_indices], models=[model], freq=freq)
        # Make a forecast and calculate metrics on test data.
        # This will fit the model first automatically.
        forecast = statsforecast.forecast(len(test_indices))
        return {
            metric_name: metric(df.iloc[test_indices][label_column], forecast[model.__class__.__name__])
            for metric_name, metric in metrics.items()
        }
    except Exception:
        # In case the model fit or eval fails, return None for all metrics.
        return {metric_name: None for metric_name, metric in metrics.items()}


def evaluate_models_with_cv(
    models: List[_TS],
    df: pd.DataFrame,
    label_column: str,
    metrics: Dict[str, Callable[[pd.Series, pd.Series], float]],
    freq: str = "D",
    cv: Union[int, TimeSeriesSplit] = 5,
) -> Dict[_TS, Dict[str, float]]:
    # Obtain CV train-test indices for each fold.
    if isinstance(cv, int):
        cv = TimeSeriesSplit(cv)
    train_test_indices = list(cv.split(df))

    # Put df into Ray object store for better performance.
    df_ref = ray.put(df)

    # Add tasks to be executed for each fold.
    fold_refs = []
    for model in models:
        fold_refs.extend(
            [
                train_and_evaluate_fold.remote(
                    model,
                    df_ref,
                    train_indices,
                    test_indices,
                    label_column,
                    metrics,
                    freq=freq,
                )
                for train_indices, test_indices in train_test_indices
            ]
        )

    fold_results = ray.get(fold_refs)

    # Split fold results into a list of CV splits-sized chunks.
    # Ray guarantees that order is preserved.
    fold_results_dim = len(fold_results)
    train_test_indices = len(train_test_indices)

    fold_results_per_model = [  # noqa: E203
        fold_results[i : i + train_test_indices] for i in range(0, fold_results_dim, train_test_indices)  # noqa: E203
    ]  # noqa: E203

    # Aggregate and average results from all folds per model.
    # We go from a list of dicts to a dict of lists and then
    # get a mean of those lists.
    mean_results_per_model = []
    for model_results in fold_results_per_model:
        aggregated_results = defaultdict(list)
        for fold_result in model_results:
            for metric, value in fold_result.items():
                aggregated_results[metric].append(value)
        mean_results = {metric: np.mean(values) for metric, values in aggregated_results.items()}
        mean_results_per_model.append(mean_results)

    # Join models and their metrics together.
    mean_results_per_model = {models[i]: mean_results_per_model[i] for i in range(len(mean_results_per_model))}
    return mean_results_per_model


def generate_configurations(search_space: Dict[Type[_TS], Dict[str, list]]) -> _TS:
    # Convert dict search space into configurations - models instantiated with specific arguments.
    for model, model_search_space in search_space.items():
        kwargs, values = model_search_space.keys(), model_search_space.values()
        # Get a product - all combinations in the per-model grid.
        for configuration in itertools.product(*values):
            yield model(**dict(zip(kwargs, configuration)))


def evaluate_search_space_with_cv(
    search_space: Dict[Type[_TS], Dict[str, list]],
    df: pd.DataFrame,
    label_column: str,
    metrics: Dict[str, Callable[[pd.Series, pd.Series], float]],
    eval_metric: str,
    mode: str = "min",
    freq: str = "D",
    cv: Union[int, TimeSeriesSplit] = 5,
) -> List[Tuple[_TS, Dict[str, float]]]:
    assert eval_metric in metrics
    assert mode in ("min", "max")

    configurations = list(generate_configurations(search_space))
    print(
        f"Evaluating {len(configurations)} configurations with {cv.get_n_splits()} splits each, "
        f"totalling {len(configurations)*cv.get_n_splits()} tasks..."
    )
    ret = evaluate_models_with_cv(configurations, df, label_column, metrics, freq=freq, cv=cv)

    # Sort the results by eval_metric
    ret = sorted(ret.items(), key=lambda x: x[1][eval_metric], reverse=(mode == "max"))
    print("Evaluation complete!")
    return ret


def get_m5_partition(unique_id: str) -> pd.DataFrame:
    ds1 = pq.read_table(
        "s3://anonymous@m5-benchmarks/data/train/target.parquet",
        filters=[("item_id", "=", unique_id)],
    )
    Y_df = ds1.to_pandas()
    # StatsForecasts expects specific column names!
    Y_df = Y_df.rename(columns={"item_id": "unique_id", "timestamp": "ds", "demand": "y"})
    Y_df["unique_id"] = Y_df["unique_id"].astype(str)
    Y_df["ds"] = pd.to_datetime(Y_df["ds"])
    Y_df = Y_df.dropna()
    constant = 10
    Y_df["y"] += constant
    return Y_df[Y_df.unique_id == unique_id]


if __name__ == "__main__":

    if ray.is_initialized() is False:
        print("Connecting to Ray cluster...")
        service_host = os.environ["RAY_HEAD_SERVICE_HOST"]
        service_port = os.environ["RAY_HEAD_SERVICE_PORT"]
        ray.util.connect(f"{service_host}:{service_port}")

    df = get_m5_partition("FOODS_1_001_CA_1")
    tuning_results = evaluate_search_space_with_cv(
        {AutoARIMA: {}, ETS: {"season_length": [6, 7], "model": ["ZNA", "ZZZ"]}},
        df,
        "y",
        {"mse": mean_squared_error, "mae": mean_absolute_error},
        "mse",
        cv=TimeSeriesSplit(test_size=1),
    )
