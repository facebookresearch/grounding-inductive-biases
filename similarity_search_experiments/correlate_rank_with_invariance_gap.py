from equivariance_measure.embedding_distances import TransformInvarianceDigest
from functools import partial
import pandas as pd
import scipy.stats as ss
import numpy as np
from pathlib import Path
from equivariance_measure.transformations import TRANSFORMATION_NAMES
from equivariance_measure.distance_measures import load_invariances_per_model
from typing import Tuple
import plotly.graph_objects as go
from scipy import stats


LABELS_PATH = Path(
    ""
)
SIM_SEARCH_DIR = Path(
    ""
)


class TransformRank:
    """Compares wordnet class pair similarity against transform rankings"""

    def __init__(
        self,
        min_magnitude: int = 3,
        max_magnitude: int = 7,
        rank_by: str = "avg_percent_similarity_change",
    ):
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        self.rank_by = rank_by
        self.df = self.load_df()
        self.store_ranks()
        self.transform_to_avg_rank = (
            self.df.groupby("transform")["transform_rank"].mean().to_dict()
        )

    def load_df(
        self, data_set_type="train", similarity_type="resnet18_no_aug"
    ) -> pd.DataFrame:
        """Loads dataframe of similarity changes"""
        labels_df = pd.read_csv(
            LABELS_PATH,
            sep=" ",
            names=["class_label", "class_num", "class_name"],
        )

        sub_dir = f"similarity_search_{similarity_type}"
        results_dir = Path(SIM_SEARCH_DIR / f"{sub_dir}").expanduser()
        transform_type = "single_transform"
        df = pd.read_csv(
            results_dir / f"{transform_type}_boosts_{data_set_type}.csv", index_col=0
        )
        df = df.merge(labels_df, on="class_label", how="left")
        df["transform"] = df["transform_name"].str.split(" ").apply(lambda x: x[0])
        df["transform_rank"] = np.nan
        df = self._filter_df(df)
        return df

    def _filter_df(self, df):
        df["magnitude"] = (
            df["transform_name"].str.split(" ").apply(lambda x: int(x[-1]))
        )
        valid_magnitudes = list(range(self.min_magnitude, self.max_magnitude + 1))
        df = df[df["magnitude"].isin(valid_magnitudes)]
        return df

    def _store_ranks_for_class(self, class_label):
        class_df = self.df[self.df["class_label"] == class_label].sort_values(
            by=self.rank_by, ascending=False
        )
        ranks = ss.rankdata(class_df["transform"], method="min")
        self.df.loc[class_df["transform_rank"].index, "transform_rank"] = ranks

    def store_ranks(self):
        for class_label in self.df["class_label"].unique():
            self._store_ranks_for_class(class_label)


class TransformInvarianceGap:
    def __init__(
        self,
        min_magnitude: int = 3,
        max_magnitude: int = 7,
    ):
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude
        self.invariance_per_model = load_invariances_per_model("train", "cos_distance")
        self.model_names = [m.model_name for m in self.invariance_per_model]

        self.transform_to_vit_gap = self.compute_gap("ViT (L/16)")
        self.transform_to_resnet_gap = self.compute_gap("ResNet18")

    def get_invariance(self, model_name: str):
        i = self.model_names.index(model_name)
        return self.invariance_per_model[i]

    def compute_gap(self, model_type: str) -> dict:
        transform_gap = dict()
        for transform in TRANSFORMATION_NAMES:
            transform_gap[transform] = self._compute_transform_gap(
                transform, model_type
            )
        return transform_gap

    def _compute_transform_gap(self, transform: str, model_type: str) -> float:
        invariance_trained = self.get_invariance(f"Trained {model_type}")
        invariance_untrained = self.get_invariance(f"Untrained {model_type}")
        gaps = []

        for magnitude in range(self.min_magnitude, self.max_magnitude + 1):
            trained = invariance_trained.get_percentile(transform, magnitude, 50)
            untrained = invariance_untrained.get_percentile(transform, magnitude, 50)
            gaps.append(trained - untrained)

        return np.mean(gaps)


class Correlate:
    def __init__(
        self,
        min_magnitude: int = 3,
        max_magnitude: int = 7,
        rank_by: str = "avg_percent_similarity_change",
    ):
        self.ranks = TransformRank(
            min_magnitude=min_magnitude, max_magnitude=max_magnitude, rank_by=rank_by
        )
        self.invariance_gap = TransformInvarianceGap(
            min_magnitude=min_magnitude, max_magnitude=max_magnitude
        )

        self.add_directional_names()
        self.add_invariance_gaps()
        self.results_df = self.ranks.df.dropna()

    def add_invariance_gaps(self):
        gap_func = partial(self.get_invariance_gap, "resnet")
        self.ranks.df["resnet_invariance_gap"] = self.ranks.df[
            "transform_directional_name"
        ].apply(gap_func)

        gap_func = partial(self.get_invariance_gap, "vit")
        self.ranks.df["vit_invariance_gap"] = self.ranks.df[
            "transform_directional_name"
        ].apply(gap_func)

    def get_invariance_gap(self, gap_type: str, transform_name: str):
        """Gap type is resnet or vit"""
        gaps = (
            self.invariance_gap.transform_to_resnet_gap
            if gap_type == "resnet"
            else self.invariance_gap.transform_to_vit_gap
        )
        if transform_name in gaps:
            return gaps[transform_name]
        return np.nan

    def add_directional_names(self) -> None:
        self.ranks.df["transform_directional_name"] = self.ranks.df[
            "transform_name"
        ].apply(self.get_directional_transform_name)

    def get_directional_transform_name(self, rank_name: str) -> str:
        is_exception = self._is_exception(rank_name)
        transform = rank_name.split(" ")[0]
        if not is_exception:
            return transform

        magnitude = int(rank_name.split(" ")[-1])

        if transform == "rescale" and magnitude <= 5:
            return "scale_zoom_in"
        elif transform == "rescale":
            return "scale_zoom_in"
        elif transform == "contrast" and magnitude <= 5:
            return "decrease_contrast"
        elif transform == "contrast":
            return "increase_contrast"

    def _is_exception(self, rank_name: str):
        exceptions = {"rescale", "contrast"}
        for exception in exceptions:
            if exception in rank_name:
                return True
        return False


def plot(top=5, rank_by="proportion_boosted"):
    corr = Correlate()
    top_k = (
        corr.results_df.groupby(["class_label"])
        .apply(lambda x: x.nlargest(top, columns=["proportion_boosted"]))[
            "transform_directional_name"
        ]
        .value_counts()
    )
    fig = go.Figure()
    x, y, sem_perc = binary_bins(top_k, corr, invariance_type="resnet")
    fig.add_trace(
        go.Bar(x=x, y=y, name="ResNet", error_y=dict(type="percent", array=sem_perc))
    )
    x, y, sem_perc = binary_bins(top_k, corr, invariance_type="vit")
    fig.add_trace(
        go.Bar(x=x, y=y, name="ViT", error_y=dict(type="percent", array=sem_perc))
    )
    fig.update_xaxes(title="Invariance after training")
    fig.update_yaxes(title="Among top 5")
    fig.update_layout(
        title="Invariance increases for factors of variation",
        yaxis=dict(tickformat=".0%"),
    )
    return fig


def binary_bins(top_k, corr, invariance_type="resnet"):
    positive_values = []
    negative_values = []
    bins = ["decreases", "increases"]

    for transform_name, top_count in top_k.to_dict().items():
        gap = corr.get_invariance_gap(invariance_type, transform_name)
        if gap > 0:
            positive_values.append(top_count)
        else:
            negative_values.append(top_count)
    positive_values = np.array(positive_values)
    negative_values = np.array(negative_values)
    total = np.sum(positive_values) + np.sum(negative_values)
    positive_prop, negative_prop = positive_values / total, negative_values / total
    y = [np.mean(negative_prop), np.mean(positive_prop)]
    sem = [stats.sem(negative_prop), stats.sem(positive_prop)]
    sem_perc = np.array(sem) / np.array(y)
    return bins, y, sem_perc
