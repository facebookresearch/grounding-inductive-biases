"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Compute transformation rank correlation against class label wordnet similarity
"""

from pathlib import Path
from nltk.corpus.reader.bracket_parse import WORD
import plotly.express as px
import pandas as pd
import numpy as np
import os
from nltk.corpus import wordnet as wn
from typing import Tuple, List, Any, Dict
from scipy.stats import spearmanr
from tqdm import tqdm


LABELS_PATH = Path(
    ""
)
SIM_SEARCH_DIR = Path(
    ""
)

RESULTS_DIR = ""
WORD_NET_SIMILARITIES = ["wup_similarity", "lch_similarity", "path_similarity"]


class WordNetTransformCorrelations:
    """Compares wordnet class pair similarity against transform rankings"""

    def __init__(
        self,
        wordnet_similarity_methods: List[str] = WORD_NET_SIMILARITIES,
        transform_similarity_methods: List[str] = [
            "spearman_rank",
            "intersection_over_union",
        ],
        max_class_pairs=10000,
        results_dir=RESULTS_DIR,
    ):
        self.max_class_pairs = max_class_pairs
        self.results_dir = results_dir
        self.df = self.load_df()
        self.wordnet_similarity_funcs = [
            WordNetSimilarity(similarity_type=w) for w in wordnet_similarity_methods
        ]
        self.transform_similarity_funcs = [
            TransformSimlarity(similarity_type=t, union_top_n=25)
            for t in transform_similarity_methods
        ]
        self.transform_similarity_funcs.extend(
            [
                TransformSimlarity(similarity_type=t, union_top_n=5)
                for t in transform_similarity_methods
            ]
        )
        self.results_df = None

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
        return df

    def records_to_df(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        return pd.DataFrame.from_records(records)

    def load_results(self, results_dir: str) -> pd.DataFrame:
        self.results_df = pd.read_csv(results_dir, index_col=0)
        return self.results_df

    def save_results(self) -> None:
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, "wordnet_correlations.csv")
        self.results_df.to_csv(path)

    def run(self) -> pd.DataFrame:
        class_names = self.df["class_name"].unique()
        records = []

        for n in tqdm(range(self.max_class_pairs)):
            c1, c2 = np.random.choice(class_names, size=2)
            record = {"class_1": c1, "class_2": c2}

            for wordnet_func in self.wordnet_similarity_funcs:
                record[wordnet_func.name] = wordnet_func(c1, c2)

            for transform_func in self.transform_similarity_funcs:
                record[transform_func.name] = transform_func(c1, c2, self.df)
            records.append(record)

        results_df = self.records_to_df(records)
        self.results_df = results_df
        self.save_results()
        return results_df


class WordNetSimilarity:
    def __init__(self, similarity_type="wup_similarity"):
        assert (
            similarity_type in WORD_NET_SIMILARITIES
        ), f"{similarity_type} not suported"
        self.similarity_type = similarity_type

    @property
    def name(self):
        return self.similarity_type

    def __call__(self, class_name_1: str, class_name_2: str) -> float:
        """Computes WordNet hierarchy similarity using least common ancenstor of the classes"""
        exceptions = {
            "n02012849": wn.synset("crane.n.05"),
            "n03126707": wn.synset("crane.n.04"),
            "n03710637": wn.synset("maillot.n.02"),
            "n03710721": wn.synset("maillot.n.01"),
        }
        word1 = exceptions.get(class_name_1, wn.synset(f"{class_name_1}.n.01"))
        word2 = exceptions.get(class_name_2, wn.synset(f"{class_name_2}.n.01"))
        return getattr(word1, self.similarity_type)(word2)


class TransformSimlarity:
    def __init__(self, similarity_type="spearman_rank", union_top_n=25):
        self.similarity_type = similarity_type
        self.union_top_n = union_top_n

    @property
    def name(self):
        if self.similarity_type == "spearman_rank":
            return self.similarity_type
        return f"{self.similarity_type}_top_{self.union_top_n}"

    def __call__(self, class_name_1: str, class_name_2: str, df: pd.DataFrame) -> float:
        return getattr(self, self.similarity_type)(class_name_1, class_name_2, df)

    def intersection_over_union(
        self, class_name_1: str, class_name_2: str, df: pd.DataFrame
    ) -> float:
        df_top_1 = (
            df[df["class_name"] == class_name_1]
            .sort_values("avg_percent_similarity_change", ascending=False)
            .head(self.union_top_n)
        )
        df_top_2 = (
            df[df["class_name"] == class_name_2]
            .sort_values("avg_percent_similarity_change", ascending=False)
            .head(self.union_top_n)
        )
        top_1_set = set(df_top_1["transform"])
        top_2_set = set(df_top_2["transform"])

        intersection = len(top_1_set.intersection(top_2_set))
        union = len(top_1_set.union(top_2_set))
        return float(intersection) / union

    def spearman_rank(
        self, class_name_1: str, class_name_2: str, df: pd.DataFrame
    ) -> float:
        """Computes Spearman's rank correlation for transforms per class.
        Transforms are ranked by % similarity change per class"""
        df_top_1 = df[df["class_name"] == class_name_1].sort_values(
            "avg_percent_similarity_change", ascending=False
        )
        df_top_2 = df[df["class_name"] == class_name_2].sort_values(
            "avg_percent_similarity_change", ascending=False
        )
        ranks_1, ranks_2 = self._build_rank_arrays(df_top_1, df_top_2)
        return spearmanr(ranks_1, ranks_2).correlation

    def _build_rank_arrays(
        self, df_top_1: pd.DataFrame, df_top_2: pd.DataFrame
    ) -> Tuple[List[int], List[int]]:
        ranks_1 = []
        ranks_2 = []

        for transform in df_top_1["transform"].unique():
            ranks_1.append(df_top_1["transform"].values.tolist().index(transform))
            ranks_2.append(df_top_2["transform"].values.tolist().index(transform))
        return ranks_1, ranks_2


def plot_violin(
    df,
    similarity_threshold=0.75,
    dissimilarity_threshold=None,
    class_similarity_type="wup_similarity",
    transform_similarity_type="spearman_rank",
):
    df = df.copy()
    is_similar = df[class_similarity_type] > similarity_threshold
    dissimilarity_threshold = (
        (1 - similarity_threshold)
        if dissimilarity_threshold is None
        else dissimilarity_threshold
    )
    is_dissimilar = df[class_similarity_type] < (dissimilarity_threshold)
    df["similarity"] = is_similar.apply(lambda x: "similar" if x else "dissimilar")
    df = df[(is_similar) | (is_dissimilar)]
    fig = px.violin(
        df,
        x=transform_similarity_type,
        color="similarity",
        color_discrete_map={"similar": "red", "dissimilar": "blue"},
        title=f"wordnet {class_similarity_type} ({similarity_threshold:0.2f} similarity threshold)",
    )
    return fig


def plot_binned_violin(
    results_df,
    wordnet_similarity_type="wup_similarity",
    transform_similarity_type="spearman_rank",
):

    _, bins = pd.cut(results_df[wordnet_similarity_type], 10, retbins=True)
    fig = px.violin(
        results_df,
        y=transform_similarity_type,
        x=pd.cut(results_df[wordnet_similarity_type], 10, labels=bins[1:]),
    )

    fig.update_xaxes(title=f"Wordnet Similarity <br> ({wordnet_similarity_type} bins)")
    fig.update_yaxes(
        title=f"Transformation Similarity <br> ({transform_similarity_type})"
    )
    return fig


if __name__ == "__main__":
    print("success")
