"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

from _pytest.fixtures import wrap_function_to_error_out_if_called_directly
import pytest
from wordnet_analysis import wordnet_correlation
from wordnet_analysis.wordnet_correlation import (
    WordNetSimilarity,
    WordNetTransformCorrelations,
    TransformSimlarity,
)


@pytest.mark.parametrize("similarity_type", wordnet_correlation.WORD_NET_SIMILARITIES)
def test_wordnet_sim(similarity_type):
    # skip slow one
    if similarity_type != "lch_similarity":
        sim = WordNetSimilarity(similarity_type=similarity_type)
        assert sim("dog", "cat") > sim("dog", "paintbrush")


@pytest.fixture
def wordnet_transform_correlations():
    wordnet_transform_correlations = WordNetTransformCorrelations(
        wordnet_correlation.WORD_NET_SIMILARITIES,
        ["spearman_rank", "intersection_over_union"],
        max_class_pairs=10,
    )
    return wordnet_transform_correlations


class TestWordNetTransformCorrelations:
    def test_load_df(self, wordnet_transform_correlations):
        assert len(wordnet_transform_correlations.df) > 1000
        assert "class_name" in wordnet_transform_correlations.df.columns

    def test_spearman_rank(self, wordnet_transform_correlations):
        sim = TransformSimlarity(similarity_type="spearman_rank")
        assert sim("Eskimo_dog", "Persian_cat", wordnet_transform_correlations.df) > 0.5

    def test_intersection_over_union(self, wordnet_transform_correlations):
        sim = TransformSimlarity(
            similarity_type="intersection_over_union", union_top_n=25
        )
        assert sim("Eskimo_dog", "Persian_cat", wordnet_transform_correlations.df) > 0.5

    def test_run(self, wordnet_transform_correlations):
        results_df = wordnet_transform_correlations.run()
        assert "spearman_rank" in results_df.columns
        assert len(results_df) == wordnet_transform_correlations.max_class_pairs
