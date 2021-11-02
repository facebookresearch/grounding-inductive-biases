"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

"""
Computes equivariance and invariance
"""
import plotly.graph_objects as go
from equivariance_measure.embedding_alignments import AlignmentsDigest
from typing import List
from itertools import cycle

GEOMETRIC = {
    "shearX",
    "shearY",
    "translateX",
    "translateY",
    "rotate",
    "scale_zoom_in",
    "scale_zoom_out",
}


def plot_equivariance(
    alignments: AlignmentsDigest,
    include_geo: bool = True,
    include_appearance: bool = False,
) -> go.Figure:
    """Plot IQR of distributions"""
    # tiny marker size to not show outliers
    marker_size = 0.01
    fig = go.Figure()
    x = generate_x(alignments.magnitudes)

    for transform_name in alignments.transform_names:
        if not is_valid_transform(transform_name, include_geo, include_appearance):
            continue
        y = get_transform_points(alignments, transform_name)
        fig.add_trace(
            go.Box(
                y=y,
                x=x,
                name=transform_name,
                marker_size=marker_size,
            )
        )

    fig.update_layout(
        xaxis=dict(title="magnitude", zeroline=False),
        yaxis=dict(title="equivariance"),
        title=f"Equivariance {alignments.data_stage} set",
        boxmode="group",
    )

    return fig


def generate_x(magnitudes: List[str], steps: int = 100) -> List[str]:
    """Returns a list of magnitudes for each point based on num steps"""
    x = []

    for magnitude in magnitudes:
        x.extend([magnitude] * steps)
    return x


def is_valid_transform(
    transform_name: str, include_geo: bool, include_appearance: bool
):
    is_geo = transform_name in GEOMETRIC
    if is_geo and include_geo:
        return True
    elif not is_geo and include_appearance:
        return True
    return False


def get_transform_points(
    alignments: AlignmentsDigest, transform_name: str
) -> List[float]:
    y = []
    for magntidue in alignments.magnitudes:

        y_for_magnitude = [
            alignments.get_percentile(transform_name, magntidue, p) for p in range(101)
        ]
        y.extend(y_for_magnitude)
    return y