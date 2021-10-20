"""
Computes equivariance and invariance
"""
import os
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from equivariance_measure.embedding_distances import InvariancesDigest
from typing import List, Dict
from functools import cache
import plotly.express as px

GEOMETRIC = {
    "shearX",
    "shearY",
    "translateX",
    "translateY",
    "rotate",
    "scale_zoom_in",
    "scale_zoom_out",
}


MODEL_TO_COLOR = {
    "Trained ViT (L/16)": px.colors.sequential.Turbo[-2],
    "Untrained ViT (L/16)": px.colors.sequential.Turbo[-4],
    "Trained ResNet18": px.colors.sequential.Turbo[1],
    "Trained w/o Aug ResNet18": px.colors.sequential.Turbo[2],
    "Untrained ResNet18": px.colors.sequential.Turbo[3],
}


@cache
def load_invariances_per_model(
    data_stage: str, distance_type: str
) -> List[InvariancesDigest]:
    parent_results_dir = ""
    invariances = []

    for model in load_vit(parent_results_dir, data_stage, distance_type):
        invariances.append(model)

    for model in load_resnet(parent_results_dir, data_stage, distance_type):
        invariances.append(model)

    return invariances


def load_vit(
    parent_results_dir: str, data_stage: str, distance_type: str, n_batches=100
) -> List[InvariancesDigest]:
    results_dir = os.path.join(
        parent_results_dir, f"invariance_measure_vit_{n_batches}_batches"
    )
    invariances_trained = InvariancesDigest(
        distance_type=distance_type,
        data_stage=data_stage,
        model_name="Trained ViT (L/16)",
    )
    invariances_trained.load(results_dir)
    invariances_untrained = InvariancesDigest(
        distance_type=distance_type,
        data_stage=data_stage,
        prefix="untrained_",
        model_name="Untrained ViT (L/16)",
    )
    invariances_untrained.load(results_dir)
    return [invariances_trained, invariances_untrained]


def load_resnet(
    parent_results_dir: str, data_stage: str, distance_type: str, n_batches=300
):
    results_dir = os.path.join(
        parent_results_dir, f"invariance_measure_{n_batches}_batches"
    )
    invariances_trained = InvariancesDigest(
        distance_type=distance_type,
        data_stage=data_stage,
        model_name="Trained ResNet18",
    )
    invariances_no_aug = InvariancesDigest(
        distance_type=distance_type,
        data_stage=data_stage,
        prefix="no_aug_",
        model_name="Trained w/o Aug ResNet18",
    )
    invariances_untrained = InvariancesDigest(
        distance_type=distance_type,
        data_stage=data_stage,
        prefix="untrained_",
        model_name="Untrained ResNet18",
    )
    invariances = [invariances_trained, invariances_untrained, invariances_no_aug]
    for invariance in invariances:
        invariance.load(results_dir)
    return invariances


def plot_transform_invariance_across_models(
    data_stage: str = "train",
    distance_type: str = "cos_distance",
    include_geo: bool = True,
    include_appearance: bool = False,
) -> Dict[str, go.Figure]:
    invariances_per_model = load_invariances_per_model(data_stage, distance_type)

    transform_to_figure = dict()

    for transform_name in invariances_per_model[0].transform_names:
        if not is_valid_transform(transform_name, include_geo, include_appearance):
            continue
        fig = plot_single_transform_invariance_across_models(
            transform_name, invariances_per_model, data_stage, distance_type
        )
        transform_to_figure[transform_name] = fig
    return transform_to_figure


def plot_single_transform_invariance_across_models(
    transform_name: str,
    invariances_per_model: List[InvariancesDigest],
    data_stage: str,
    distance_type: str,
    evens_only: bool = False,
    legend: bool = True,
):
    marker_size = 0.01
    fig = go.Figure()
    x = generate_x(invariances_per_model[0].magnitudes)
    if evens_only:
        x = generate_x_evens(invariances_per_model[0].magnitudes)

    for invariances in invariances_per_model:
        y = get_transform_points(invariances, transform_name)
        if evens_only:
            y = get_transform_points_evens(invariances, transform_name)
        fig.add_trace(
            go.Box(
                y=y,
                x=x,
                name=invariances.model_name,
                marker_size=marker_size,
                marker_color=MODEL_TO_COLOR[invariances.model_name],
            )
        )

    # Change the bar mode
    fig.update_layout(
        xaxis=dict(title="Magnitude", zeroline=False),
        yaxis=dict(title="Invariance"),
        title=f"{transform_name} invariance {distance_type} {data_stage} set",
        boxmode="group",
        showlegend=legend,
    )
    fig.update_yaxes(range=[-0.5, 1.0])
    return fig


def plot_invariance(
    invariances: InvariancesDigest,
    include_geo: bool = True,
    include_appearance: bool = False,
) -> go.Figure:
    """Plots median of distributions"""
    marker_size = 0.01
    fig = go.Figure()
    x = generate_x(invariances.magnitudes)

    for transform_name in invariances.transform_names:
        if not is_valid_transform(transform_name, include_geo, include_appearance):
            continue
        y = get_transform_points(invariances, transform_name)
        fig.add_trace(
            go.Box(
                y=y,
                x=x,
                name=transform_name,
                marker_size=marker_size,
            )
        )

    # Change the bar mode
    fig.update_layout(
        xaxis=dict(title="magnitude", zeroline=False),
        yaxis=dict(title="Invariance"),
        title=f"Invariance {invariances.distance_type} {invariances.data_stage} set",
        boxmode="group",
    )
    return fig


def plot_invariance_baseline(
    invariances: InvariancesDigest,
    include_geo: bool = True,
    include_appearance: bool = False,
) -> go.Figure:
    marker_size = 0.01
    fig = go.Figure()
    x = generate_x(invariances.magnitudes)

    for transform_name in invariances.transform_names:
        if not is_valid_transform(transform_name, include_geo, include_appearance):
            continue
        y = get_baseline_points(invariances, transform_name)
        fig.add_trace(
            go.Box(
                y=y,
                x=x,
                name=transform_name,
                marker_size=marker_size,
            )
        )

    # Change the bar mode
    fig.update_layout(
        xaxis=dict(title="magnitude", zeroline=False),
        yaxis=dict(title="Baseline"),
        title=f"Baseline Invariance {invariances.distance_type} {invariances.data_stage} set",
        box_mode="group",
    )
    return fig


def get_baseline_points(invariances, transform_name):
    y = []
    for magntidue in invariances.magnitudes:
        y_for_magnitude = [
            invariances.get_baseline_percentile(transform_name, magntidue, p)
            for p in range(101)
        ]
        y.extend(y_for_magnitude)
    return y


def generate_x(magnitudes: List[str], steps: int = 100) -> List[str]:
    """Returns a list of magnitudes for each point based on num steps"""
    x = []

    for magnitude in magnitudes:
        x.extend([magnitude] * steps)
    return x


def generate_x_evens(magnitudes: List[str], steps: int = 100) -> List[str]:
    x = []

    for magnitude in magnitudes[::2]:
        x.extend([magnitude] * steps)
    return x


def get_transform_points(
    invariances: InvariancesDigest, transform_name: str
) -> List[float]:
    y = []
    for magntidue in invariances.magnitudes:
        y_for_magnitude = [
            invariances.get_percentile(transform_name, magntidue, p) for p in range(100)
        ]
        y.extend(y_for_magnitude)
    return y


def get_transform_points_evens(
    invariances: InvariancesDigest, transform_name: str
) -> List[float]:
    y = []
    for magntidue in invariances.magnitudes[::2]:
        y_for_magnitude = [
            invariances.get_percentile(transform_name, magntidue, p) for p in range(100)
        ]
        y.extend(y_for_magnitude)
    return y


def is_valid_transform(
    transform_name: str, include_geo: bool, include_appearance: bool
):
    is_geo = transform_name in GEOMETRIC
    if is_geo and include_geo:
        return True
    elif not is_geo and include_appearance:
        return True
    return False
