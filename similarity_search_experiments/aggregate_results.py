"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import json
import os
from pathlib import Path
import yaml
import pandas as pd
from typing import List, Optional


def aggregate_results(
    logs_dir: Path,
    results_dir: Optional[Path],
    file_name_prefix: List[str] = ["single_transform_boosts", "subpolicy_boosts"],
    file_type: str = "csv",
):
    """Aggregates dataframes from parallel runs"""
    file_names = find_file_names(logs_dir, file_name_prefix, file_type)
    dfs = [pd.DataFrame() for file_name in file_names]

    for dir_path in logs_dir.iterdir():
        if is_digit_directory(dir_path):
            for i, file_name in enumerate(file_names):
                dir_df = pd.read_csv(dir_path / file_name, index_col=0)
                dfs[i] = dfs[i].append(dir_df, ignore_index=True)
    # save aggregate
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)
    for df, name in zip(dfs, file_names):
        df.to_csv(logs_dir / name)

        if results_dir:
            save_path = results_dir / name
            df.to_csv(save_path)
    # save original log source for reference or debuggging
    if results_dir:
        record_original_logs(file_names, logs_dir, results_dir)


def record_original_logs(
    file_names: List[str], logs_dir: Path, results_dir: Path
) -> None:
    """Records the original logs directory for each new result file in file_names"""
    logs_json = dict()
    logs_file = results_dir / "original_logs.json"
    # load existing
    if logs_file.exists():
        with open(logs_file) as json_file:
            logs_json = json.load(json_file)
    # update
    for file_name in file_names:
        logs_json[file_name] = str(logs_dir)
    # save
    with open(logs_file, "w") as json_file:
        json.dump(logs_json, json_file)


def find_file_names(
    logs_dir: Path,
    file_name_prefixes: List[str] = ["single_transform_boosts", "subpolicy_boosts"],
    file_type: str = "csv",
) -> List[str]:
    file_name_prefixes = tuple(file_name_prefixes)
    file_names = set()
    for dir_path in logs_dir.iterdir():
        if is_digit_directory(dir_path):
            for file in dir_path.iterdir():
                if file.name.startswith(file_name_prefixes) and file.suffix.endswith(
                    file_type
                ):
                    file_names.add(file.name)
    return list(file_names)


def is_digit_directory(dir_path: Path):
    """Returns true if directory name is made of digits"""
    if str(dir_path.name).isdigit():
        return True
    return False


def get_experiment_name(dir_path: Path) -> str:
    """Returns experiment name from Hydra logs"""
    name = list((dir_path / "0").glob("*.log"))[0].stem
    return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate results")
    parser.add_argument(
        "logs_dir",
        type=Path,
        help="directory containing logs with results",
    )
    user = os.environ.get("USER")
    parser.add_argument(
        "--results_dir",
        type=Path,
        help="directory to store aggregated result",
        default=Path(
            f"/checkpoint/{user}/Real-Data-Equivariance/results/similarity-search/"
        ),
    )
    args = parser.parse_args()
    subdir = get_experiment_name(args.logs_dir)
    aggregate_results(args.logs_dir, args.results_dir / subdir)
