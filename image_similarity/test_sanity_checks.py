import pytest
from image_similarity import sanity_checks
from hydra.experimental import compose, initialize


@pytest.fixture
def cfg():
    """Returns `test_config.yaml` using hydra"""
    with initialize(config_path="."):
        # config is relative to a module
        cfg = compose(
            config_name="config_test",
        )
    return cfg


def test_sanity_checks(cfg):
    """Integration test to ensure main launcher works"""
    sanity_checks.main(cfg)


@pytest.mark.parametrize("similarity_type", ["resnet18", "resnet18_no_aug"])
def test_run_within_v_across(cfg, similarity_type):
    image_net_dir = sanity_checks.get_image_net_dir()
    class_labels = sanity_checks.get_class_labels(cfg, image_net_dir)

    across_v_within = sanity_checks.run_within_v_across(
        image_net_dir,
        class_labels,
        similarity_type,
        max_n_image_pairs=cfg.sanity_checks.max_n_image_pairs,
        max_class_label_combinations=cfg.sanity_checks.max_class_label_combinations,
        randomly_rotate=False,
    )
    assert "avg within" in across_v_within
    assert "std within" in across_v_within
    assert across_v_within["std within"] > 0.0


@pytest.mark.parametrize("similarity_type", ["resnet18", "resnet18_no_aug"])
def test_run_transformed_similarities(cfg, similarity_type):
    image_net_dir = sanity_checks.get_image_net_dir()
    class_labels = sanity_checks.get_class_labels(cfg, image_net_dir)

    transformed_similarities = sanity_checks.run_transformed_similarities(
        image_net_dir,
        class_labels,
        similarity_type,
        max_n_image_pairs=cfg.sanity_checks.max_n_image_pairs,
    )

    assert "avg for small rotation" in transformed_similarities
    assert "std for small rotation" in transformed_similarities
    assert transformed_similarities["std for large rotation"] > 0.0


@pytest.mark.parametrize("similarity_type", ["resnet18", "resnet18_no_aug"])
def test_run_within_v_across_indep_transformed(cfg, similarity_type):
    image_net_dir = sanity_checks.get_image_net_dir()
    class_labels = sanity_checks.get_class_labels(cfg, image_net_dir)
    similarities = sanity_checks.run_within_v_across_indep_transformed(
        image_net_dir,
        class_labels,
        similarity_type,
        max_n_image_pairs=cfg.sanity_checks.max_n_image_pairs,
        max_class_label_combinations=cfg.sanity_checks.max_class_label_combinations,
    )
    assert "avg transformed across" in similarities
    assert similarities["std transformed within"] > 0.0
