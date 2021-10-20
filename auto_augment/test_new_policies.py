from auto_augment import new_policies, best_policies


def test_to_scale():
    assert best_policies.to_scale(0) == 1.0
    assert best_policies.to_scale(5) < 1.0
    assert best_policies.to_scale(6) > 1.0


def test_imagenet_plus_geometric():
    imagenet_geo = new_policies.ImageNetPlusGeometric()
    single_transforms = imagenet_geo.get_unique_single_transformations()
    assert len(single_transforms) > 1