# ResNet18 Image Similarity


## Sanity Check Experiments
Commands below should be run within the `image-similarity` directory.

Launch on cluster
```
$ python sanity_checks.py --multirun
```

Run locally
```
$ python sanity_checks.py hydra/launcher=submitit_local 
```

Run tests
```
python -m pytest test_sanity_checks.py
```