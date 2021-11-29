

# Installation

To enable importing of modules, from the parent directory execute:

`pip install -e .`


To install requirements: `python -m pip install requirements.txt`

To save requirements: `python -m pip list --format=freeze --exclude-editable -f https://download.pytorch.org/whl/torch_stable.html > requirements.txt`

* Note we use Python 3.9.4 for our experiments

The code in augerino_lib is an extension (with our modification) from the original Augerino code https://github.com/g-benton/learning-invariances. The code in data_augmentation/my_training.py is a modification of the pytorch example in https://github.com/pytorch/examples/tree/master/imagenet.

# Running the code 

For the experiments of Section 2:

To train the model update the appropriate `result_dir` (folder where to save the results) and `data` (your path to ImageNet data) in `config_test_ddp_local.yaml` and run `python hydra_app_local.py`.
To test the model update the appropriate `result_dir` (folder where the trained models are) and `job_id` (id of the folder for a specific trained model you want to test) in `config_test_local.yaml` and run `python hydra_test_local.py`.


For remaining experiments: 

Navigate to the corresponding directory, then execute: `python run.py -m` with the corresponding `config.yaml` file (which stores experiment configs).

# License

Consult License.md
