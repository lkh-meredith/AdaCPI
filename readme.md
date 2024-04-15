# AdaCPI

This is code of our paper "Adaptive Cross-modal Prompt Interaction with Optimal Layer Search for Vision-Language Model".

## Install

* Build conda enviroment

  ```
  #create a conda environment
  conda create -y -n AdaCPI python=3.9

  # Activate the environment
  conda activate AdaCPI

  #Install torch and torchvision (refer to https://pytorch.org)
  pip install torch==2.1.2+cu118 torchaudio==2.1.2+cu118 torchvision==0.16.2+cu118
  ```
* Install `dassl` enviroment. Follow the instructions described [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation) to install it.
* Clone this project and install requirements

  ```
  pip install -r requirements.txt
  ```

## Download Dataset:

Please refer to CoOp's [DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) file to download the corresponding eight datasets in our paper, and put them into the `data/` file.

## Train and Evaluate

Please revise the dataset path in the configure file `configs/trainer/AdaCPI` and run the commands from `the scripts/AdaCPI/`.

For example, the training and evaluation commands for AdaCPI on Caltech101 as follows.

### Few-shot learning setting

1 shot: `bash scripts/AdaCPI/main.sh caltech101 1`

2 shot: `bash scripts/AdaCPI/main.sh caltech101 2`

4shot: `bash scripts/AdaCPI/main.sh caltech101 4`

8 shot: `bash scripts/AdaCPI/main.sh caltech101 8`

16shot: `bash scripts/AdaCPI/main.sh caltech101 16`

After the experiments are finished, you can use `parse_test_res.py` to calculate the average results instead of mannually looking into the log files. For the experiments of 1 shot, the file structure of `output/` is

```
output
|-- AdaCPI/
|    |-- caltech101/
|    |    |-- 1shots/
|    |    |    |-- seed1/
|    |    |    |-- seed2/
|    |    |    |-- seed3/
```

To calculate the average results for this folder, you can run

`python parse_test_res.py output/AdaCPI/caltech101/1shots`

### Base-to-new generalization setting

To reproduce this setting's experiments, you can run `scripts/AdaCPI/base2new_train.sh` and `scripts/AdaCPI/base2new_test.sh`, respectively. The former trains a AdaCPI model on the base classes while the latter evaluates the trianed AdaCPI on the novel classes. Both scripts have three input arguments, i.e., `DATASET`, `LAMBDA` and `EPOCH`. Below we provide an example on how to evaluate the model on Caltech101.

```
bash scripts/AdaCPI/base2new_train.sh caltech101 1.0 20
bash scripts/AdaCPI/base2new_test.sh caltech101 1.0 20
```

When the evaluation is done, you can also use `python parse_test_res.py` to calculate the average results. Afer the expriments above, you would get

```
output
|–– AdaCPI/
|   |–– base2new/
|   |   |–– train_base/
|   |   |   |–– caltech101/
|   |   |   |   |–– 16shots/
|   |   |   |   |   |–– seed1/
|   |   |   |   |   |–– seed2/
|   |   |   |   |   |–– seed3/
|   |   |–– test_new/
|   |   |   |–– caltech101/
|   |   |   |   |–– 16shots/
|   |   |   |   |   |–– seed1/
|   |   |   |   |   |–– seed2/
|   |   |   |   |   |–– seed3/
```

Then, to get the average performance on the base classes, run

`python parse_test_res.py output/AdaCPI/base2new/train_base/caltech101/16shots`

To get the average performance on the new classes, run

`python parse_test_res.py output/AdaCPI/base2new/test_new/caltech101/16shots`

## Acknowledgements

Our code is based on CoOp and MaPLe repository. We thank the authors for releasing their code.
