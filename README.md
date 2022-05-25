# A Spectral Sequence-Based Nonlocal Long Short-Term Memory Network for Hyperspectral Image Classification

## Paper
[A Spectral Sequence-Based Nonlocal Long Short-Term Memory Network for Hyperspectral Image Classification](https://ieeexplore.ieee.org/document/9736454)

## Requirements
This tool is implemented under the following environment:
 - Tensorflow 1.10.0
 - Keras 2.2.0
 - GeForce GTX 1080 Ti

## Usage
```
pip install spectral
pip install opencv-python
pip install scikit-learn
```
Run `six_spectral.py`

## Note
```
CellRNN
├── module
│    ├── R.py
│    └── F.py
├── ___prepareSpectral___.py
├── ConfigSetting.py
├── AAA.py
├── IndianPines
│    ├── Indian_pines.mat
│    ├── Indian_pines_gt.mat
├── HoustonU
│    ├── HoustonU_gt.mat
│    └── HoustonU.mat
├── PaviaU
│    ├── Pavia_gt.mat
│    └── Pavia.mat
└── construct
    ├── train_data_H_new.mat
    └── test_data_H_new.mat
```
The directory tree listed as above. ___prepareSpectral___.py is used to finish the data preparation.
Be sure to modify the path variable under ConfigSetting.py to reflect your local environment.
