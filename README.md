# GATS: A Time-Series Dataset for Addressing General Aviation Flight Safety 

This repository is the official implementation of [GATS: A Time-Series Dataset for Addressing General Aviation Flight Safety ]. 

## Requirements

To install requirements:

```setup
conda env create -f env.yml
```

## Training

We provide several benchmark tasks that can be be reproduced.

To train the autoencoder for missing data reconstruction, run the command:

```train
python benchmarks.autoencoder.train_autoencoder.py --train_data_dir <path_to_training_data> --val_data_dir <path_tovalidation_data> --job_name <my_job_name>
```

For running masked column regression with SimCLR, run:
```
python -m benchmarks.simclr_regression.regression -n 'My Job Name'  -m <simclr model path> -e <n epochs> -g <cuda GPU address (e.g. 'cuda:0')>
```

To run aircraft classification with ConvMHSA, run:
```
python -m benchmarks.conv_mhsa.train -e <n epochs> -n 'My Job Name' -l 1e-5 -g <cuda GPU address (e.g. 'cuda:0')>
```

For running aircraft classification with SimCLR, run:
```
python -m benchmarks.simclr_classifier.classifier -m <simclr model path> -n 'My Job Name' -e <n epochs> -g <cuda GPU address (e.g. 'cuda:0')>
```

## Evaluation

To evaluate the autoencoder for missing data reconstruction, run the command:
```eval
python benchmarks.autoencoder.test_autoencoder --data_dir <path_to_testing_data> --model_path <path_to_trained_model> --norm_params_path <path_to_normalization_params_from_training>
```

To evaluate a masked column regression with SimCLR, run:
```
python -m benchmarks.simclr_regression.regression -m <model path> -E -r <mask ratio> -M <mask length>
```

To evaluate an aircraft classification ConvMHSA model, run:
```
python -m benchmarks.conv_mhsa.train -m <model_path>
```

## Results

Our model achieves the following performance on the provided preprocessed data:

### Missing Data Reconstruction
| Model name         | Mean Absolute Error  | Mean Squared Error |
| ------------------ |---------------- | -------------- |
| Masked Autoencoder  |     0.46        |      0.62      |
| SimCLR + Classification Head   |    4.44       |      25.50  |

### Airframe Model Classification
| Model name         | Accuracy |
| ------------------ | -------------- |
| ConvMHSA  |     0.99      |
| SimCLR + Classification Head  |    0.82  |

### Airframe Class Classification
| Model name         | Accuracy |
| ------------------ | -------------- |
| ConvMHSA  |     1.00      |
| SimCLR + Classification Head  |    0.30  |


## Contributing

This code is under the MIT License. Please refer to LICENSE.txt for more information.
