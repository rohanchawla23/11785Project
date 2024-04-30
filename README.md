## Baseline Implementation
Cancer Risk Net fork: https://github.com/jesse12shen/CancerRiskNet

### PanCan-Prediction-Mamba Implementation
The dataset is loaded and preprocessed in the same way that it was for our implementation of CancerRiskNet. However, in this model we use the dataset to train a selective structured state space model (aka Mamba) for various epochs. Training and analysis was performed in a google Colab environment using an NVIDIA A100 GPU with High-RAM enabled. This particular notebook was formatted to work within my own google drive, so the file locations for the datasets, ICD diagnostic code descriptions, etc., will need to be updated to match your local running environment. These files can be found in the `data` folder.
