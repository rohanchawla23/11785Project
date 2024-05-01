## Baseline Implementation
Cancer Risk Net fork: https://github.com/jesse12shen/CancerRiskNet

### PanCan-Prediction-Mamba Implementation
The dataset is loaded and preprocessed in the same way that it was for our implementation of CancerRiskNet. However, in this model we use the dataset to train a selective structured state space model (aka Mamba) for various epochs. Training and analysis was performed in a google Colab environment using an NVIDIA A100 GPU with High-RAM enabled. This particular notebook was formatted to work within my own google drive, so the file locations for the datasets, ICD diagnostic code descriptions, etc., will need to be updated to match your local running environment. These files can be found in the `data` folder. To run the notebook, simply run cells in sequence. The args variable contains most relevant hyperparameters, and the code automatically graphs train/val losses and AUROCs.

To produce `mimic_data.json`, we used the original MIMIC-IV dataset: https://drive.google.com/drive/folders/1XqxW2jY1IbAWpprwevzeRbGy3nAuPWhW?usp=sharing. This data can also be accessed from https://mimic.mit.edu/docs/gettingstarted/. Unfortunately, the dataset is too large to include in this repository directly. We used two tables within the `hosp` folder of this dataset to create our JSON file: `d_icd_diagnoses.csv` and `admissions.csv`.
