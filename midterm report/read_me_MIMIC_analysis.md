# MIMIC-IV analysis script info
This folder contains the 2 python scripts I used and the specific csvs from MIMIC-IV that gave me comparison data for our midterm report.
# CSV files:
Files I created from scripts:
*'cancer patient dates.csv': I created this initial file to get patient ids who were pancreatic cancer positive from the mimic_data.json file. I didn't know pandas' .to_csv method produces some incomprehensible (for me) readouts. Nor did I consider multiple cancer diagnoses when making this csv, hence the repeat ids.
*'cancer patient records.csv': The better cancer-positive patient data csv. Upon realizing my mistake, I rewrote the csv.

# Python scripts:
*'analysis.py': My first script for finding the length of datasets (I used an IDE (Pycharm) so explicit running of len(dataset) was unnecessary).
*'analysis2.py': Continuation of the MIMIC subset data from analysis.py. 
# Notes:
* Commented out calculations I already had. Current versions are not stand-alone.
