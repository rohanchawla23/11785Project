import datetime

import matplotlib.pyplot as plt
# import seaborn
import random
import math
import pickle
import json
import numpy as np
import pandas as pd
import datetime as dt
# from collections import defaultdict
# from imblearn.over_sampling import SMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from imblearn.pipeline import Pipeline as ImbPipeline
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import average_precision_score
import warnings
# warnings.filterwar
# nings('ignore')
# pat_d = pd.read_csv('patients.csv')
# total = len(pat_d.gender)
# t_male = pat_d[pat_d.gender=='M']
# t_fem = pat_d[pat_d.gender=='F']
def st_transf(s:str):
    '''
    For converting the feature vectors into a list of ints and then summing
    those
    :param s: str from the csv that can be cast as a list of ints
    :return collapse: int summed of an int list
    '''
    s_li = s.split(', ')
    s_li[0] = s_li[0][1]
    s_li[-1] = s_li[-1][0]
    i_s = [int(k) for k in s_li]
    collapse = np.sum(i_s)
    return collapse
can_d = pd.read_csv('MIMIC.csv',
                    # dtype = {'Unnamed: 0':int,
                    #                       'Procedures_ICD_Features':int,
                    #                       'Medication_NDC_Features':int},
                    converters={'Procedures_ICD_Features': st_transf,
                                'Medication_NDC_Features': st_transf} )

j_records = pd.read_json('mimic_data.json')
year_data = pd.read_csv('patients.csv')
# can_d1 = can_d.copy()
c_pos = can_d[can_d.PANCAN == 1]
key_list = c_pos.PATIENT_KEY.array
c_pat_rec = j_records[key_list[0]]
pan_can_code = 'C25'
disease_codes = pd.read_csv('d_icd_diagnoses.csv')
patient_rec = c_pat_rec.copy()
patient_rec.index = [c_pat_rec.index[0],'date of diagnosis', c_pat_rec.index[2]]
d_date = None
d_age = np.empty(len(c_pos), dtype=datetime.timedelta)
for j in range(1,len(key_list)):
    if key_list[j] in j_records:
        c_pat_rec = pd.concat([c_pat_rec, j_records[key_list[j]]],axis=1)

        # per patient
        event_list = j_records[key_list[j]].events
        n_diseases = 0
        for k in range(len(event_list)):

            if pan_can_code in event_list[k]['codes']:
                d_date = event_list[k]['admdate'].split('-')
                d_date = dt.date(int(d_date[0]), int(d_date[1]), int(d_date[2]))
                b_date = j_records[key_list[j]].birthdate.split('-')
                b_date = dt.date(int(b_date[0]), int(b_date[1]), int(b_date[2]))
                d_age[j] = d_date - b_date
        patient_rec = pd.concat([patient_rec, j_records[key_list[j]]],
                                        axis=1)
# year calculation #

## (Simpler) cancer demographics ##
# total = len(can_d1)
# comp_med = can_d1.Procedures_ICD_Features.median() + can_d1.Medication_NDC_Features.median()
# n_can = np.sum(can_d.PANCAN)
# t_fem = can_d[can_d.GENDER == 1] # 1 is female based on cross-referencing
# patients.csv
# c_pos = can_d[can_d.PANCAN == 1]
# c_total = len(c_pos)
# c_p_m = len(c_pos[c_pos.GENDER == 0])
# c_p_f_m = c_p_m/c_total
# disease_code_c_med = c_pos.Procedures_ICD_Features.median() + \
#         c_pos.Medication_NDC_Features.median()
# c25 is the general catch-all for pancan in the codes
# c_p_f = len(c_pos[c_pos.GENDER == 1])
# c_p_f_p = c_p_f/c_total

print('debug draft')