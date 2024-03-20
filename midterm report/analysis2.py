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
# c_pos = can_d[can_d.PANCAN == 1]
# key_list = c_pos.PATIENT_KEY.array
# c_pat_rec = j_records[j_records.columns[0]]
# c_pat_rec = pd.read_csv('cancer patient records.csv',index_col=0)
#                         # ,dtype={'columns': int}))
#                         # converters=)
pan_can_code = 'C25'
disease_codes = pd.read_csv('d_icd_diagnoses.csv')
# patient_rec = c_pat_rec[c_pat_rec.columns[0]]
patient_rec = pd.read_csv('cancer patient dates.csv', index_col=0)
# to get patient_rec data:
# for m in range(1,len(c_pat_rec.columns)):
#     try:
#         int(c_pat_rec.columns[m])
#     except:
#         pass
#     else:
#         patient_rec = pd.concat([patient_rec, c_pat_rec[c_pat_rec.columns[
#             m]]],
#                                       axis=1)

# patient_rec.index = [c_pat_rec.index[0],'date_of_diagnosis',
# c_pat_rec.index[2]] # saving and reading csv seems to change the indices
# diag_date = pd.DataFrame(data=np.empty((1,len(patient_rec.columns)),
#                                     dtype=dt.datetime),
#                          index=['date_of_diagnosis'],
# columns=patient_rec.columns)
# patient_rec = pd.concat([patient_rec, diag_date],axis=0)

# diseases = np.zeros(len(patient_rec.columns))
# event_list = j_records[j_records.columns[0]].events
# cancer patients only: # commented out because 0 is not the first patient
# column
def event_sort(event):
    return event['admdate']
j_records[int(patient_rec.columns[0])].events.sort(key=event_sort)
event_list = j_records[int(patient_rec.columns[0])].events
n_diseases = 0
# traj = np.empty((len(c_pat_rec.columns)),dtype=dt.timedelta)
last_event = event_list[-1]['admdate'].split('-')
last_event = dt.date(int(last_event[0]), int(last_event[1]), int(last_event[2]))
start = event_list[0]['admdate'].split('-')
start = dt.date(int(start[0]), int(start[1]), int(start[2]))
traj = [last_event - start]
timewise_n_dis = np.zeros((4,len(patient_rec.columns))) #0: 12+, 1: 6-12,
# 2: 3-6, 3: <3
thresholds = [dt.timedelta(days=365), dt.timedelta(days=183), dt.timedelta(
    days=92)]
# all_disease = np.empty(len(patient_rec.columns),dtype=bool)
for k in range(len(event_list)):
    if event_list[k]['codes'] in disease_codes.icd_code.values:
        n_diseases += 1
    cancer_p = False
    d_date = event_list[k]['admdate'].split('-')
    d_date = dt.date(int(d_date[0]), int(d_date[1]), int(d_date[2]))
    if pan_can_code in event_list[k]['codes'] and not cancer_p:
        cancer_p = True
        patient_rec[patient_rec.columns[0]].date_of_diagnosis = d_date
        b_date = j_records[int(patient_rec.columns[0])].birthdate.split('-')
        b_date = dt.date(int(b_date[0]), int(b_date[1]), int(b_date[2]))
        d_age = [d_date - b_date]
    # assuming date_of_diagnosis correct:
    pre_pc_t = patient_rec[patient_rec.columns[0]].date_of_diagnosis- d_date
    if pre_pc_t > \
            thresholds[0]:
        timewise_n_dis[0,0] += 1
    if pre_pc_t < thresholds[0] and pre_pc_t > thresholds[1]:
        timewise_n_dis[1, 0] += 1
    if pre_pc_t < thresholds[1] and pre_pc_t > thresholds[2]:
        timewise_n_dis[2, 0] += 1
    if pre_pc_t < thresholds[2]:
        timewise_n_dis[3, 0] += 1
# all_disease[0] = n_diseases == len(event_list)
# diseases[0] = n_diseases
for j in range(1, len(patient_rec.columns)):
    # if key_list[j] in j_records:

        # per patient
        j_records[int(patient_rec.columns[j])].events.sort(key=event_sort)
        event_list = j_records[int(patient_rec.columns[j])].events
        n_diseases = 0
        last_event = event_list[-1]['admdate'].split('-')
        last_event = dt.date(int(last_event[0]), int(last_event[1]), int(last_event[2]))
        start = event_list[0]['admdate'].split('-')
        start = dt.date(int(start[0]), int(start[1]), int(start[2]))
        traj = np.append(traj, [np.abs(last_event - start)]) # not really
    # sure why I get negatives -- it's because events aren't sorted
        cancer_p = False
        for k in range(len(event_list)):
            if event_list[k]['codes'] in disease_codes.icd_code.values:
                n_diseases += 1
            d_date = event_list[k]['admdate'].split('-')
            d_date = dt.date(int(d_date[0]), int(d_date[1]), int(d_date[2]))
            if pan_can_code in event_list[k]['codes'] and not cancer_p:
                cancer_p = True
                patient_rec[patient_rec.columns[j]].date_of_diagnosis = d_date
                b_date = j_records[j_records.columns[j]].birthdate.split('-')
                b_date = dt.date(int(b_date[0]), int(b_date[1]), int(b_date[2]))
                d_age = np.append(d_age, d_date - b_date)
                # c_pat_rec = pd.concat([c_pat_rec, j_records[j_records.columns[j]]],
                #                       axis=1)
        # patient_rec = pd.concat([patient_rec, j_records[j_records.columns[j]]],
        #                                 axis=1)
        # diseases[j] = n_diseases
            pre_pc_t = patient_rec[patient_rec.columns[0]].date_of_diagnosis \
                       - d_date
            if pre_pc_t > \
                    thresholds[0]:
                timewise_n_dis[0, j] += 1
            if pre_pc_t < thresholds[0] and pre_pc_t > thresholds[1]:
                timewise_n_dis[1, j] += 1
            if pre_pc_t < thresholds[1] and pre_pc_t > thresholds[2]:
                timewise_n_dis[2, j] += 1
            if pre_pc_t < thresholds[2]:
                timewise_n_dis[3, j] += 1
        # all_disease[j] = n_diseases == len(event_list) # is in fact all
    # diseases
# d_median = np.median(diseases) # 18 is the general dataset, 41 is for cancer
# patients
median_Traj = np.median(traj)
patient_rec.to_csv('cancer patient dates.csv')
# age_of_diag = np.median(d_age) # I would guess it comes down to the shape
# of the data being extracted
total_n_dis_PC = np.sum(timewise_n_dis,axis=1)
print('debug draft') # throwaway line of code for putting a breakpoint in my IDE