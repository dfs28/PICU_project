### Correlation plots
#Setup

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load data
data = np.load('/store/DAMTP/dfs28/PICU_data/np_arrays.npz')
array3d = data['d3']
array2d = data['d2']
outcomes = data['outcomes']
characteristics = data['chars']
splines = data['splines']
point_cols = data['point_cols']
series_cols = data['series_cols']

#Combine the series cols objects and name them for labelling of axes
series_cols_mean = ['Mean ' + i for i in series_cols]
series_cols_std = ['STD ' + i for i in series_cols]
for i in series_cols_std:
    series_cols_mean.append(i)
outcome_labels = ['Death <2d', 'Death this admission >2d', 'Survives', 'Discharge <48h', 'Discharge 48h-7d', 'Discharge >7d', 'Deterioration >6h', 'Deterioration 6-24h', 'No deterioration <24h']
point_cols = [i for i in point_cols]

#Plot correlations with point variables and themselves
fig, ax1 = plt.subplots(1, 1, figsize = (20, 15))
array2d_cors = np.corrcoef(array2d, rowvar=False)
array2d_cors[np.isnan(array2d_cors)] = 0
for i in range(array2d_cors.shape[0]):
    array2d_cors[i, i] = np.nan
ax1 = sns.heatmap(array2d_cors, xticklabels = data['point_cols'], yticklabels = data['point_cols'])
fig.savefig("/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/arr2d_heatmap.png")

#Plot correlations of point vars and outcomes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (40, 20), gridspec_kw={'width_ratios': [1, 2.5]})
array2d_cors_outcomes = np.corrcoef(array2d, outcomes[:, 2:11], rowvar=False)[53:62, :53]
array2d_cors_outcomes[np.isnan(array2d_cors_outcomes)] = 0
sns.heatmap(array2d_cors_outcomes, yticklabels = outcome_labels, xticklabels = data['point_cols'], cbar = False, ax = ax1)

#Plot correlations of series vars and outcomes
chars_cors_outcomes = np.corrcoef(characteristics, outcomes[:, 2:11], rowvar=False)[120:, :120]
chars_cors_outcomes[np.isnan(chars_cors_outcomes)] = 0
sns.heatmap(chars_cors_outcomes, xticklabels = series_cols_mean, ax = ax2)
fig.subplots_adjust(wspace=0.01)
ax2.yaxis.tick_right()
plt.savefig("/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/all_outcomes_heatmap.png")

#Plot correlations of series cols and themselves
fig, ax1 = plt.subplots(1, 1, figsize = (25, 20))
chars_cors = np.corrcoef(characteristics, rowvar=False)
chars_cors[np.isnan(chars_cors)] = 0
for i in range(chars_cors.shape[0]):
    chars_cors[i, i] = np.nan
sns_plot = sns.heatmap(chars_cors, xticklabels = series_cols_mean, yticklabels = series_cols_mean)
fig.savefig("/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/chars_heatmap.png")

#Plot correlations with point variables and themselves
fig, ax1 = plt.subplots(1, 1, figsize = (50, 40))
all_cors = np.corrcoef(np.concatenate((array2d, characteristics), axis=1), rowvar=False)
all_cors[np.isnan(all_cors)] = 0
for i in range(all_cors.shape[0]):
    all_cors[i, i] = np.nan
ax1 = sns.heatmap(all_cors, xticklabels = point_cols + series_cols_mean, yticklabels = point_cols + series_cols_mean)
fig.savefig("/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/all_cors_heatmap.png")

#Plot some patient data
#In reality import the big flowsheet to use thing
flowsheet = pd.read_csv('/store/DAMTP/dfs28/PICU_data/flowsheet_sample_zscores.csv', parse_dates = ['taken_datetime'])
print('Flowsheet loaded: ', datetime.now().strftime("%H:%M:%S"))

#Fix issue with ethnicity, sex and died
died = {'N': 0, 'Y': 1}
flowsheet['died'] = flowsheet['died'].replace(died)
sex = {'F': 0, 'M': 1, 'I': 2}
flowsheet['sex'] = flowsheet['sex'].replace(sex)
ethnicity = {j:i[0] for i, j in np.ndenumerate(flowsheet['ethnicity'].unique())}
flowsheet['ethnicity'] = flowsheet['ethnicity'].replace(ethnicity)

point_cols = ['ALT', 'Albumin', 'AlkPhos','AST', 'Aspartate', 'Amylase', 'APTT', 'Anion_gap', 'Base_excess', 'Basophils', 'Bicarb',
            'pH', 'Blood_culture', 'CRP', 'Ca2.', 'Cl', 'Eosinophils', 'FHHb', 'FMetHb', 'FO2Hb', 'Glucose', 'HCT.', 'HCT', 'INR',
            'Lactate', 'Lymphs', 'Mg', 'Monocytes', 'Neuts', 'P50', 'PaCO2', 'PcCO2', 'PmCO2', 'PaO2', 'PcO2', 'PmO2', 'PO2', 
            'PvCO2', 'PcO2.1', 'Phos', 'Plts', 'K.', 'PT', 'Retics', 'Na.', 'TT', 'Bili', 'WCC', 'Strong_ion_gap', 'Age_yrs', 'sex', 
            'ethnicity', 'Weight_z_scores']


series_cols =  ['Ventilation', 'HFO', 'IPAP', 'EPAP', 'Tracheostomy', 'ETCO2', 'FiO2', 'O2Flow.kg', 'Ventilation_L_min', 'Ventilation.ml.',
                'MeanAirwayPressure', 'Ventilation_missing', 'O2Flow.kg_missing', 'IPAP_missing', 'EPAP_missing', 'FiO2_missing', 'HFO_missing',
                'Tracheostomy_missing', 'Ventilation.ml._missing', 'MeanAirwayPressure_missing', 'ETCO2_missing', 'SBP_zscore', 'DBP_zscore', 
                'MAP_zscore', 'SysBP_missing', 'DiaBP_missing', 'MAP_missing', 'HR_zscore', 'HR_missing', 'Comfort.Alertness', 'Comfort.BP', 'Comfort.Calmness',
                'Comfort', 'Comfort.HR', 'Comfort.Resp', 'AVPU', 'GCS_V', 'GCS_E', 'GCS_M', 'GCS', 'GCS_missing', 'AVPU_missing', 'CRT', 'CRT_missing',
                'SpO2', 'SpO2_missing', 'interpolated_ht_m', 'ECMO', 'ECMO_missing', 'Inotropes_kg', 'Inotropes_missing', 'RR_zscore', 'RR_missing',
                'dialysis', 'dialysis_missing', 'Temp', 'Temp_missing', 'Urine_output_kg', 'Urine_output_missing_kg', 'PEWS']

all_used_cols = point_cols + series_cols

array_use = np.array(flowsheet.loc[:, all_used_cols])

length = 180
i = 0
start_position = i*length
end_position = (i + 1)*length
temp_array = array_use[start_position:end_position, :]

my_cmap = sns.color_palette("vlag", as_cmap=True)
fig, ax1 = plt.subplots(1, 1, figsize = (25, 20))
pt_data = array3d[33000, :, :]
sns_plot = sns.heatmap(temp_array,  cmap = my_cmap)
fig.savefig("/mhome/damtp/q/dfs28/Project/PICU_project/figs/PICU/Sample_data.png")