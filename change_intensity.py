import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import datetime
import some_function as sf

choice = ['noise_opt2', 'stokes_opt2', 'gain_opt3', 'noise_opt4', 'stokes_opt3', 'iteration', 'gauss', 'square', 'shotnoise']
i = 8
filename = 'E:/python/code/data/candidate/' + choice[i] + '.csv'
wave = pd.read_csv(filename, header=None).values.flatten()
if i == 8:
    wave = wave
else:
    wave = wave/np.mean(wave)/3
var = []
var_sq = []
var_asq = []
intensity = []
initial_sq = []
initial_asq = []

today = str(datetime.date.today())
path_today = 'E:/python/code/intensity/' + today
start_datetime = str(datetime.datetime.now())[11:19]
start_time = sf.change_colon_into_dash(start_datetime)
path_today += '/' + choice[i]
path_today_ini = path_today + '/ini_data'
path_today_data = path_today + '/processed_data'
if not os.path.exists(path_today):
    os.makedirs(path_today)
    os.makedirs(path_today_ini)
    os.makedirs(path_today_data)
path_today = path_today + '/'
path_today_data += '/'
path_today_ini += '/'
'''
sf.array_to_csv(array=wave/2, filename='E:/python/documents/wave_opt/gausswave.csv')
gain = sf.convey_params_get_cost([1])

a = np.linspace(0.05, 0.13, 8)
'''
for k in range(8):
    wave_int = wave

    filename = 'E:/python/documents/wave_opt/gausswave.csv'
    sf.array_to_csv(filename=filename, array=wave_int)

    gate = sf.convey_params_get_cost([-1])

    filename_seedlight = 'D:/shot noise/' + sf.today_str() + '/try2/seed.csv'
    filename_SQ = 'D:/shot noise/' + sf.today_str() + '/try/SQ.csv'
    filename_ASQ = 'D:/shot noise/' + sf.today_str() + '/try/ASQ.csv'

    df_seed = pd.read_csv(filename_seedlight, header=None).values
    df_SQ = pd.read_csv(filename_SQ, header=None).values
    df_ASQ = pd.read_csv(filename_ASQ, header=None).values

    initial_sq.append(df_SQ.tolist())
    initial_asq.append(df_ASQ.tolist())

    res = df_SQ - df_ASQ

    df_SQ = df_SQ[np.where(np.abs(res - np.mean(res)) < 1 * gate)]
    res = res[np.where(np.abs(res - np.mean(res)) < 1 * gate)]

    initial_sq.append(df_SQ.tolist())
    initial_asq.append(df_ASQ.tolist())

    intensity_sq = np.mean(df_SQ)
    varian = np.var(res)
    print('var and intensity are:', varian, intensity_sq)
    print('cost is:', np.log10(varian))
    var.append(varian)
    var_sq.append((np.var(df_SQ)))
    var_asq.append(np.var(df_ASQ))
    intensity.append(intensity_sq)


filepath = path_today_data + str(start_time) + '.csv'
array = np.array([intensity, var, var_sq, var_asq])
sf.array_to_csv(filepath, array=array)

file_saveSQ = path_today_ini + str(start_time) + '_SQ' + '.csv'
file_saveASQ = path_today_ini + str(start_time) + '_ASQ' + '.csv'
sf.array_to_csv(file_saveASQ, initial_asq)
sf.array_to_csv(file_saveSQ, initial_sq)

intensity, var = array[0], array[1]
'''
slope, intercept = sf.linear_regre_1d(intensity, var)
print('slope and intercept are:', slope, intercept)
x=np.linspace(0,np.max(intensity),100)
y=slope*x+intercept
'''

plt.figure(figsize=(16, 9))
plt.title('Variance vs Intensity', fontsize=24)
plt.xlabel('Intensity', fontsize=20)
plt.ylabel('Variance', fontsize=20)
plt.scatter(intensity, var, c='navy')
#plt.scatter(intensity, var_sq, c='darkviolet')
'''
plt.plot(x, y, c='red')
'''
plt.show()