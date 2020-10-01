import numpy as np
import pandas as pd

import datetime
import some_function as sf

def generate_wave(day=None, start_time = '20-50-26',choice='e'):
    if not day:
        day = str(datetime.date.today())
    if day != '2020-07-03':
        path_today = 'E:/python/code/data/' + day +'/' + start_time
    else:
        path_today = 'E:/python/code/data/' + day

    filename_s = path_today + '/seedlight_' + start_time + '.csv'
    filename_e = path_today + '/excited_light_' + start_time + '.csv'
    filename_p = path_today + '/pre_best_paras_' + start_time + '.csv'
    filename_b = path_today + '/best_paras_' + start_time + '.csv'
    dic = {
        'e': filename_e,
        's': filename_s,
        'p': filename_p,
        'b': filename_b,
    }

    filename = dic[choice]
    df = pd.read_csv(filename, header=None)
    y = np.array(df)
    y = sf.wave_normalized(y)
    return y

def main():
    sep = ['p','s','e','b']
    day = '2020-07-08'
    start_time = '15-51-13'
    for x in sep:
        wave = generate_wave(day=day, start_time=start_time, choice=x)

        filename='E:/python/documents/wave_opt/gausswave.csv'
        sf.array_to_csv(wave, filename=filename)

        cost = sf.convey_params_get_cost([1])

if __name__ == '__main__':
    main()