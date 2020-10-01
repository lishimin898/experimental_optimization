import numpy as np
import matplotlib.pyplot as plt
import time as t
import some_function as sf
import pandas as pd
import os

def generate_wave():
    nums = 200
    x = np.linspace(0,200,nums)
    u, sig = 100, 30
    #y = 1*np.exp(-(x-u)**2/(2*sig**2))
    choice = ['noise_opt2', 'stokes_opt2', 'gain_opt3', 'noise_opt4', 'stokes_opt3', 'iteration', 'gauss', 'square', 'shotnoise']
    i = 8
    filename = 'E:/python/code/data/candidate/' + choice[i] + '.csv'
    y = pd.read_csv(filename, header=None).values.flatten()

    #print(y)
    #y = np.concatenate([y,y])
    #y = 0.5*np.array([1]+[1]*(nums-2)+[1])
    '''
    today = str(datetime.date.today())
    path_today = 'E:/python/code/data/' + today
    start_time = '20-50-26'
    filename_s = path_today + '/seedlight_' + start_time + '.csv'
    filename_e = path_today + '/excited_light_' + start_time + '.csv'
    filename_p = path_today + '/pre_best_paras_' + start_time + '.csv'
    filename = filename_s
    df = pd.read_csv(filename, header=None)
    y = np.array(df)
    print(y)
    y = wave_normalized(y)
    '''
    '''
    y = np.random.uniform(0.25,0.75,size=x.shape)
    y = np.concatenate([np.array([0]),y,np.array([0])])
    y = s.gaussian_interpolation(params=y, n=nums)[0]
    '''
    #y = np.array([0.5])
    #y = 1-np.abs(0.01*(x-100))
    #y = np.sin(0.1*x)
    #y = y/np.mean(y)/3
    
    '''
    plt.plot(y)
    plt.show()
    '''
    return y

def generateDC(u):
    x = np.linspace(0, 200, 200)
    y = np.array([u]*200)
    return y

def main_test():
    wave = generate_wave()
    #wave = sf.wave_normalization(wave)

    filename='E:/python/documents/wave_opt/gausswave.csv'
    sf.array_to_csv(array=wave, filename=filename)
    signal=[-1]
    gate = sf.convey_params_get_cost(signal)
    if signal[0]<0:
        filename_SQ = 'D:/shot noise/' + sf.today_str() + '/try/SQ.csv'
        filename_ASQ = 'D:/shot noise/' + sf.today_str() + '/try/ASQ.csv'
        filename_excited = 'D:/shot noise/' + sf.today_str() + '/try/excited.csv'
        df_SQ = pd.read_csv(filename_SQ, header=None).values
        df_ASQ = pd.read_csv(filename_ASQ, header=None).values
        '''
        df_excited = pd.read_csv(filename_excited, header=None).values
        df_SQ = df_SQ - np.min(df_excited)*len(df_excited)
        '''
        inten_SQ = np.mean(df_SQ)
        std_SQ = np.var(df_SQ)
        print('intensity of SQ and var of SQ are:', inten_SQ, std_SQ)
        res = df_SQ - df_ASQ

        df_SQ = df_SQ[np.where(np.abs(res - np.mean(res)) < 3 * gate)]
        res = res[np.where(np.abs(res - np.mean(res)) < 3 * gate)]

        cost = np.var(res)/np.mean(df_SQ)
        print('var is:', 10*np.log10(cost))

        filename_seed = 'D:/shot noise/' + sf.today_str() + '/try2/seed.csv'
        filename_everyseed = 'D:/shot noise/' + sf.today_str() + '/try2/everyseed.csv'
        df_seed = pd.read_csv(filename_seed, header=None).values
        df_everyseed = pd.read_csv(filename_everyseed, header=None).values
        inten_seed = np.mean(df_everyseed-np.min(df_seed)*len(df_seed))
        var_seed = np.var(df_everyseed-np.min(df_seed)*len(df_seed))
        print('intensity of seedlight and var of seedlight are:', inten_seed, var_seed)

def testAOM():
    res = []
    for i in range(1001):
        wave = generateDC(0.001*i)

        filename = 'E:/python/documents/wave_opt/gausswave.csv'
        sf.array_to_csv(wave, filename=filename)

        cost = sf.convey_params_get_cost([1])
        res.append(cost)
        t.sleep(0.1)

    res = np.array(res)
    filename = 'E:/python/data/20200622/aom.csv'
    sf.array_to_csv(res, filename=filename)

def seedlight_test():
    wave = generate_wave()
    wave = sf.wave_normalization(wave)

    filename = 'E:/python/documents/wave_opt/gausswave.csv'
    sf.array_to_csv(array=wave, filename=filename)
    signal = [-1]
    gate = sf.convey_params_get_cost(signal)
    if signal[0] < 0:
        filename_seed = 'D:/shot noise/' + sf.today_str() + '/try2/seed.csv'
        filename_everyseed = 'D:/shot noise/' + sf.today_str() + '/try2/everyseed.csv'
        df_seed = pd.read_csv(filename_seed, header=None).values
        df_everyseed = pd.read_csv(filename_everyseed, header=None).values
        df_everyseed = df_everyseed - np.min(df_seed)*len(df_seed)
        inten_seed = np.mean(df_everyseed)
        var_seed = np.std(df_everyseed)
        print('intensity of seedlight and std of seedlight are:', inten_seed, var_seed)

    return inten_seed, var_seed

def stokes_test():
    wave = generate_wave()
    wave = sf.wave_normalization(wave)

    filename = 'E:/python/documents/wave_opt/gausswave.csv'
    sf.array_to_csv(array=wave, filename=filename)
    signal = [-1]
    gate = sf.convey_params_get_cost(signal)
    if signal[0] < 0:
        filename_SQ = 'D:/shot noise/' + sf.today_str() + '/try/SQ.csv'
        filename_ASQ = 'D:/shot noise/' + sf.today_str() + '/try/ASQ.csv'
        filename_excited = 'D:/shot noise/' + sf.today_str() + '/try/excited.csv'
        df_SQ = pd.read_csv(filename_SQ, header=None).values
        df_ASQ = pd.read_csv(filename_ASQ, header=None).values
        df_excited = pd.read_csv(filename_excited, header=None).values
        df_SQ = df_SQ - np.min(df_excited) * len(df_excited)
        inten_SQ = np.mean(df_SQ)
        std_SQ = np.var(df_SQ)
        print('intensity of SQ and var of SQ are:', inten_SQ, std_SQ)
    return inten_SQ,std_SQ


if __name__ == '__main__':
    '''
    x, y = [], []
    for i in range(5):
        I, v = stokes_test()
        x.append(I)
        y.append(v)
    x = np.array(x)
    y = np.array(y)
    print('average intensity of stokes light is:', np.mean(x))
    print('average std is:', np.mean(y))
    print('relative error is:', np.mean(y/x))
    path = './data/stokes_test'
    if not os.path.exists(path):
        os.makedirs(path)
    sf.array_to_csv(filename='./data/stokes_test/lock.csv', array=[x, y])
    '''
    main_test()
