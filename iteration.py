import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datetime
import os
import some_function as sf

class wave_iter():
    def __init__(self, nums_iter, init_wave=None, filename=None, save_stokes=True, show_plot=True):
        self.nums_iter = nums_iter
        if not init_wave:
            init_wave = 'g'
        self.init_wave = init_wave
        if not filename:
            filename = 'E:/python/documents/wave_opt/gausswave.csv'
        self.filename = filename

        self.cost_list = []
        self.seed_intensity = []

        if save_stokes:
            today = str(datetime.date.today())
            path_today = 'E:/python/code/iteration/' + today
            start_datetime = str(datetime.datetime.now())[11:19]
            start_time = sf.change_colon_into_dash(start_datetime)
            path_today += '/' + str(start_time)
            if not os.path.exists(path_today):
                os.makedirs(path_today)
            self.path_today = path_today+'/'
        else:
            self.path_today = ''

        self.show_plot = show_plot

    def gauss_wave(self):
        nums = 135
        x = np.linspace(0, 200, nums)
        u, s = 100, 30
        y = np.exp(-(x-u)**2/(2*s**2))
        y = np.concatenate([y,y])
        return y

    def square_wave(self):
        nums = 135
        x = np.linspace(0, 200, nums)
        y = np.array([0] + [1] * (nums - 2) + [0])
        return y

    def triangle_wave(self):
        nums = 200
        x = np.linspace(0, 200, nums)
        y = 1 - np.abs(0.01 * (x - 100))
        return y

    def AWG(self, csv_file):
        df = pd.read_csv(csv_file, header=None)
        return np.array(df).flatten()

    def generate_wave(self, count):
        filename_excited = 'D:/shot noise/' + sf.today_str() + '/try/excited.csv'
        filename_seed = 'D:/shot noise/' + sf.today_str() + '/try2/seed.csv'

        df = pd.read_csv(filename_excited, header=None)
        df_seed = pd.read_csv(filename_seed, header=None)

        wave_seed = df_seed.values.flatten()
        self.seed_intensity.append(np.sum(wave_seed))
        print('seedlight:', np.sum(wave_seed))

        wave = df.values.flatten()
        wave = sf.wave_normalization(wave)
        wave=np.maximum(wave,0)
        sf.array_to_csv(filename=self.filename, array=wave)

        if self.path_today:
            savefile_name = self.path_today+str(count)+'.csv'
            seedfile = self.path_today+'seedlight'+str(count)+'.csv'
            sf.array_to_csv(filename=savefile_name,array=wave)
            sf.array_to_csv(filename=seedfile, array=wave_seed)

    def run(self, csv_file=None,signal=[1]):
        if not csv_file:
            if self.init_wave == 'g':
                wave = self.gauss_wave()
            elif self.init_wave == 't':
                wave = self.triangle_wave()
            elif self.init_wave == 's':
                wave = self.square_wave()
        else:
            while not os.path.isfile(csv_file):
                csv_file = input('Please input the path of Arbitary Wave:')
            wave = self.AWG(csv_file)
        wave = sf.wave_normalization(wave)
        sf.array_to_csv(array=wave,filename=self.filename)
        if self.path_today:
            sf.array_to_csv(array=wave,filename=self.path_today+'initwave'+'.csv')
        
        for i in range(self.nums_iter):
            print('Run:', i+1)
            self.signal = signal[0]
            cost = sf.convey_params_get_cost(params=signal)
            if self.signal<0:
                filename_SQ = 'D:/shot noise/' + sf.today_str() + '/try/SQ.csv'
                filename_ASQ = 'D:/shot noise/' + sf.today_str() + '/try/ASQ.csv'
                df_SQ = pd.read_csv(filename_SQ, header=None).values
                df_ASQ = pd.read_csv(filename_ASQ, header=None).values
                res = df_SQ - df_ASQ
                gate = cost
                res = res[np.where(np.abs(res - np.mean(res)) < 3 * gate)]
                var_diff = 10*np.log10(np.var(res)/np.mean(df_SQ))
                print('var_diff is:',var_diff)
            self.cost_list.append(var_diff)
            self.generate_wave(count=i+1)

        if self.path_today:
            filename = self.path_today+'Gain_vs_run'+'.csv'
            sf.array_to_csv(array=self.cost_list, filename=filename)
            filename = self.path_today + 'Seed intensity_vs_run' + '.csv'
            sf.array_to_csv(array=self.seed_intensity,filename=filename)

        self.plot_costs_vs_run()

    def plot_costs_vs_run(self):
        plt.figure(figsize=(16,9))
        plt.scatter(range(len(self.cost_list)), self.cost_list, c='purple')

        plt.xlabel('Run numbers', fontsize = 20)
        if self.signal>0:
            plt.ylabel('Gain', fontsize = 20)
            plt.title('Gain vs Run numbers', fontsize=24)
        else:
            plt.ylabel('Variance', fontsize=20)
            plt.title('Variance vs Run numbers', fontsize=24)

        if self.path_today:
            plt_path = self.path_today+'figure.png'
            plt.savefig(plt_path, dpi=300)
        if self.show_plot:
            plt.show()

if __name__== '__main__':
    nums_iter = 15
    w = wave_iter(nums_iter=nums_iter, init_wave='g', save_stokes=True)
    csv_file = 'E:/python/code/data/candidate/noise_opt.csv'
    signal = [-1]
    w.run(signal=signal,
          #csv_file=csv_file
          )