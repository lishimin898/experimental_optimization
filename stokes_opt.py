# import from m-loop
import mloop.interfaces as mli
import mloop.controllers as mlc
import mloop.visualizations as mlv
# import for data analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import some useful functions from a py file in the same folder
import some_function as sf
# import for gaussian process regression
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as skk
# other import
import os
import datetime

class CustomInterface_2(mli.Interface):  # the second interface to optimize the stokes noise

    def __init__(self, nums_params):
        # You must include the super command to call the parent class, Interface, constructor
        super(CustomInterface_2, self).__init__()

        self.nums_params = nums_params

        self.paras_scale = 0.042

    def get_next_cost_dict(self, params_dict):
        # Get parameters from the provided dictionary
        params = params_dict['params']
        #wave = self.gaussian_interpolation(params=x)
        wave = sf.wave_normalization(params)
        filename = 'E:/python/documents/wave_opt/gausswave.csv'
        sf.array_to_csv(filename=filename, array=wave)

        siganl = [1]
        gain = sf.convey_params_get_cost(siganl)

        if gain < 0.007 or np.max(wave) >= 2:
            bad = True
            cost = 1
            uncer = 0
        else:
            wave = wave/abs(gain)*self.paras_scale
            sf.array_to_csv(filename=filename,array=wave)

            signal = [-1]
            gate = sf.convey_params_get_cost(signal)

            filename_SQ = 'D:/shot noise/' + sf.today_str() + '/try/SQ.csv'
            filename_ASQ = 'D:/shot noise/' + sf.today_str() + '/try/ASQ.csv'
            df_SQ = pd.read_csv(filename_SQ, header=None).values
            df_ASQ = pd.read_csv(filename_ASQ, header=None).values

            res = df_SQ
            res = res[np.where(np.abs(res-np.mean(res))<(3*gate))]

            varian = np.var(res)/np.mean(res)
            print('var is:', varian)

            cost = 10*np.log10(varian)
            bad = False
            uncer = sf.estim_noise_uncer(res)*10/np.var(res)/np.log(10)

        cost_dict = {'cost': cost, 'uncer': uncer, 'bad': bad}
        return cost_dict

    def gaussian_interpolation(self, params, m=200):
        a = np.linspace(0, 100, self.nums_params)
        a = np.concatenate([np.array([0]),a,np.array([0])])
        x_train = np.atleast_2d(a).T
        y_train = params
        x_test = np.atleast_2d(np.linspace(0, 100, m+1)).T

        kernel = skk.RBF()
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, alpha=1e-4, normalize_y=True)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        return y_pred

def main():
    # create a file folder to save today's data
    today = str(datetime.date.today())
    path_today = 'E:/python/code/data/' + today
    if not os.path.exists(path_today):
        os.makedirs(path_today)


    n = 7
    interface_2 = CustomInterface_2(n)
    controller = mlc.create_controller(interface_2,
                                       #training_type='differential_evolution',
                                       controller_type='neural_net',
                                       max_num_runs=100,
                                       keep_prob=0.8,
                                       num_params=n,
                                       min_boundary=[0.2]*n,
                                       max_boundary=[1]*n)

    # start the second optimization
    controller.optimize()

    # create the file folder of this optimization
    start_datetime = str(controller.start_datetime)[11:19]
    start_time = sf.change_colon_into_dash(start_datetime)
    path_today += '/' + str(start_time)
    os.makedirs(path_today)

    # save the best wave
    res = sf.add_zero(controller.best_params)
    filename = path_today + '/best_paras_2_' + str(start_time) + '.csv'
    sf.array_to_csv(filename=filename, array=res)

    # save the predicted wave
    res = sf.add_zero(controller.predicted_best_parameters)

    filename = path_today + '/pre_best_paras_2_' + str(start_time) + '.csv'
    sf.array_to_csv(filename=filename, array=res)

    # save the seed light of the predicted wave
    res = sf.wave_normalization(res)

    filename = 'E:/python/documents/wave_opt/gausswave.csv'
    sf.array_to_csv(filename=filename, array=res)
    siganl = [1]
    gain = sf.convey_params_get_cost(siganl)

    res=res/gain*interface_2.paras_scale
    filename = 'E:/python/documents/wave_opt/gausswave.csv'
    sf.array_to_csv(filename=filename, array=res)

    signal = [-1]
    gate = sf.convey_params_get_cost(signal)

    filename_SQ = 'D:/shot noise/' + sf.today_str() + '/try/SQ.csv'
    filename_ASQ = 'D:/shot noise/' + sf.today_str() + '/try/ASQ.csv'
    df_SQ = pd.read_csv(filename_SQ, header=None).values
    df_ASQ = pd.read_csv(filename_ASQ, header=None).values
    res = df_SQ - df_ASQ
    df_SQ = df_SQ[np.where(np.abs(res - np.mean(res)) < 3 * gate)]
    res = res[np.where(np.abs(res - np.mean(res)) < 3 * gate)]
    cost = np.var(res)/np.mean(df_SQ)
    print('predicted best cost_2 is:', controller.predicted_best_cost)
    print('actual cost_2 is:', 10*np.log10(cost))

    filename_seedlight = 'D:/shot noise/' + sf.today_str() + '/try2/seed.csv'
    filename = path_today + '/seedlight_2_' + str(start_time) + '.csv'
    df = pd.read_csv(filename_seedlight, header=None)
    df.to_csv(filename, header=0, index=0)

    # save the excited stokes light
    filename_excited = 'D:/shot noise/' + sf.today_str() + '/try/excited.csv'
    filename = path_today + '/excited_light_2_' + str(start_time) + '.csv'
    df = pd.read_csv(filename_excited, header=None)
    df.to_csv(filename, header=0, index=0)

    filename_SQ = 'D:/shot noise/' + sf.today_str() + '/try/SQ.csv'
    filename = path_today + '/SQ_2_' + str(start_time) + '.csv'
    df = pd.read_csv(filename_SQ, header=None)
    df.to_csv(filename, header=0, index=0)

    # save the anti-stokes light
    filename_excited = 'D:/shot noise/' + sf.today_str() + '/try/antistokes.csv'
    filename = path_today + '/antistokes_2_' + str(start_time) + '.csv'
    df = pd.read_csv(filename_excited, header=None)
    df.to_csv(filename, header=0, index=0)

    filename_ASQ = 'D:/shot noise/' + sf.today_str() + '/try/ASQ.csv'
    filename = path_today + '/ASQ_2_' + str(start_time) + '.csv'
    df = pd.read_csv(filename_ASQ, header=None)
    df.to_csv(filename, header=0, index=0)

    # show visualizations
    mlv.show_all_default_visualizations(controller)


# Ensures main is run when this code is run as a script
if __name__ == '__main__':
    main()