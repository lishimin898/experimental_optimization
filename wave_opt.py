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


class CustomInterface_1(mli.Interface):

    def __init__(self, nums_params=9):
        # You must include the super command to call the parent class, Interface, constructor
        super(CustomInterface_1, self).__init__()
        # use neuralnet to control the shape of the wave,the default number of points(parameters) is 9
        self.nums_params = nums_params

    def get_next_cost_dict(self, params_dict):
        # Get parameters from the provided dictionary
        params = params_dict['params']

        # input_inter = s.gaussian_interpolation(x)  # GPR interpolation
        input_inter = sf.wave_normalization(params)
        filename = 'E:/python/documents/wave_opt/gausswave.csv'
        sf.array_to_csv(filename=filename, array=input_inter)  # save wave into a csv file

        # To optimize the gain, signal=[1] is to tell labview the optimizing target is gain
        signal = [1]
        gain = sf.convey_params_get_cost(signal)
        '''
        filename_seedlight = 'D:/shot noise/' + today_str() + '/try2/SQ.csv'
        df = pd.read_csv(filename_seedlight, header=None)
        wave = np.array(df)
        array_to_csv(filename=filename,array=wave)

        signal = [1]
        cost = sf.convey_params_get_cost(signal)  
        '''
        if gain > 0:
            cost = 10*np.log10(1/gain)  # the target is the minimum value of the cost, also is the maximum value of the gain
            uncer = 0
            bad = False
        else:
            cost = 10*np.log10(-1/gain)
            uncer = 0
            bad = True

        cost_dict = {'cost': cost, 'uncer': uncer, 'bad': bad}

        return cost_dict



class CustomInterface_2(mli.Interface):  # the second interface to optimize the noise

    def __init__(self, nums_params):
        # You must include the super command to call the parent class, Interface, constructor
        super(CustomInterface_2, self).__init__()

        self.nums_params = nums_params                  

    def get_next_cost_dict(self, params_dict):
        # Get parameters from the provided dictionary
        params = params_dict['params']

        #wave = self.gaussian_interpolation(params=x)
        wave = sf.wave_normalization(params)

        filename = 'E:/python/documents/wave_opt/gausswave.csv'
        sf.array_to_csv(filename=filename, array=wave)  

        signal = [-1]
        gate = sf.convey_params_get_cost(signal)  

        filename_SQ = 'D:/shot noise/' + sf.today_str() + '/try/SQ.csv'
        filename_ASQ = 'D:/shot noise/' + sf.today_str() + '/try/ASQ.csv'
        df_SQ = pd.read_csv(filename_SQ, header=None).values
        df_ASQ = pd.read_csv(filename_ASQ, header=None).values
        res = df_SQ-df_ASQ
        res = res[np.where(np.abs(res-np.mean(res))<3*gate)]
        varian = np.var(res)/np.mean(df_SQ)
        print('var is:', varian)
        cost = 100*np.log10(varian)

        uncer = 0
        bad = False
        cost_dict = {'cost': cost, 'uncer': uncer, 'bad': bad}
        return cost_dict

    def gaussian_interpolation(self, params, m=200):  
        a = np.linspace(0, 100, self.nums_params)
        a = np.concatenate([np.array([0]),a,np.array([0])])
        x_train = np.atleast_2d(a).T
        y_train = params
        x_test = np.atleast_2d(np.linspace(0, 100, m+1)).T

        kernel = skk.RBF()
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, alpha=1e-4,normalize_y=True)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        return y_pred



def relu(x):  
    return np.maximum(0, x)


def main():
    # create a file folder to save today's data
    today = str(datetime.date.today())
    path_today = 'E:/python/code/data/' + today
    if not os.path.exists(path_today):
        os.makedirs(path_today)

    # interface to optimize gain
    interface_1 = CustomInterface_1(nums_params=7)
    controller = mlc.create_controller(interface_1,
                                       #training_type='random',
                                       controller_type='neural_net',
                                       max_num_runs=100,
                                       # target_cost = -2.99,
                                       keep_prob=0.8,
                                       num_params=interface_1.nums_params,
                                       min_boundary=[0.2] * interface_1.nums_params,
                                       max_boundary=[1] * interface_1.nums_params)

    # To run M-LOOP and find the optimal parameters just use the controller method optimize
    controller.optimize()

    # create the file folder of this optimization
    start_datetime = str(controller.start_datetime)[11:19]
    start_time = sf.change_colon_into_dash(start_datetime)
    path_today += '/'+str(start_time)
    os.makedirs(path_today)

    # save the best wave
    res = sf.add_zero(controller.best_params)
    filename = path_today + '/best_paras_' + str(start_time) + '.csv'
    sf.array_to_csv(filename=filename, array=res)

    # save the predicted wave
    res = sf.add_zero(controller.predicted_best_parameters)
    filename = path_today+'/pre_best_paras_'+str(start_time)+'.csv'
    sf.array_to_csv(filename=filename, array=res)

    # save the seed light of the predicted wave
    filename = 'E:/python/documents/wave_opt/gausswave.csv'
    sf.array_to_csv(filename=filename, array=res)

    signal = [1]
    gain = sf.convey_params_get_cost(signal)
    print('predicted best cost is:', controller.predicted_best_cost)
    print('actual cost is:', 10*np.log10(gain))

    filename_seedlight = 'D:/shot noise/'+sf.today_str()+'/try2/seed.csv'
    filename = path_today+'/seedlight_'+str(start_time)+'.csv'
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


    '''
    nums_params_2 = 18

    # get the predicted best parameters and its uncertainty, be ready for the second optimization
    y_pred, sigma_pred = sf.gaussian_interpolation(params=res, m=nums_params_2+2)         

    norm_sigma_pred = np.mean(sigma_pred)

    # calculate the boundary of parameters
    if norm_sigma_pred  <= 1e-2:                                               
        print('the sigma is too tiny,it is:',norm_sigma_pred)
        k = 0.2/norm_sigma_pred
    else:
        k = 2
    
    min_boundary = relu(y_pred - k*np.abs(sigma_pred)).tolist()
    max_boundary = np.maximum(y_pred + k*np.abs(sigma_pred),0.1).tolist()
    n = len(min_boundary)

    # interface  and controller to optimize the noise
    interface_2 = CustomInterface_2(nums_params_2)                            
    controller = mlc.create_controller(interface_2,
                                       #training_type='differential_evolution',
                                       controller_type='neural_net',
                                       max_num_runs=100,
                                       keep_prob=0.8,
                                       num_params=n,
                                       min_boundary=min_boundary,
                                       max_boundary=max_boundary)
    
    # start the second optimization
    controller.optimize()
    
    # save the best wave
    res = sf.add_zero(controller.best_params)
    filename = path_today + '/best_paras_2_' + str(start_time) + '.csv'
    sf.array_to_csv(filename=filename, array=res)

    # save the predicted wave
    res = sf.add_zero(controller.predicted_best_parameters)
    filename = path_today+'/pre_best_paras_2_'+str(start_time)+'.csv'
    sf.array_to_csv(filename=filename, array=res)

    # save the seed light of the predicted wave
    filename = 'E:/python/documents/wave_opt/gausswave.csv'
    sf.array_to_csv(filename=filename, array=res)

    signal = [-1]
    gate = sf.convey_params_get_cost(signal)
    filename_SQ = 'D:/shot noise/' + sf.today_str() + '/try/SQ.csv'
    filename_ASQ = 'D:/shot noise/' + sf.today_str() + '/try/ASQ.csv'
    df_SQ = pd.read_csv(filename_SQ, header=None).values
    df_ASQ = pd.read_csv(filename_ASQ, header=None).values
    res = df_SQ - df_ASQ
    res = res[np.where(np.abs(res-np.mean(res))<3*gate)]
    cost = np.var(res)/np.mean(df_SQ)
    print('predicted best cost_2 is:',controller.predicted_best_cost)
    print('actual cost_2 is:', cost)

    filename_seedlight = 'D:/shot noise/'+sf.today_str()+'/try2/seed.csv'
    filename = path_today+'/seedlight_2_'+str(start_time)+'.csv'
    df = pd.read_csv(filename_seedlight, header=None)
    df.to_csv(filename, header=0, index=0)

    # save the excited stokes light
    filename_excited = 'D:/shot noise/' + sf.today_str() + '/try/excited.csv'
    filename = path_today + '/excited_light_2_' + str(start_time) + '.csv'
    df = pd.read_csv(filename_excited, header=None)
    df.to_csv(filename, header=0, index=0)

    # save the excited anti-stokes light
    filename_excited = 'D:/shot noise/' + sf.today_str() + '/try/antistokes.csv'
    filename = path_today + '/antistokes_2_' + str(start_time) + '.csv'
    df = pd.read_csv(filename_excited, header=None)
    df.to_csv(filename, header=0, index=0)

    # show visualizations
    mlv.show_all_default_visualizations(controller)
    '''

# Ensures main is run when this code is run as a script
if __name__ == '__main__':
    main()
