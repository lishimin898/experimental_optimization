#Imports for M-LOOP
import mloop.interfaces as mli
import mloop.controllers as mlc
import mloop.visualizations as mlv

#Other imports
import numpy as np
import time
import matplotlib.pyplot as plt

#Declare your custom class that inherets from the Interface class
class CustomInterface(mli.Interface):
    
    #Initialization of the interface, including this method is optional
    def __init__(self):
        #You must include the super command to call the parent class, Interface, constructor 
        super(CustomInterface,self).__init__()
        self.l=400  # mm
        self.r=75   # mm 
        
    
    def get_next_cost_dict(self,params_dict):
        
        #Get parameters from the provided dictionary
        params = params_dict['params']
        
        cost=cal_grad(params=params,l=self.l,r=self.r,show_plot=False)

        uncer = 0
        #The evaluation will always be a success
        
        bad = False
        
        cost_dict = {'cost':cost, 'uncer':uncer, 'bad':bad}
        return cost_dict


def cal_grad(params,l=238,r=75,I=1,gama=7e9,show_plot=True):
    params=np.concatenate([-params,params])/1000                  # m
    params=np.expand_dims(params,axis=-1)

    z=np.arange(-int(np.ceil(l/2)),int(np.ceil(l/2))+1)/1000      # m
    r_central=r/2
    low=int(np.floor(l/2-r_central/2))
    up=int(np.ceil(l/2+r_central/2))
    z_center=z[low:up+1]

    d=np.square(params-z)
    d_center=np.square(params-z_center)

    u=4*np.pi*1e-7
    N=1
    r /=1000
    B=u*r**2*N*I/(2*np.power(r**2+d,1.5))
    B=B.sum(axis=0)
    B_center=u*r**2*N*I/(2*np.power(r**2+d_center,1.5))
    B_center=B_center.sum(axis=0)

    Bmax=B_center.max(axis=-1)
    Bmin=B_center.min(axis=-1)
    dy=Bmax-Bmin
    rel_err=dy/Bmax
    grad=dy*gama
    print('grad, rel_err and B are:',grad,rel_err,Bmax)
    if show_plot:
        plt.figure(figsize=(16,9))
        plt.title('B-Z',fontsize=24)
        plt.xlabel('Z',fontsize=20)
        plt.ylabel('B',fontsize=20)
        plt.plot(z,B,c='darkviolet')
        plt.show()
    
    return 10*np.log10(rel_err)

def main():
    #M-LOOP can be run with three commands
    
    #First create your interface
    interface = CustomInterface()
    #Next create the controller. Provide it with your interface and any options you want to set
    n=10
    d=25
    '''
    para=np.linspace(0,interface.l/2-r,n+1)
    low=para[0:n]
    up=para[1:n+1]
    '''
    low=np.array([0]*n)
    up=np.array([interface.l/2-d/2]*n)
    '''
    #z=np.array([0.003, 0.009, 0.009, 0.009, 0.019, 0.021, 0.032, 0.038, 0.043, 0.061, 0.078, 0.079, 0.1, 0.103, 0.103])
    z=np.array([0.0, 0.008, 0.019, 0.034, 0.035, 0.042, 0.063, 0.089, 0.099, 0.099])
    half=0.03
    low=z-half
    low=np.maximum(0,low)
    up=z+half
    up=np.minimum(interface.l/200-r,up)
    '''
    controller = mlc.create_controller(interface, 
                                       #training_type='random',
                                       controller_type='neural_net',
                                       max_num_runs = 2000,
                                       #target_cost = 0,
                                       num_params = n, 
                                       min_boundary = low.tolist(),
                                       max_boundary = up.tolist(),
                                       keep_prob=0.8)
    #To run M-LOOP and find the optimal parameters just use the controller method optimize
    controller.optimize()
    
    #The results of the optimization will be saved to files and can also be accessed as attributes of the controller.
    print('Best parameters found:')
    print(controller.best_params)
    
    #You can also run the default sets of visualizations for the controller with one command
    mlv.show_all_default_visualizations(controller,max_parameters_per_plot=2)
    '''
    mlv.create_neural_net_learner_visualizations(controller.ml_learner.total_archive_filename, 
                                                file_type=controller.ml_learner.learner_archive_file_type,
                                                plot_cross_sections=True)
    plt.show()
    '''
    cal_grad(params=controller.best_params,l=interface.l,r=interface.r)
    cal_grad(params=controller.predicted_best_parameters,l=interface.l,r=interface.r)


main()
