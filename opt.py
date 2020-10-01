#Imports for python 2 compatibility
from __future__ import absolute_import, division, print_function
__metaclass__ = type

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
        
    
    def get_next_cost_dict(self,params_dict):
        
        #Get parameters from the provided dictionary
        params = params_dict['params']
        params = np.round(params,decimals=4)
        cost=self.cal_grad(params=params)

        uncer = 0
        #The evaluation will always be a success
        if cost<0:
            bad=True
        else:
            bad = False
        
        cost_dict = {'cost':cost, 'uncer':uncer, 'bad':bad}
        return cost_dict

    def cal_grad(self,params):
        l=30
        r=0.0756/2
        #n=20
        u=4*np.pi*1e-7
        N=1
        I=1
        params=np.round(params,decimals=4)
        inv_para=l/100-params[-1::-1]
        z=np.concatenate([params,inv_para])
        res=[]
        for s in range(0,l+1):
            s/=100
            d=np.square(z-s)
            b=np.sum(u*r**2*N/(2*np.power(r**2+d,1.5)))*I
            res.append(b)
        array=res[12:19]
        return (np.max(array)-np.min(array))*7e9

def cal_grad(params):
        l=30
        r=0.0756/2
        #n=20
        u=4*np.pi*1e-7
        N=1
        I=1
        params=np.round(params,decimals=4)
        inv_para=l/100-params[-1::-1]
        z=np.concatenate([params,inv_para])
        print('z is:',z.tolist())
        res=[]
        x=[]
        for s in range(0,l+1):
            s/=100
            x.append(s)
            d=np.square(z-s)
            b=np.sum(u*r**2*N/(2*np.power(r**2+d,1.5)))*I
            res.append(b)
        array=res[12:19]
        dy=np.max(array)-np.min(array)
        B=np.mean(array)
        grad=dy*7e9
        print('grad is:',grad)
        print('ralative grad is:',dy/B)
        plt.figure(figsize=(16,9))
        plt.title('B-Z',fontsize=24)
        plt.xlabel('Z',fontsize=20)
        plt.ylabel('B',fontsize=20)
        plt.plot(x,res,c='darkviolet')
        plt.show()
        return grad
    
def main():
    #M-LOOP can be run with three commands
    
    #First create your interface
    interface = CustomInterface()
    #Next create the controller. Provide it with your interface and any options you want to set
    n=6
    para=np.linspace(0,0.15,n+1)
    low=para[0:n]
    up=para[1:n+1]
    controller = mlc.create_controller(interface, 
                                        #training_type='random',
                                        controller_type='neural_net',
                                       max_num_runs = 6000,
                                       target_cost = 0,
                                       num_params = n, 
                                       min_boundary = low.tolist(),
                                       max_boundary = up.tolist(),
                                       keep_prob=1)
    #To run M-LOOP and find the optimal parameters just use the controller method optimize
    controller.optimize()
    
    #The results of the optimization will be saved to files and can also be accessed as attributes of the controller.
    print('Best parameters found:')
    print(controller.best_params)
    
    #You can also run the default sets of visualizations for the controller with one command
    mlv.show_all_default_visualizations(controller)
    
    mlv.create_neural_net_learner_visualizations(controller.ml_learner.total_archive_filename, 
                                                file_type=controller.ml_learner.learner_archive_file_type,
                                                plot_cross_sections=True)
    plt.show()

    cal_grad(controller.best_params)
    cal_grad(controller.predicted_best_parameters)
    

#Ensures main is run when this code is run as a script
if __name__ == '__main__':
    main()