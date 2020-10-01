import mloop.visualizations as mlv
import matplotlib.pyplot as plt

def neuralnet_visualization_from_archieve(datetime):
    file_path = './M-LOOP_archives/'
    #datetime = '2020-06-30_14-54'
    controller_filename = file_path + 'controller_archive_' + datetime + '.txt'
    learner_filename = file_path + 'learner_archive_' + datetime + '.txt'

    mlv.show_all_default_visualizations_from_archive(controller_filename=controller_filename,
                                                     learner_filename=learner_filename,
                                                     controller_type='neural_net')
    plt.show(）
    
def visualization_from_controller_and_learner_archive(datetime)：
    rootpath='./M-LOOP_archives'
    filetype='txt'
    #date='2020-09-16_16-23'
    controller_archive_filename=rootpath+'/controller_archive_'+date+'.'+filetype
    learner_archive_filename=rootpath+'/learner_archive_'+date+'.'+filetype
    mlv.create_controller_visualizations(controller_archive_filename,
                                    file_type=filetype,
                                    max_parameters_per_plot=2)
    mlv.create_neural_net_learner_visualizations(learner_archive_filename, 
                                                file_type=filetype,
                                                plot_cross_sections=True)
    plt.show()
    
