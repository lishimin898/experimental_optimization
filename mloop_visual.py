import mloop.visualizations as mlv

def neuralnet_visualization_from_archieve(datetime):
    file_path = './M-LOOP_archives/'
    #datetime = '2020-06-30_14-54'
    controller_filename = file_path + 'controller_archive_' + datetime + '.txt'
    learner_filename = file_path + 'learner_archive_' + datetime + '.txt'

    mlv.show_all_default_visualizations_from_archive(controller_filename=controller_filename,
                                                     learner_filename=learner_filename,
                                                     controller_type='neural_net')