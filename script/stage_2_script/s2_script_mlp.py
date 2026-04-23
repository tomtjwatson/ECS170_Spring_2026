from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting import Setting
# from local_code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
# from local_code.stage_1_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import os

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization section ---------------
    data_obj = Dataset_Loader('train/test', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_source_file_name_train = 'train.csv'
    data_obj.dataset_source_file_name_test = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', '')

    # matplot convergence curve plot
    method_obj.result_destination_folder_path = '../../script/stage_2_script/'

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting('pre-partitioned', '')
    # setting_obj = Setting_KFold_CV('k fold cross validation', '')
    # setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy, f1, precision, recall', '')
    # ------------------------------------------------------

    # clean prev convergence plots
    for plot_file in ['convergence_accuracy.png', 'convergence_loss.png']:
        path = method_obj.result_destination_folder_path + plot_file
        if os.path.exists(path):
            os.remove(path)
            print(f'cleaned {plot_file}')
        else:
            print(f'{plot_file} does not exist')

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    # mean_score, std_score = setting_obj.load_run_save_evaluate()
    scores, x = setting_obj.load_run_save_evaluate()
    print('************ Learned model applied to learning set ************')
    # print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    for metric, value in scores.items():
        print(f'{metric}: {value}')
    print('************ Finish ************')
    # ------------------------------------------------------
    

    