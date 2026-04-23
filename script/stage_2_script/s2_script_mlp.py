from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting import Setting
# from local_code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
# from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
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

    # ---- objection initialization setction ---------------

    dataset_obj = Dataset_Loader('stage2', '')
    dataset_obj.dataset_source_folder_path = '../../data/stage_2_data/'

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting('pre-partitioned', '')
    # setting_obj = Setting_KFold_CV('k fold cross validation', '')
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    mlp_plot = 'MLP_convergence_plot.png'
    def plot_clean(plot_path):
        if os.path.exists(plot_path):
            os.remove(plot_path)
            print(f'{plot_path} cleaned')
        else:
            print(f'{plot_path} does not exist')

    print('************ Start ************')
    plot_clean(mlp_plot)
    setting_obj.prepare(dataset_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    # result, chud = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    setting_obj.load_run_save_evaluate()
    # for metric, value in result.items():
    #     print(f'  {metric:<20} {value:.4f}')
    print('************ Finish ************')
    # ------------------------------------------------------
    

    