from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP_exp1 import Method_MLP
from local_code.stage_2_code.Method_MLP_exp1 import Method_MLP_exp1
from local_code.stage_2_code.Method_MLP_exp2 import Method_MLP_exp2
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from local_code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------

    train_obj = Dataset_Loader('stage2', '')
    train_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    train_obj.dataset_source_file_name = 'train.csv'

    test_obj = Dataset_Loader('stage2', '')
    test_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    test_obj.dataset_source_file_name = 'test.csv'

    # method_obj = Method_MLP('multi-layer perceptron', '')
    method_obj = Method_MLP_exp1('multi-layer perceptron (exp1)', '')
    # method_obj = Method_MLP_exp2('multi-layer perceptron (exp2)', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_KFold_CV('k fold cross validation', '')
    #setting_obj = Setting_Tra
    # in_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(train_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.test_dataset = test_obj
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    