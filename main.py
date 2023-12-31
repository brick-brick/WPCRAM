import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import json
import datetime
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import time
import numpy as np
import random
import config
import copy
import foolsgold2 as fg
# import foolsgold3 as fg


import train
import train2
# import train5
import test
from utils.image_helper import ImageHelper
from utils.loan_helper import LoanHelper
import utils.csv_record as csv_record


logger = logging.getLogger("logger")
# set random seeds
torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)
np.random.seed(1)


if __name__ == '__main__':


    # load hyperparameters
    parser = argparse.ArgumentParser(description='CRFL')
    parser.add_argument('--params')
    #parser.add_argument('--params', description='params file to be loaded')
    args = parser.parse_args()
    with open(f'./{args.params}', 'r') as f:
        params_loaded = yaml.load(f,Loader=yaml.FullLoader)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')
    iffg_smooth = 0
    # load data
    if params_loaded['type'] == config.TYPE_LOAN:
        iffg_smooth = 1
        helper = LoanHelper(current_time=current_time, params=params_loaded,
                            name=params_loaded.get('name', 'loan'))
        helper.load_data(params_loaded)

    elif params_loaded['type'] == config.TYPE_MNIST:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'mnist'))
        helper.load_data()
    elif params_loaded['type'] == config.TYPE_EMNIST:
        helper = ImageHelper(current_time=current_time, params=params_loaded,
                             name=params_loaded.get('name', 'emnist'))
        helper.load_data()

    else:
        helper = None

    logger.info(f'load data done')

    # create model
    helper.create_model()

    # save parameters:
    with open(f'{helper.folder_path}/params.yaml', 'w') as f:
        yaml.dump(helper.params, f)


    if helper.params['is_poison']:
        logger.info(f"Poisoned following participants: {(helper.params['adversary_list'])}")

    # Memory = None
    memory = None
    epo = helper.params['epochs']
    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # 开始每一个epoch的迭代
    for epoch in range(helper.start_epoch, helper.params['epochs'] + 1):
        start_time = time.time()
        t = time.time()
        agent_name_keys = helper.participants_list

        # update local models
        # LOAN训练
        submit_params_update_dict, num_samples_dict, client_grads= train2.FLtrain(
            helper=helper,
            start_epoch=epoch,
            local_model=helper.local_model.to(config.device),
            target_model=helper.target_model.to(config.device),
            is_poison=helper.params['is_poison'],
            agent_name_keys=agent_name_keys)


        # break
        # client_grads = torch.tensor([item.detach().numpy() for item in client_grads])
        # grad_len = np.array(client_grads[0][-2].data.numpy().shape).prod()
        grad_len = np.array(client_grads[0][-2].cpu().data.numpy().shape).prod()
        # train5时候用下面这句
        # num_clients = len(client_grads)
        # memory更新
        # 的时候用下面这句
        num_clients = helper.params['num_models']
        if memory is None:
            # memory = np.zeros((helper.params['num_models'], grad_len))
            memory = np.zeros((num_clients, grad_len))


        grads = np.zeros((num_clients, grad_len))
        for i in range(len(client_grads)):
        #     上面这个是配合train5时候用的
        # for i in range(len(client_grads)):
            grads[i] = np.reshape(client_grads[i][-2].cpu().data.numpy(), (grad_len))
        #     第i个client_grads的倒数第二行的数据
        memory += grads

        # 权重更新,v是最高相似度

        # foolsgold_weight, v, mini_v__ind, num = fg.foolsgold(memory)  # Use FG
        # foolsgold_weight, v, mini_v__ind, num = fg.foolsgold(grads)  # Use FG
        foolsgold_weight, v, mini_v__ind, num = fg.foolsgold(memory, iffg_smooth)  # Use FG
        logger.info(
            # f'foolsgold尺寸为{num}, memory:{memory}\n, foolsgold_weight: {foolsgold_weight}\n'
            f'foolsgold尺寸为{num}, foolsgold_weight: {foolsgold_weight}\n'
            # f'memory:{memory},\n client_grads:{client_grads},\n'
            f'grads:{grads},\n'
            f'v:{v}\n,最小v为第{mini_v__ind}个客户')
        # logger.info(
        #     f'memory:{memory},\n client_grads:{client_grads}, \ngrads: {grads}, foolsgold_weight: {foolsgold_weight}\n'
        #     f'v:{v}\n,v的尺寸为{v.shape}\n')
        # 增添结束


        is_updated = True
        # 聚合
        # sever aggregation
        if helper.params['aggregation_methods'] == config.AGGR_MEAN_PARAM:
            helper.average_models_params(submit_params_update_dict,agent_name_keys, target_model=helper.target_model.to(config.device))

        elif helper.params['aggregation_methods'] == config.AGGR_GEO_MED:
            maxiter = config.geom_median_maxiter
            # logger.info(f'num_samples_dict:{num_samples_dict}')
            num_oracle_calls, is_updated, names, weights, alphas = helper.geometric_median_update(helper.target_model.to(config.device), submit_params_update_dict, agent_name_keys, num_samples_dict, foolsgold_weight, maxiter=maxiter)

        # clip the global model
        if params_loaded['type'] == config.TYPE_MNIST:
            dynamic_thres= epoch *0.1+2
        elif params_loaded['type'] == config.TYPE_LOAN:
            dynamic_thres = epoch*0.025+2
        elif params_loaded['type'] == config.TYPE_EMNIST:
            dynamic_thres= epoch*0.25+4
        param_clip_thres =  helper.params["param_clip_thres"]
        if dynamic_thres < param_clip_thres:
            param_clip_thres= dynamic_thres

        current_norm = helper.clip_weight_norm(helper.target_model.to(config.device), param_clip_thres )
        csv_record.add_norm_result(current_norm)

        # test acc after clipping
        epoch_loss, epoch_acc, epoch_corret, epoch_total = test.clean_test(helper=helper, epoch=epoch,
                                                                       model=helper.target_model.to(config.device), is_poison=False,
                                                                       visualize=True, agent_name_key="global")
        csv_record.test_result.append(["global", epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])
        csv_record.add_global_acc_result(epoch_acc)

        # # 用的混合数据集进行测试
        # if helper.params['is_poison'] and epoch >= helper.params['poison_epochs'][0]:
        #     epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.mix_test(helper=helper,
        #                                                                             epoch=epoch,
        #                                                                             model=helper.target_model.to(config.device),
        #                                                                             is_poison=True,
        #                                                                             visualize=True,
        #                                                                             agent_name_key="global")
        #
        #     csv_record.posiontest_result.append(
        #         ["global", epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])
        #     原本的adv_test
        if helper.params['is_poison'] and epoch >= helper.params['poison_epochs'][0]:
            epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.adv_test(helper=helper,
                                                                                    epoch=epoch,
                                                                                    model=helper.target_model.to(config.device),
                                                                                    is_poison=True,
                                                                                    visualize=True,
                                                                                    agent_name_key="global")

            csv_record.posiontest_result.append(
                ["global", epoch, epoch_loss, epoch_acc_p, epoch_corret, epoch_total])

        # save model
        helper.save_model(epoch=epoch, val_loss=epoch_loss)
        # save csv file
        csv_record.save_result_csv(epoch, helper.folder_path)

        # add noise
        logger.info(f" epoch: {epoch} add noise on the global model!")
        for name, param in helper.target_model.state_dict().items():
            param.add_(helper.dp_noise(param, helper.params['sigma_param']))
        #
        # if epoch == 1:
        #     break

    logger.info(f"This run has a label: {helper.params['current_time']}. ")

