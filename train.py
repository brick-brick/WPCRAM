import utils.csv_record as csv_record
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import main
import test
import copy
import config
import numpy as np
import foolsgold as fg
from tqdm import tqdm, trange


def FLtrain(helper, start_epoch, local_model, target_model, is_poison, agent_name_keys):
    submit_params_update_dict = dict()
    num_samples_dict = dict()
    client_grads = []

    # device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # 对每个客户进行迭代
    for model_id in range(helper.params['num_models']):
        client_grad = []
        agent_name_key = agent_name_keys[model_id]
        ## Synchronize LR and models
        model = local_model
        model.copy_params(target_model.state_dict())
        # optimizer = torch.optim.Adam(model.parameters(), lr=helper.params['lr'], weight_decay= 0.001)
        # optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'])
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'])
        # optimizer = torch.optim.SGD(model.parameters(), lr= helper.params['lr'], weight_decay= 0.001)
        model.train()

        localmodel_poison_epochs = helper.params['poison_epochs']
        AGENT_POISON_AT_THIS_ROUND = False
        epoch = start_epoch

        if is_poison and agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
            AGENT_POISON_AT_THIS_ROUND = True
            main.logger.info(f'poison local model {agent_name_key} ')



        target_params = dict()
        for name, param in target_model.named_parameters():
            target_params[name] = target_model.state_dict()[name].clone().detach()
            # target_params[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)

        temp_local_epoch = (epoch - 1) * helper.params['internal_epochs']
        for internal_epoch in range(1, helper.params['internal_epochs'] + 1):
            temp_local_epoch += 1

            if helper.params['type'] == config.TYPE_LOAN:
                data_iterator = helper.statehelper_dic[agent_name_key].get_trainloader()
            else:
                _, data_iterator = helper.train_data[agent_name_key]
            total_loss = 0.
            correct = 0
            dataset_size = 0
            poison_data_count = 0 

            model.train()
            # temp_batch_id = 0
            for batch_id, batch in enumerate(tqdm(data_iterator)):
          
                optimizer.zero_grad()

                if helper.params['type'] == config.TYPE_LOAN:
                    if AGENT_POISON_AT_THIS_ROUND:
                        data, targets, poison_num = helper.statehelper_dic[agent_name_key].get_poison_batch(batch, feature_dict=helper.feature_dict,evaluation=False)
                        poison_data_count+= poison_num
                    else: 
                        data, targets = helper.statehelper_dic[agent_name_key].get_batch(data_iterator, batch,evaluation=False)
                else:
                    if AGENT_POISON_AT_THIS_ROUND:
                        data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=-1,evaluation=False)
                        poison_data_count+= poison_num
                    else: 
                        data, targets = helper.get_batch(data_iterator, batch,evaluation=False)

                data = data
                dataset_size += len(data)
                output = model(data)

                loss = nn.functional.cross_entropy(output, targets)

                # main.logger.info(f"output:{output}\n,loss:{loss}\n ")

                loss.backward()

                # main.logger.info(f"batch_id:{batch_id}\n")
                # client_grad = []
                # 最后一个batch
                optimizer.step()
                # if batch_id == 29:
                # MNIST=29,Loan = 62,MNSIT_50=14
                # main.logger.info(f"batch_id:{batch_id}")
                # if batch_id == 29:
                # 从这里注释


                # if batch_id == len(data_iterator):
                if batch_id == 29:
                #     为什么还会取12
                # if batch_id == 29:
                    for name, params in model.named_parameters():
                        # main.logger.info(f"name:{name}\n params:{params}\n")
                        # main.logger.info(f"params.requires_grad:{params.requires_grad}")
                        if params.requires_grad:
                            client_grad.append(params.grad.cpu().clone())
                    client_grads.append(client_grad)


                # 注释结束
                # for name, params in model.named_parameters():
                #     # main.logger.info(f"name:{name}\n params:{params}\n")
                #     # main.logger.info(f"params.requires_grad:{params.requires_grad}")
                #     if params.requires_grad:
                #         client_grad.append(params.grad.cpu().clone())
                # client_grads.append(client_grad)

                # optimizer.step()

                total_loss += loss.data
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
             
              
            acc = 100.0 * (float(correct) / float(dataset_size))
            total_l = total_loss / dataset_size



            if AGENT_POISON_AT_THIS_ROUND:
                main.logger.info(
                '___PoisonTrain {} ,  epoch {:3d}, local model {}, internal_epoch {:3d},Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.4f}%), train_poison_data_count: {}'.format(model.name, epoch, agent_name_key,
                                                                                internal_epoch,
                                                                                total_l, correct, dataset_size,
                                                                                acc, poison_data_count))
                
            else:
                main.logger.info(
                    '___Train {},  epoch {:3d}, local model {}, internal_epoch {:3d},Average loss: {:.4f}, '
                    'Accuracy: {}/{} ({:.4f}%)'.format(model.name, epoch, agent_name_key, internal_epoch, total_l, correct, dataset_size,
                                                        acc))
            csv_record.train_result.append([agent_name_key, temp_local_epoch,
                                            epoch, internal_epoch, total_l.item(), acc, correct, dataset_size])

        
            num_samples_dict[agent_name_key] = dataset_size

            # temp_batch_id = batch_id
            
        # scale: no matter poisoning or not
        if  agent_name_key in helper.params['adversary_list'] and (epoch in localmodel_poison_epochs):
            main.logger.info("scaled!!")
            for name, data in model.state_dict().items():
                data = data
                new_value = target_params[name] + (data - target_params[name]) * helper.params['scale_factor']
                model.state_dict()[name].copy_(new_value)
            
    
        # test local model after internal epochs are finished
        
        epoch_loss, epoch_acc, epoch_corret, epoch_total = test.clean_test(helper=helper, epoch=epoch,
                                                                    model=model, is_poison=AGENT_POISON_AT_THIS_ROUND, visualize=True,
                                                                    agent_name_key=agent_name_key)
        csv_record.test_result.append([agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

     
        
        if AGENT_POISON_AT_THIS_ROUND:
            epoch_loss, epoch_acc, epoch_corret, epoch_total = test.adv_test(helper=helper,
                                                                                    epoch=epoch,
                                                                                    model=model,
                                                                                    is_poison=AGENT_POISON_AT_THIS_ROUND,
                                                                                    visualize=True,
                                                                                    agent_name_key=agent_name_key)
            csv_record.posiontest_result.append(
                [agent_name_key, epoch, epoch_loss, epoch_acc, epoch_corret, epoch_total])

        # update the model params
        client_pramas_update = dict()
        for name, data in model.state_dict().items():
            data = data
            client_pramas_update[name] = torch.zeros_like(data)
            client_pramas_update[name] = (data - target_params[name])
        
        submit_params_update_dict[agent_name_key] = client_pramas_update
        # main.logger.info(f"client_grads:{client_grads}\n client_grads_shape:{len(client_grads)}"
        #                  f"\nclient_pramas_update: {client_pramas_update}")

    return submit_params_update_dict, num_samples_dict, client_grads
