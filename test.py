import torch
import torch.nn as nn
import config

import main

def clean_test(helper, epoch,
           model, is_poison=False, visualize=True, agent_name_key=""):
    model = model.to(config.device)
    model.eval()
    total_loss = 0
    correct = 0
    dataset_size = 0

   
    if helper.params['type'] == config.TYPE_LOAN:
        for i in range(0, len(helper.allStateHelperList)):
            state_helper = helper.allStateHelperList[i]
            data_iterator = state_helper.get_testloader()
            for batch_id, batch in enumerate(data_iterator):
                data, targets = state_helper.get_batch(data_iterator, batch, evaluation=True)
                dataset_size += len(data)
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                          reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    else:
        data_iterator = helper.test_data
        for batch_id, batch in enumerate(data_iterator):
            data, targets = helper.get_batch(data_iterator, batch, evaluation=True)
            dataset_size += len(data)
            output = model(data).to(config.device)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                        reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))  if dataset_size!=0 else 0
    total_l = total_loss / dataset_size if dataset_size!=0 else 0

    main.logger.info('___Test-clean {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                        total_l, correct, dataset_size,
                                                        acc))
    

    model.train()
    return (total_l, acc, correct, dataset_size)



def adv_test(helper, epoch,
                  model, is_poison=False, visualize=True, agent_name_key=""):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    model = model.to(config.device)
   
    
    if helper.params['type'] == config.TYPE_LOAN:
        for i in range(0, len(helper.allStateHelperList)):
            state_helper = helper.allStateHelperList[i]
            data_iterator = state_helper.get_testloader()
            for batch_id, batch in enumerate(data_iterator):
                data, targets, poison_num = state_helper.get_poison_batch(batch, feature_dict=helper.feature_dict,evaluation=True)
                poison_data_count += poison_num
                dataset_size += len(data)
                data = data.to(config.device)
                output = model(data)
                total_loss += nn.functional.cross_entropy(output, targets,
                                                          reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
        
    else:

        data_iterator = helper.test_data_poison
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num = helper.get_poison_batch(batch, adversarial_index=-1, evaluation=True)
            data = data.to(config.device)
            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data).to(config.device)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                        reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(poison_data_count))  if poison_data_count!=0 else 0
    total_l = total_loss / poison_data_count if poison_data_count!=0 else 0
    main.logger.info('___Test-poison {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                        total_l, correct, poison_data_count,
                                                        acc))

       
    model.train()
    return total_l, acc, correct, poison_data_count


def mix_test(helper, epoch,
             model, is_poison=False, visualize=True, agent_name_key=""):
    model.eval()
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    poison_data_count = 0
    data_counts = 0
    model = model.to(config.device)

    if helper.params['type'] == config.TYPE_LOAN:
        for i in range(0, len(helper.allStateHelperList)):
            state_helper = helper.allStateHelperList[i]
            data_iterator = state_helper.get_testloader()
            for batch_id, batch in enumerate(data_iterator):
                data, targets, poison_num, data_count = state_helper.get_mix_batch(batch, feature_dict=helper.feature_dict,
                                                                          evaluation=True)
                poison_data_count += poison_num
                dataset_size += len(data)
                data_counts += data_count
                data = data.to(config.device)
                output = model(data)

                total_loss += nn.functional.cross_entropy(output, targets,
                                                          reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    else:

        data_iterator = helper.test_data_poison
        for batch_id, batch in enumerate(data_iterator):
            data, targets, poison_num, data_count = helper.get_mix_batch(batch, adversarial_index=-1, evaluation=True)
            data = data.to(config.device)
            data_counts += data_count
            poison_data_count += poison_num
            dataset_size += len(data)
            output = model(data).to(config.device)
            main.logger.info('mixed_output:'output)
            total_loss += nn.functional.cross_entropy(output, targets,
                                                      reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(data_counts)) if poison_data_count != 0 else 0
    total_l = total_loss / data_counts if data_counts != 0 else 0
    main.logger.info('___Test-mixed {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                     'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                        total_l, correct, data_counts,
                                                        acc))

    model.train()
    return total_l, acc, correct, poison_data_count