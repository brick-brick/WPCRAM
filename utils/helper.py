from shutil import copyfile
import math
import torch
import logging
import torch.nn.functional as F
import time
logger = logging.getLogger("logger")
import os
import json
import numpy as np
import config
import copy
import utils.csv_record
import foolsgold2
import main


class Helper:
    def __init__(self, current_time, params, name):
        self.current_time = current_time
        self.target_model = None
        self.local_model = None

        self.train_data = None
        self.test_data = None
        self.poisoned_data = None
        self.test_data_poison = None

        self.params = params
        self.name = name
        self.best_loss = math.inf
        self.folder_path = f'saved_models/model_{self.name}_{current_time}'
        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            logger.info('Folder already exists')
        logger.addHandler(logging.FileHandler(filename=f'{self.folder_path}/log.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        logger.info(f'current path: {self.folder_path}')
       
        self.params['current_time'] = self.current_time
        self.params['folder_path'] = self.folder_path
       


    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        if not self.params['save_model']:
            return False
        torch.save(state, filename)

        if is_best:
            copyfile(filename, 'model_best.pth.tar')

    @staticmethod
    def model_global_norm(model):
        squared_sum = 0
        model = model.to(config.device)
        for name, layer in model.named_parameters():
            squared_sum += torch.sum(torch.pow(layer.data.to(config.device), 2))
            # squared_sum += torch.sum(torch.pow(layer.data, 2))
        return math.sqrt(squared_sum)

    @staticmethod
    def clip_weight_norm(model,clip):
        model = model.to(config.device)
        total_norm = Helper.model_global_norm(model)
        logger.info("total_norm: " + str(total_norm)+ "clip_norm: "+str(clip ))
        max_norm = clip
        clip_coef = max_norm / (total_norm + 1e-6)
        current_norm = total_norm
        if total_norm > max_norm:
            for name, layer in model.named_parameters():
                layer.data.to(config.device).mul_(clip_coef)
                # layer.data.mul_(clip_coef)
            current_norm = Helper.model_global_norm(model)
            logger.info("clip~~~ norm after clipping: "+ str(current_norm) )
        return current_norm

  
    @staticmethod
    def dp_noise(param, sigma):

        noised_layer = torch.cuda.FloatTensor(param.shape).normal_(mean=0, std=sigma)

        return noised_layer


    def average_models_params(self, submit_params_update_dict,agent_name_keys,target_model):
        """
        Perform FedAvg algorithm on model params

        """
        target_model = target_model.to(config.device)
        agg_params_update=  dict()
        for name, data in target_model.state_dict().items():
            data = data.to(config.device)
            agg_params_update[name] = torch.zeros_like(data)


        for i in range(0, len(agent_name_keys)):
            client_params_update = submit_params_update_dict[agent_name_keys[i]]
            for name, data in client_params_update.items():
                data = data.to(config.device)
                agg_params_update[name].add_(client_params_update[name] /  self.params["num_models"])


        for name, data in target_model.state_dict().items():

            data = data.to(config.device)
            update_per_layer = agg_params_update[name]  
  
            data.add_(update_per_layer)

    def dp_average_models_params(self, submit_params_update_dict, agent_name_keys, target_model):
        """
        Perform DP-FedAvg algorithm on model params

        """
        # 原论文里用的阈值是5
        M = 5
        target_model = target_model.to(config.device)
        agg_params_update = dict()
        for name, data in target_model.state_dict().items():
            data = data.to(config.device)
            agg_params_update[name] = torch.zeros_like(data)

        for i in range(0, len(agent_name_keys)):
            client_params_update = submit_params_update_dict[agent_name_keys[i]]
            for name, data in client_params_update.items():
                data = data.to(config.device)
                # agg_params_update[name].add_(client_params_update[name] / self.params["num_models"])
                # 根据论文 can you really backdoor federated learning自己写的
                agg_params_update[name].add_(client_params_update[name] / max(1,(torch.norm(client_params_update[name], p =2))/M))

        for name, data in target_model.state_dict().items():
            data = data.to(config.device)
            update_per_layer = agg_params_update[name]

            data.add_(update_per_layer)

    def geometric_median_update(self, target_model, submit_params_update_dict,agent_name_keys, num_samples_dict, foolsgold_weight, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6, max_update_norm= None,
                                if_fg = 1):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
               """
        # target_model = target_model.to(config.device)

        points = []
        alphas = []
        names = []
        for i in range(0, len(agent_name_keys)):
            points.append(submit_params_update_dict[agent_name_keys[i]])
            alphas.append(num_samples_dict[agent_name_keys[i]]*foolsgold_weight[i])
            names.append(agent_name_keys[i])


        # alphas = foolsgold_weight
        # alphas = torch.tensor(foolsgold_weight)
        # points = foolsgold_weight.copy()
        # # 此处初始化的alpha应该改成更新的foolsgold权重？
        alphas = np.asarray(alphas, dtype=np.float64) / sum(alphas)
        # alphas = torch.tensor(foolsgold_weight, dtype=torch.float64) / sum(foolsgold_weight)
        alphas = torch.from_numpy(alphas).float()

        
        # median = Helper.weighted_average_oracle(points, alphas, foolsgold_weight)
        median = Helper.weighted_average_oracle(points, alphas)
        num_oracle_calls = 1

        # logging
        obj_val = Helper.geometric_median_objective(median, points, alphas)
        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append(log_entry)
        if verbose:
            logger.info('Starting Weiszfeld algorithm')
            logger.info(log_entry)
        logger.info(f'[rfa agg] init. name: {names}, weight: {alphas}')
        # start
        wv=None
        # eps = 1e-
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = torch.tensor([alpha / max(eps, Helper.l2dist(median, p)) for alpha, p in zip(alphas, points)],
                                 dtype=alphas.dtype)

            weights = weights / weights.sum()
            median = Helper.weighted_average_oracle(points, weights)
            num_oracle_calls += 1
            obj_val = Helper.geometric_median_objective(median, points, alphas)
            log_entry = [i + 1, obj_val,
                         (prev_obj_val - obj_val) / obj_val,
                         Helper.l2dist(median, prev_median)]
            logs.append(log_entry)
            if verbose:
                logger.info(log_entry)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
            logger.info(f'[rfa agg] iter:  {i}, prev_obj_val: {prev_obj_val}, obj_val: {obj_val}, abs dis: { abs(prev_obj_val - obj_val)}')
            logger.info(f'[rfa agg] iter:  {i}, weight: {weights}')
            wv=copy.deepcopy(weights)
        alphas = [Helper.l2dist(median, p) for p in points]

        update_norm = 0
        for name, data in median.items():
            data = data.to(config.device)
            update_norm += torch.sum(torch.pow(data, 2))
        update_norm= math.sqrt(update_norm)

        if max_update_norm is None or update_norm < max_update_norm:
            for name, data in target_model.state_dict().items():
                data = data.to(config.device)
                update_per_layer = median[name] 
                data.add_(update_per_layer)
            is_updated = True
        else:
            logger.info('\t\t\tUpdate norm = {} is too large. Update rejected'.format(update_norm))
            is_updated = False

        utils.csv_record.add_weight_result(names, wv.cpu().numpy().tolist(), alphas)
        logger.info(f'wv:  {wv}, wv_shape:{wv.shape}')
        return num_oracle_calls, is_updated, names, wv.cpu().numpy().tolist(),alphas

    def fg_geo_update(self, target_model, submit_params_update_dict, agent_name_keys, num_samples_dict,
                                foolsgold_weight):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
               """
        # target_model = target_model.to(config.device)

        points = []
        alphas = []
        names = []
        agg_params_update = dict()
        for name, data in target_model.state_dict().items():
            data = data.to(config.device)
            agg_params_update[name] = torch.zeros_like(data)

        for i in range(0, len(agent_name_keys)):
            points.append(submit_params_update_dict[agent_name_keys[i]])
            alphas.append(num_samples_dict[agent_name_keys[i]] * foolsgold_weight[i])
            names.append(agent_name_keys[i])

        alphas = np.asarray(alphas, dtype=np.float64) / sum(alphas)
        # alphas = torch.tensor(foolsgold_weight, dtype=torch.float64) / sum(foolsgold_weight)
        alphas = torch.from_numpy(alphas).float()

        for i in range(0, len(agent_name_keys)):
            client_params_update = submit_params_update_dict[agent_name_keys[i]]
            for name, data in target_model.state_dict().items():
                # data = data.to(config.device)
                agg_params_update[name].add_(client_params_update[name] * alphas[i])

        for name, data in target_model.state_dict().items():
            data = data.to(config.device)
            update_per_layer = agg_params_update[name]
            data.add_(update_per_layer)
        is_updated = True


        utils.csv_record.add_weight_result(names, alphas.cpu().numpy().tolist(), alphas)
        return  is_updated, names, alphas




    @staticmethod
    def l2dist(p1, p2):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        squared_sum = 0
        for name, data in p1.items():
            data = data.to(config.device)
            squared_sum += torch.sum(torch.pow(p1[name]- p2[name], 2))
        return math.sqrt(squared_sum)


    @staticmethod
    def geometric_median_objective(median, points, alphas):
        """Compute geometric median objective."""
        temp_sum= 0
        for alpha, p in zip(alphas, points):
            temp_sum += alpha * Helper.l2dist(median, p)
        return temp_sum

     

    @staticmethod
    def weighted_average_oracle(points, weights):
        """Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        tot_weights = torch.sum(weights)

        weighted_updates= dict()

        for name, data in points[0].items():
            data = data.to(config.device)
            weighted_updates[name]=  torch.zeros_like(data)
        for w, p in zip(weights, points): # 对每一个agent
            for name, data in weighted_updates.items():
                data = data.to(config.device)
                temp = (w / tot_weights).float().to(config.device)
                temp= temp* (p[name].float())
                # temp = w / tot_weights * p[name]
                if temp.dtype!=data.dtype:
                    temp = temp.type_as(data)
                data.add_(temp)

        return weighted_updates

    def save_model(self, model=None, epoch=0, val_loss=0):

        if model is None:
            model = self.target_model.to(config.device)
        model = model.to(config.device)
        if self.params['save_model']:
            # save_model
            # logger.info("saving model")
            model_name = '{0}/model_last.pt.tar'.format(self.params['folder_path'])
            saved_dict = {'state_dict': model.state_dict(), 'epoch': epoch,
                          'lr': self.params['lr']}
            self.save_checkpoint(saved_dict, False, model_name)
            # if epoch in self.params['save_on_epochs']:
            if self.params['type'] == config.TYPE_LOAN:
                if epoch % 2 ==0:
                    logger.info(f'Saving model on epoch {epoch}')
                    self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            else:
                if epoch % 5 ==0:
                    logger.info(f'Saving model on epoch {epoch}')
                    self.save_checkpoint(saved_dict, False, filename=f'{model_name}.epoch_{epoch}')
            if val_loss < self.best_loss:
                self.save_checkpoint(saved_dict, False, f'{model_name}.best')
                self.best_loss = val_loss

  

  
