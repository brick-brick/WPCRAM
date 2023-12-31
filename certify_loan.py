import matplotlib
import logging
logger = logging.getLogger("logger")
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np


from typing import *
import pandas as pd
import seaborn as sns
import math
import logging
logger = logging.getLogger("logger")
sns.set()


import yaml
from scipy.stats import norm



class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()



class CertifiedRate(Accuracy):
    def __init__(self, smoothed_fname,agg_weight=None,M=0,alpha= 0):
        cert_bound, cert_bound_exp, is_acc = certify(smoothed_fname,agg_weight=agg_weight,M=M,alpha= alpha)
        print('Rr:')
        print(cert_bound)
        self.cert_bound = cert_bound
        self.cert_bound_exp = cert_bound_exp
        self.is_acc = is_acc

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        print(f'certified_rate:{np.array([self.at_radius(radius) for radius in radii])},shape:{np.array([self.at_radius(radius) for radius in radii]).shape}')
        count = 0
        temp = 0
        for i in np.array([self.at_radius(radius) for radius in radii]):
            if i!= 0:
                count = count + 1
                temp = temp + i

        print(f'certified_rate_mean:{np.array([self.at_radius(radius) for radius in radii]).mean()}')
        print(f'rate_mean:{temp/count}')
        # indx = -1
        # for i in range(len(radii)):
        #     if radii[i] <= 0:
        #         indx = i - 1
        #         break
        print(f'radius[0] certified_rate:{self.at_radius(radii[0])}, radius[] rate:{self.at_radius(math.floor(self.cert_bound[0]))}')
        # print(f'radius[0] certified_rate:{self.at_radius(radii[0])}, radius[] rate:{self.at_radius(radii[376.4652252])}')
        return np.array([self.at_radius(radius) for radius in radii])

    def at_radius(self, radius: float):
        return (self.cert_bound  >= radius).mean()

class CertifiedAcc(Accuracy):
    def __init__(self, smoothed_fname, agg_weight=None,M=0,alpha= 0):
        cert_bound, cert_bound_exp, is_acc = certify(smoothed_fname,agg_weight=agg_weight, M=M,alpha= alpha)
        self.cert_bound = cert_bound
        self.cert_bound_exp = cert_bound_exp
        self.is_acc = is_acc

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        print(f'certified_acc:{np.array([self.at_radius(radius) for radius in radii])}, shape:{np.array([self.at_radius(radius) for radius in radii]).shape}')
        print(f'certified_acc_mean:{np.array([self.at_radius(radius) for radius in radii]).mean()}')
        count = 0
        temp = 0
        for i in np.array([self.at_radius(radius) for radius in radii]):
            if i!= 0:
                count = count + 1
                temp = temp + i

        print(f'certified_acc_mean:{np.array([self.at_radius(radius) for radius in radii]).mean()}')
        print(f'acc_mean:{temp/count}')


        print(f'radius[0] certified_acc:{self.at_radius(radii[0])}, radius[] acc:{self.at_radius(math.floor(self.cert_bound[0]))}')
        return np.array([self.at_radius(radius) for radius in radii])

    def at_radius(self, radius: float):
        return (np.logical_and(self.cert_bound>=radius, self.is_acc)).mean()


class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x


def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.001) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    # main.logger.info(f"RAD:{max_radius}")
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("radius", fontsize=16)
    plt.ylabel("certified accuracy", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_certified_rate(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.001) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("radius", fontsize=16)
    plt.ylabel("certified rate", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=16)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
   
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def cal_prob_bound(pa, pb, sigma_test,epoch, training_params,agg_weight=None):
    sigma_train = training_params['sigma_param']
    sigma_test= sigma_test
    eta = training_params['lr'] # lr 
    T = epoch # epoch 
    N =  training_params['num_models']  # number of clients
    R = len(training_params['adversary_list'])  # number of R
    q_B = training_params['poisoning_per_batch']  # poison per batch
    n_B = training_params['batch_size']  # poison per batch
    gamma= training_params['scale_factor'] #scale
    tau =  int(28628/n_B)
    rho_tadv= 2.15 # attack at round 6
    L_z =  math.sqrt(2+2*rho_tadv+rho_tadv**2)
    if agg_weight==None:
        agg_weight= []
        for i in range(0,R):
            agg_weight.append(float(1/N))
    weighted_avg =0   
    for i in range(0,R):
        weighted_avg+= agg_weight[i]**2
    t_adv= training_params['poison_epochs'][0]
    if pa==1.0:
        return 100000
    fraction= - math.log(1- (math.sqrt(pa)-math.sqrt(pb))**2) * sigma_train**2
    denominator= 2* R* tau**2 *L_z**2 * weighted_avg * gamma**2 * eta**2 * float(q_B**2 / n_B**2 ) 
    contract=1
    for _epoch in range(t_adv+1, T): # from round t_adv+1 to round T-1 
        rho_t = _epoch *0.025+2
        contract *= 2*norm.cdf(rho_t*1.0/sigma_train)-1 
    rho_T = T *0.025+2
    contract  *= (2*norm.cdf(rho_T*1.0/sigma_test)-1) # round T
    denominator= denominator * contract
    delta_pat  = math.sqrt(fraction/  denominator)
    logger.info(f'N:{N}, R:{R}, certified_R:{delta_pat}')

    return delta_pat 


def certify(smoothed_fname,agg_weight=None, M=0, alpha= 0):

    foldername= smoothed_fname.split('/')
    epoch = int(foldername[-1].split('_')[-1])

    foldername = os.path.join(foldername[0],foldername[1])

    training_param_fname= os.path.join(foldername,'params.yaml')
    with open(training_param_fname, 'r') as f:
        training_params = yaml.load(f, Loader=yaml.FullLoader)
    print(training_params)

    if M==0:
        M= params_loaded['N_m']
    if alpha==0:
        alpha = params_loaded['alpha']
    
    # data_file_path =  os.path.join(foldername, "pred_poison_Epoch%dM%dSigma%.4f.txt"%(epoch,params_loaded['N_m'], params_loaded['test_sigma']))
    data_file_path =  os.path.join(foldername, "pred_clean_Epoch%dM%dSigma%.4f.txt"%(epoch,M, params_loaded['test_sigma']))

    num_samples= 10000
    df = pd.read_csv(data_file_path, delimiter="\t")
    print(len(np.array(df["pa_exp"])))
    # pa,pb是每一条数据的预测标签概率第一与第二大的上下界，is_acc表示预测是否正确
    pa_exp = np.array(df["pa_exp"])[:num_samples]
    pb_exp = np.array(df["pb_exp"])[:num_samples]
    is_acc = np.array(df["is_acc"])[:num_samples]

    heof_factor = np.sqrt(np.log(1/alpha)/2/M)
    pa = np.maximum(1e-8, pa_exp - heof_factor) # [num_samples]
    pb = np.minimum(1-1e-8, pb_exp + heof_factor) # [num_samples]

    # Calculate the metrics
    cert_bound= np.zeros_like(pa)
    cert_bound_exp = np.zeros_like(pa)
    for i in range(len(pa)):
        cert_bound[i]  = cal_prob_bound(pa=pa[i], pb=pb[i],sigma_test=params_loaded['test_sigma'], epoch=epoch, training_params=training_params,agg_weight=agg_weight )
        cert_bound_exp[i]  = cal_prob_bound(pa=pa_exp[i], pb=pb_exp[i],sigma_test=params_loaded['test_sigma'], epoch=epoch, training_params =training_params,agg_weight=agg_weight )
    return cert_bound, cert_bound_exp, is_acc






if __name__ == "__main__":
    with open(f'./configs/loan_smooth_params.yaml', 'r') as f:
        params_loaded = yaml.load(f, Loader=yaml.FullLoader)
  

    #  ### vary N  
    # plot_certified_rate(
    #     "plots/loan/vary_N_tadv6_T100_cer_rate", "vary N ($t_{adv}=6$, T=100, R=1, $\gamma=10$)", 6, [
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.40.57/model_last.pt.tar.epoch_100"), "N = 10"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), "N = 20"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.41.10/model_last.pt.tar.epoch_100"), "N = 30"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.41.18/model_last.pt.tar.epoch_100"), "N = 40"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.41.25/model_last.pt.tar.epoch_100"), "N = 50"),

    #     ])
    # plot_certified_accuracy(
    #      "plots/loan/vary_N_tadv6_T100_cer_acc", "vary N ($t_{adv}=6$, T=100, R=1, $\gamma=10$)", 6, [
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.40.57/model_last.pt.tar.epoch_100"), "N = 10"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), "N = 20"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.41.10/model_last.pt.tar.epoch_100"), "N = 30"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.41.18/model_last.pt.tar.epoch_100"), "N = 40"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.41.25/model_last.pt.tar.epoch_100"), "N = 50"),
    #     ])

    # #  #### noise
    # plot_certified_rate(
    #     "plots/loan/vary_sigma_T100_cer_rate", "vary $\sigma$ ($t_{adv}=6$, T=100, $\gamma=10$, R=1)", 6, [
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.42.36/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.005"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.010"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.42.44/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.015"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.42.50/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.020"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.42.56/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.025"),
    #     ])
    # plot_certified_accuracy(
    #     "plots/loan/vary_sigma_T100_cer_acc", "vary $\sigma$ ($t_{adv}=6$, T=30, $\gamma=100$, R=1)", 6, [
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.42.36/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.005"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.010"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.42.44/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.015"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.42.50/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.020"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.42.56/model_last.pt.tar.epoch_100"), " $\sigma$ = 0.025"),
    #     ])

    
    
    #### vary T 
    # plot_certified_rate(
    #     "plots/loan/vary_T_tadv6_cer_rate", "vary T ($t_{adv}=6$, R=1, $\gamma=10$)", 2.5, [
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_30"), "T = 30"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_50"), "T = 50"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_60"), "T = 60"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_80"), "T = 80"),
    #      
            
    #     ])
    # plot_certified_accuracy(
    #     "plots/loan/vary_T_tadv6_cer_acc", "vary T ($t_{adv}=6$, R=1, $\gamma=10$)", 2.5, [
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_30"), "T = 30"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_50"), "T = 50"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_60"), "T = 60"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_80"), "T = 80"),
    #   
    #     ])
    

    # #### vary R 
    # plot_certified_rate(
    #     "plots/loan/vary_R_tadv6_T100_cer_rate", "vary R ($t_{adv}=6$, T=100, $\gamma=10$)", 2.5, [
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " R = 1"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.38.30/model_last.pt.tar.epoch_100"), " R = 2"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.38.40/model_last.pt.tar.epoch_100"), " R = 3"),
    #     ])

    # plot_certified_accuracy(
    #     "plots/loan/vary_R_tadv6_T100_cer_acc", "vary R ($t_{adv}=6$, T=100, $\gamma=10$)", 2.5, [
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " R = 1"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.38.30/model_last.pt.tar.epoch_100"), " R = 2"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.38.40/model_last.pt.tar.epoch_100"), " R = 3"),
    #     ])
    #


    # # robust RFA 
    # plot_certified_accuracy(
    # "plots/loan/vary_agg_tadv6_T100_cer_acc", "vary R ($t_{adv}=6$, T=100, $\gamma=10$)", 18, [
    #     Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.49.17/model_last.pt.tar.epoch_100",agg_weight=[0.0104]), " R = 1, RFA"),
    #     Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.49.52/model_last.pt.tar.epoch_100",agg_weight=[0.0086, 0.0094]), " R = 2, RFA"),
    #     Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.50.09/model_last.pt.tar.epoch_100",agg_weight=[2.3699e-03, 2.5833e-03, 2.5176e-03]), " R = 3, RFA"),
    # ])
    # plot_certified_rate(
    # "plots/loan/vary_agg_tadv6_T100_cer_rate", "vary R ($t_{adv}=6$, T=100, $\gamma=10$)", 18, [
    #     Line(CertifiedRate("saved_models/model_loan_Feb.04_04.49.17/model_last.pt.tar.epoch_100",agg_weight=[0.0104]), " R = 1, RFA"),
    #     Line(CertifiedRate("saved_models/model_loan_Feb.04_04.49.52/model_last.pt.tar.epoch_100",agg_weight=[0.0086, 0.0094]), " R = 2, RFA"),
    #     Line(CertifiedRate("saved_models/model_loan_Feb.04_04.50.09/model_last.pt.tar.epoch_100",agg_weight=[2.3699e-03, 2.5833e-03, 2.5176e-03]), " R = 3, RFA"),
    # ])


    # # robust RFA
    # plot_certified_accuracy(
    # "plots/loan/vary_agg_tadv6_T100_cer_acc", "vary R ($t_{adv}=6$, T=100, $\gamma=10$)", 18, [
    #     Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.49.17/model_last.pt.tar.epoch_100",agg_weight=[0.0104]), " R = 1, RFA"),
    #     Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.49.52/model_last.pt.tar.epoch_100",agg_weight=[0.0086, 0.0094]), " R = 2, RFA"),
    #     Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.50.09/model_last.pt.tar.epoch_100",agg_weight=[2.3699e-03, 2.5833e-03, 2.5176e-03]), " R = 3, RFA"),
    # ])
    # plot_certified_rate(
    # "plots/loan/vary_agg_tadv6_T100_cer_rate", "vary R ($t_{adv}=6$, T=100, $\gamma=10$)", 18, [
    #     Line(CertifiedRate("saved_models/model_loan_Feb.04_04.49.17/model_last.pt.tar.epoch_100",agg_weight=[0.0104]), " R = 1, RFA"),
    #     Line(CertifiedRate("saved_models/model_loan_Feb.04_04.49.52/model_last.pt.tar.epoch_100",agg_weight=[0.0086, 0.0094]), " R = 2, RFA"),
    #     Line(CertifiedRate("saved_models/model_loan_Feb.04_04.50.09/model_last.pt.tar.epoch_100",agg_weight=[2.3699e-03, 2.5833e-03, 2.5176e-03]), " R = 3, RFA"),
    # ])

    # robust RFA
    # plot_certified_accuracy(
    # "plots/loan/vary_agg_tadv6_T100_cer_acc", "vary R ($t_{adv}=6$, T=100, $\gamma=10$)", 18, [
    #     Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.49.17/model_last.pt.tar.epoch_100",agg_weight=[0.0104]), " R = 1, RFA"),
    #     Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.49.52/model_last.pt.tar.epoch_100",agg_weight=[0.0086, 0.0094]), " R = 2, RFA"),
    #     Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.50.09/model_last.pt.tar.epoch_100",agg_weight=[2.3699e-03, 2.5833e-03, 2.5176e-03]), " R = 3, RFA"),
    # ])
    # plot_certified_rate(
    # "plots/loan/vary_agg_tadv6_T100_cer_rate", "vary R ($t_{adv}=6$, T=100, $\gamma=10$)", 1000, [
    #     Line(CertifiedRate("saved_models/model_loan_May.06_18.20.10/model_last.pt.tar.epoch_100",agg_weight=[0.01238, 0.01568, 0.04513, 0.03619]), " R = 4, CRFL"),
    #     Line(CertifiedRate("saved_models/model_loan_Feb.06_09.16.35/model_last.pt.tar.epoch_100",agg_weight=[0.00004591, 0.00003566, 0.00001527, 0.00001995]), " R = 4, MDCR"),
    # ])
    plot_certified_rate(
    "plots/loan/vary_agg_tadv6_T100_cer_rate", "vary R ($t_{adv}=6$, T=800, $\gamma=10$)", 1000, [
        # Line(CertifiedRate("saved_models/model_loan_Oct.26_10.26.45/model_last.pt.tar.epoch_40",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
        Line(CertifiedRate("saved_models/model_loan_Oct.25_20.26.59/model_last.pt.tar.epoch_100",agg_weight=[1.0027e-06, 1.2876e-06, 3.1768e-06, 2.6025e-06]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.13_20.44.43/model_last.pt.tar.epoch_100",agg_weight=[1.4281e-05, 1.8850e-05, 4.7604e-05, 3.7737e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.12_21.22.53/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.12_13.33.16/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.11_21.33.44/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.02_16.19.10/model_last.pt.tar.epoch_100",agg_weight=[9.6407e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.02_09.37.49/model_last.pt.tar.epoch_100",agg_weight=[9.6407e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.11_15.36.23/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.10_16.53.31/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.09_12.34.00/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.08_22.31.14/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.07_09.24.45/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.06_17.45.26/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.05_19.42.47/model_last.pt.tar.epoch_100",agg_weight=[9.6407e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.04_21.42.23/model_last.pt.tar.epoch_100",agg_weight=[9.6407e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.01_22.07.01/model_last.pt.tar.epoch_100",agg_weight=[9.6407e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.01_15.59.47/model_last.pt.tar.epoch_100",agg_weight=[9.6407e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.01_09.48.18/model_last.pt.tar.epoch_100",agg_weight=[9.6408e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Dec.01_09.48.18/model_last.pt.tar.epoch_100",agg_weight=[9.6408e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.25_20.26.59/model_last.pt.tar.epoch_100",agg_weight=[1.0155e-06, 1.1213e-06, 3.0843e-06, 2.5144e-06]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Nov.30_23.36.10/model_last.pt.tar.epoch_100",agg_weight=[1.0155e-06, 1.1213e-06, 3.0843e-06, 2.5144e-06]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Nov.30_16.59.21/model_last.pt.tar.epoch_100",agg_weight=[1.8008e-06, 1.9700e-06, 5.2551e-06, 4.4257e-06]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Nov.30_10.05.53/model_last.pt.tar.epoch_100",agg_weight=[3.6387e-06, 3.9805e-06, 1.0204e-05, 9.0297e-06]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Nov.29_19.59.12/model_last.pt.tar.epoch_100",agg_weight=[7.8739e-06, 8.8277e-06, 2.1621e-05, 2.0654e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Nov.28_21.07.45/model_last.pt.tar.epoch_100",agg_weight=[1.0272e-05, 1.2332e-05, 3.0118e-05, 2.7374e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Nov.27_22.29.36/model_last.pt.tar.epoch_140",agg_weight=[6.7846e-06, 6.9319e-06, 1.6027e-05, 1.6193e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Nov.27_10.51.28/model_last.pt.tar.epoch_120",agg_weight=[1.5002e-05, 1.7055e-05, 3.9458e-05, 3.7011e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.21_10.09.12/model_last.pt.tar.epoch_100",agg_weight=[1.0528e-04, 1.2101e-04, 2.2673e-04, 2.0182e-04]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.19_21.46.26/model_last.pt.tar.epoch_100",agg_weight=[9.2084e-06, 1.0444e-05, 1.7731e-05, 1.6410e-05, 2.8860e-06]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.23_16.41.25/model_last.pt.tar.epoch_100",agg_weight=[1.0010e-06, 1.2852e-06, 3.1710e-06]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.25_20.26.59/model_last.pt.tar.epoch_100",agg_weight=[1.0027e-06, 1.2876e-06, 3.1768e-06, 2.6025e-06]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.23_20.31.44/model_last.pt.tar.epoch_60",agg_weight=[0.0000042880, 0.0000052076, 0.000013619, 0.000011356]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.23_20.31.44/model_last.pt.tar.epoch_60",agg_weight=[0.0000042880, 0.0000052076, 0.000013619, 0.000011356]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.24_22.30.37/model_last.pt.tar.epoch_40",agg_weight=[6.8286e-07, 8.5386e-07, 2.4469e-06, 1.7947e-06]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.23_16.41.25/model_last.pt.tar.epoch_100",agg_weight=[0.000001001, 0.0000012852, 0.000003171, 0.0000025974]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.17_22.14.24/model_last.pt.tar.epoch_100",agg_weight=[0.0000058971, 0.0000076193, 0.000021128, 0.000014881]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Nov.24_15.36.07/model_last.pt.tar.epoch_100",agg_weight=[9.6644e-06, 1.1123e-05, 2.6658e-05, 2.3946e-05, 2.2730e-06, 1.8931e-05,
        # Line(CertifiedRate("saved_models/model_loan_Nov.24_15.36.07/model_last.pt.tar.epoch_100",agg_weight=[1.0758e-05, 1.2443e-05, 2.9469e-05, 2.7885e-05, 2.5354e-06, 2.1048e-05,
        # Line(CertifiedRate("saved_models/model_loan_Nov.26_21.45.22/model_last.pt.tar.epoch_100",agg_weight=[1.2585e-05, 1.4720e-05, 3.5689e-05, 3.3686e-05, 2.9692e-06, 2.5363e-05,
        # 1.5197e-05, 3.3589e-06, 2.4033e-06]), " R = 4, MDCR"),
        # 1.2176e-05, 2.8453e-06]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Nov.22_23.03.49/model_last.pt.tar.epoch_100",agg_weight=[1.0045e-05, 1.1727e-05, 2.8316e-05, 2.7760e-05, 2.3748e-06, 2.0122e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Nov.22_15.35.02/model_last.pt.tar.epoch_100",agg_weight=[1.1437e-05, 1.4514e-05, 3.6322e-05, 2.9649e-05, 2.6262e-06]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Nov.20_20.58.13/model_last.pt.tar.epoch_100",agg_weight=[3.9778e-06, 4.9433e-06, 1.3450e-05, 1.1833e-05]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Nov.20_10.26.02/model_last.pt.tar.epoch_100",agg_weight=[1.2867e-06, 1.4446e-06, 1.0463e-06, 1.3084e-06]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.22_23.55.06/model_last.pt.tar.epoch_80",agg_weight=[0.000031092, 0.000041629, 0.00010529, 0.000085331]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.22_10.16.16/model_last.pt.tar.epoch_100",agg_weight=[0.000010846]), " R = 1, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.21_23.23.29/model_last.pt.tar.epoch_100",agg_weight=[0.0000092709, 0.000010628]), " R = 3, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.21_10.09.12/model_last.pt.tar.epoch_100",agg_weight=[0.0000039778, 0.0000049433, 0.00001345, 0.000011833]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.20_22.11.35/model_last.pt.tar.epoch_100",agg_weight=[0.00010528, 0.00012101, 0.0002673, 0.0002018]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.19_21.46.26/model_last.pt.tar.epoch_100",agg_weight=[0.0000092084, 0.000010444, 0.000017731, 0.00001641, 0.000002886]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Oct.19_08.48.38/model_last.pt.tar.epoch_100",agg_weight=[0.0000094896, 0.00010979, 0.000018691, 0.00001607]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Aug.22_10.48.13/model_last.pt.tar.epoch_100",agg_weight=[0.0001914, 0.0002356, 0.0003834, 0.9025]), " R = 4, MDCR"),
        # Line(CertifiedRate("saved_models/model_loan_Feb.04_19.09.27/model_last.pt.tar.epoch_100",agg_weight=[0.00013025, 0.00018528, 0.00023402, 0.00033203]), " R = 3, MDCR"),
    ])
    plot_certified_accuracy(
        "plots/loan/vary_gamma_tadv6_T100_cer_acc", "vary $\gamma$ ($t_{adv}=6$, T=100, R=4)", 1000, [
            # Line(CertifiedAcc("saved_models/model_loan_Oct.21_10.09.12/model_last.pt.tar.epoch_100",agg_weight=[0.0000039778, 0.0000049433, 0.00001345, 0.000011833]), " R = 4, MDCR"),
            Line(CertifiedAcc("saved_models/model_loan_Oct.25_20.26.59/model_last.pt.tar.epoch_100",agg_weight=[1.0027e-06, 1.2876e-06, 3.1768e-06, 2.6025e-06]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.13_20.44.43/model_last.pt.tar.epoch_100",agg_weight=[1.4281e-05, 1.8850e-05, 4.7604e-05, 3.7737e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.13_20.44.43/model_last.pt.tar.epoch_100",agg_weight=[1.4281e-05, 1.8850e-05, 4.7604e-05, 3.7737e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.12_21.22.53/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.12_13.33.16/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.11_21.33.44/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.02_16.19.10/model_last.pt.tar.epoch_100",agg_weight=[9.6407e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.02_09.37.49/model_last.pt.tar.epoch_100",agg_weight=[9.6407e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.11_15.36.23/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.10_16.53.31/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.09_12.34.00/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.08_22.31.14/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.07_09.24.45/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.06_17.45.26/model_last.pt.tar.epoch_100",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.05_19.42.47/model_last.pt.tar.epoch_100",agg_weight=[9.6407e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.04_21.42.23/model_last.pt.tar.epoch_100",agg_weight=[9.6407e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.01_22.07.01/model_last.pt.tar.epoch_100",agg_weight=[9.6407e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.01_15.59.47/model_last.pt.tar.epoch_100",agg_weight=[9.6407e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.01_09.48.18/model_last.pt.tar.epoch_100",agg_weight=[9.6408e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Dec.01_09.48.18/model_last.pt.tar.epoch_100",agg_weight=[9.6408e-06, 1.1889e-05, 2.9445e-05, 2.5570e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.25_20.26.59/model_last.pt.tar.epoch_100",agg_weight=[1.0155e-06, 1.1213e-06, 3.0843e-06, 2.5144e-06]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Nov.30_23.36.10/model_last.pt.tar.epoch_100",agg_weight=[1.0155e-06, 1.1213e-06, 3.0843e-06, 2.5144e-06]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Nov.30_16.59.21/model_last.pt.tar.epoch_100",agg_weight=[1.8008e-06, 1.9700e-06, 5.2551e-06, 4.4257e-06]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Nov.30_10.05.53/model_last.pt.tar.epoch_100",agg_weight=[3.6387e-06, 3.9805e-06, 1.0204e-05, 9.0297e-06]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Nov.29_19.59.12/model_last.pt.tar.epoch_100",agg_weight=[7.8739e-06, 8.8277e-06, 2.1621e-05, 2.0654e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Nov.28_21.07.45/model_last.pt.tar.epoch_100",agg_weight=[1.0272e-05, 1.2332e-05, 3.0118e-05, 2.7374e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Nov.27_22.29.36/model_last.pt.tar.epoch_140",agg_weight=[6.7846e-06, 6.9319e-06, 1.6027e-05, 1.6193e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Nov.27_10.51.28/model_last.pt.tar.epoch_120",agg_weight=[1.5002e-05, 1.7055e-05, 3.9458e-05, 3.7011e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.20_22.11.35/model_last.pt.tar.epoch_100",agg_weight=[1.0528e-04, 1.2101e-04, 2.2673e-04, 2.0182e-04]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Nov.23_16.27.28/model_last.pt.tar.epoch_100",agg_weight=[9.6644e-06, 1.1123e-05, 2.6658e-05, 2.3946e-05, 2.2730e-06, 1.8931e-05,
            # Line(CertifiedAcc("saved_models/model_loan_Nov.24_15.36.07/model_last.pt.tar.epoch_100",agg_weight=[1.0758e-05, 1.2443e-05, 2.9469e-05, 2.7885e-05, 2.5354e-06, 2.1048e-05,
        # 1.2176e-05, 2.8453e-06]), " R = 4, MDCR"),
        #     Line(CertifiedAcc("saved_models/model_loan_Nov.26_21.45.22/model_last.pt.tar.epoch_100",agg_weight=[1.2585e-05, 1.4720e-05, 3.5689e-05, 3.3686e-05, 2.9692e-06, 2.5363e-05,
        # 1.5197e-05, 3.3589e-06, 2.4033e-06]), " R = 4, MDCR"),
        # 1.1431e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Nov.22_23.03.49/model_last.pt.tar.epoch_100",agg_weight=[1.0045e-05, 1.1727e-05, 2.8316e-05, 2.7760e-05, 2.3748e-06, 2.0122e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Nov.22_15.35.02/model_last.pt.tar.epoch_100",agg_weight=[1.1437e-05, 1.4514e-05, 3.6322e-05, 2.9649e-05, 2.6262e-06]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Nov.20_20.58.13/model_last.pt.tar.epoch_100",agg_weight=[3.9778e-06, 4.9433e-06, 1.3450e-05, 1.1833e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Nov.20_10.26.02/model_last.pt.tar.epoch_100",agg_weight=[1.2867e-06, 1.4446e-06, 1.0463e-06, 1.3084e-06]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.22_23.55.06/model_last.pt.tar.epoch_80",agg_weight=[0.000031092, 0.000041629, 0.00010529, 0.000085331]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.19_21.46.26/model_last.pt.tar.epoch_100",agg_weight=[9.2084e-06, 1.0444e-05, 1.7731e-05, 1.6410e-05, 2.8860e-06]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.24_22.30.37/model_last.pt.tar.epoch_40",agg_weight=[6.8286e-07, 8.5386e-07, 2.4469e-06, 1.7947e-06]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.23_16.41.25/model_last.pt.tar.epoch_100",agg_weight=[1.0010e-06, 1.2852e-06, 3.1710e-06]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.25_20.26.59/model_last.pt.tar.epoch_100",agg_weight=[1.0027e-06, 1.2876e-06, 3.1768e-06, 2.6025e-06]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.19_08.48.38/model_last.pt.tar.epoch_100",agg_weight=[0.0000094896, 0.00010979, 0.000018691, 0.00001607]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.19_08.48.38/model_last.pt.tar.epoch_100",agg_weight=[0.0000094896, 0.00010979, 0.000018691, 0.00001607]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.21_23.23.29/model_last.pt.tar.epoch_100",agg_weight=[0.0000092709, 0.000010628]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.22_10.16.16/model_last.pt.tar.epoch_100",agg_weight=[0.000010846]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.19_21.46.26/model_last.pt.tar.epoch_100",agg_weight=[0.0000092084, 0.000010444, 0.000017731, 0.00001641, 0.000002886]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.26_10.26.45/model_last.pt.tar.epoch_40",agg_weight=[1.0743e-05, 1.2486e-05, 3.0092e-05, 2.8951e-05]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.23_20.31.44/model_last.pt.tar.epoch_60",agg_weight=[0.0000042880, 0.0000052076, 0.000013619, 0.000011356]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.23_16.41.25/model_last.pt.tar.epoch_100",agg_weight=[0.000001001, 0.0000012852, 0.000003171, 0.0000025974]), " R = 4, MDCR"),
            # Line(CertifiedAcc("saved_models/model_loan_Oct.17_22.14.24/model_last.pt.tar.epoch_100",agg_weight=[0.0000058971, 0.0000076193, 0.000021128, 0.000014881]), " R = 4, MDCR"),
    ])
    # # gammma
    # plot_certified_accuracy(
    #     "plots/loan/vary_gamma_tadv6_T100_cer_acc", "vary $\gamma$ ($t_{adv}=6$, T=100, R=1)", 2.5, [
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), "$\gamma$ = 10"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.43.48/model_last.pt.tar.epoch_100"), "$\gamma$ = 20"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.43.59/model_last.pt.tar.epoch_100"), "$\gamma$ = 30"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.44.06/model_last.pt.tar.epoch_100"), "$\gamma$ = 50"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.44.29/model_last.pt.tar.epoch_100"), "$\gamma$ = 100"),
    #     ])
    
    
    # plot_certified_rate(
    #     "plots/loan/vary_gamma_tadv6_T100_cer_rate", "vary $\gamma$ ($t_{adv}=6$, T=100, R=1)", 2.5, [
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), "$\gamma$ = 10"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.43.48/model_last.pt.tar.epoch_100"), "$\gamma$ = 20"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.43.59/model_last.pt.tar.epoch_100"), "$\gamma$ = 30"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.44.06/model_last.pt.tar.epoch_100"), "$\gamma$ = 50"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.44.29/model_last.pt.tar.epoch_100"), "$\gamma$ = 100"),
    #     ])
 
    ##### t_adv 
    # plot_certified_rate(
    #     "plots/loan/vary_tadv_T100_cer_rate", "vary $t_{adv}$ ($\gamma=10$, T=100, R=1)", 2.5, [
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 6"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.39.18/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 20"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.39.26/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 40"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.39.34/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 60"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.39.42/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 80"),
    #     ])

    # plot_certified_accuracy(
    #      "plots/loan/vary_tadv_T100_cer_acc", "vary $t_{adv}$ ($\gamma=10$, T=100, R=1)", 2.5, [
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 6"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.39.18/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 20"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.39.26/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 40"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.39.34/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 60"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.39.42/model_last.pt.tar.epoch_100"), " $t_{adv}$ = 80"),
    #     ])

    # plot_certified_accuracy(
    #     "plots/loan/vary_tadv_T50_cer_acc", "vary $t_{adv}$ ($\gamma=100$, T=50, R=2)", 0.15, [
    #         Line(CertifiedAcc("saved_models/model_mnist_Feb.01_20.36.31/model_last.pt.tar.epoch_50"), " $t_{adv}$ = 20"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Feb.01_20.37.23/model_last.pt.tar.epoch_50"), " $t_{adv}$ = 40"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Feb.01_20.37.38/model_last.pt.tar.epoch_50"), " $t_{adv}$ = 43"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Feb.01_20.38.48/model_last.pt.tar.epoch_50"), " $t_{adv}$ = 45"),
    #     ])


    

    ##### poison_ratio 
    # plot_certified_rate(
    #     "plots/loan/vary_qn_T100_cer_rate", "vary $q_{B_i}/n_{B_i}$ ($\gamma=10$, $t_{adv}=6$, T=100, R=1)", 5.0, [
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.45.58/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 2.5%"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 5%"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.46.15/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 10%"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.46.22/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 15%"),
    #         Line(CertifiedRate("saved_models/model_loan_Feb.04_04.46.29/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 20%"),
    #     ])

    # plot_certified_accuracy(
    #      "plots/loan/vary_qn_T100_cer_acc", "vary $q_{B_i}/n_{B_i}$ ($\gamma=10$, $t_{adv}=6$, T=100, R=1)", 5.0, [
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.45.58/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 2.5%"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.37.13/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 5%"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.46.15/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 10%"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.46.22/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 15%"),
    #         Line(CertifiedAcc("saved_models/model_loan_Feb.04_04.46.29/model_last.pt.tar.epoch_100"), " $q_{B_i}/n_{B_i}$ = 20%"),
    #     ])



    # ### vary M
    # plot_certified_accuracy(
    #     "plots/loan/vary_M_T45_cer_acc", "vary M ($t_{adv}=40$, T=45, $\gamma=20$, R=1)", 1.25 , [
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=100), " M = 100"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=500), " M = 500"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=1000), " M = 1000"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=2000), " M = 2000"),
    #     ]
    #     )

    # plot_certified_rate(
    #     "plots/loan/vary_M_T45_cer_rate", "vary M ($t_{adv}=40$, T=45, $\gamma=20$, R=1)", 1.25 , [
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=100), " M = 100"),
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=500), " M = 500"),
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=1000), " M = 1000"),
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",M=2000), " M = 2000"),
    #     ]
    # )



    # ### vary alpha
    # plot_certified_accuracy(
    #     "plots/loan/vary_alpha_T45_cer_acc", "vary $alpha$ ($t_{adv}=40$, T=45, $\gamma=20$, R=1)", 1.25 , [
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.01), " 99% confidence"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.001), " 99.9% confidence"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.0001), " 99.99% confidence"),
    #         Line(CertifiedAcc("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.00001), " 99.999% confidence"),
    #     ]
    #     )
    # plot_certified_rate(
    #     "plots/loan/vary_alpha_T45_cer_rate", "vary $alpha$ ($t_{adv}=40$, T=45, $\gamma=20$, R=1)", 1.25 , [
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.01), " 99% confidence"),
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.001), " 99.9% confidence"),
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.0001), " 99.99% confidence"),
    #         Line(CertifiedRate("saved_models/model_mnist_Jan.31_22.02.08/model_last.pt.tar.epoch_45",alpha=0.00001), " 99.999% confidence"),
    #     ]
    #     )
