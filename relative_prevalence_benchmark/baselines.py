from rpy2.robjects import pandas2ri
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from rpy2.rinterface import SexpVector
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
import pdb
import numpy as np

from eval_fs import eval_relative_prior
from sarpu.pu_learning import *
from km.Kernel_MPE_grad_threshold import wrapper as ramaswamy



utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)
packnames = ('osqp')
utils.install_packages(StrVector(packnames))
osqp = importr('osqp', lib_loc='/mnt/hdd/home/divyas/R/x86_64-pc-linux-gnu-library/4.0')

rpy2.robjects.numpy2ri.activate()
r = robjects.r
r['source']('../PU_class_prior/logistic_CDMM.R')
r['source']('../PU_class_prior/optimize_b_MM.R')
cdmm_function = robjects.globalenv['logistic_fit_cdmm']

def cdmm_get_prior(x_train, s_train, g_config):
    n_groups = g_config['n_groups']
    nr, nc = x_train[:,n_groups:].shape
    x_r = robjects.r.matrix(x_train[:,2], nrow=nr, ncol=nc)
    s = robjects.r.matrix(s_train, nrow=s_train.shape[0], ncol=1)
    
    robjects.r.assign("x", x_r)
    robjects.r.assign("s", s)

    result_r = cdmm_function(robjects.globalenv["x"], robjects.globalenv["s"])
    result = robjects.conversion.rpy2py(result_r)
    pred_prior = np.mean(s_train)/result[0][0]
    return pred_prior

    
def cdmm(x1_train, x2_train, s1_train, s2_train, g_config):
    pred_g1_prior = cdmm_get_prior(x1_train, s1_train, g_config)
    pred_g2_prior = cdmm_get_prior(x2_train, s2_train, g_config)
    pred_rel_prior = pred_g1_prior/pred_g2_prior
    return pred_rel_prior, pred_g1_prior, pred_g2_prior, (None, None)

def supervised_rel_prior(x1_train, x2_train, y1_train, y2_train, x_test, 
                         g_config, classification_model=None):
    n_groups = g_config['n_groups']
    n_attrs = g_config['n_attributes']
    classification_attributes = [i + n_groups for i in range(n_attrs)]
    

    f1_model, info = pu_learn_neg(x1_train, y1_train, 
                                 classification_attributes=classification_attributes,
                                 classification_model=classification_model)
    f2_model, info = pu_learn_neg(x2_train, y2_train, 
                                 classification_attributes=classification_attributes,
                                 classification_model=classification_model)
    pred_rel_prior, pred_g1_prior, pred_g2_prior = eval_relative_prior(x_test, f1_model, f2_model)
    return pred_rel_prior, pred_g1_prior, pred_g2_prior, (f1_model, f2_model)

def sar_em_rel_prior(x1_train, x2_train, s1_train, s2_train, x_test, 
                     g_config, classification_model=None):
    n_groups = g_config['n_groups']
    n_attrs = g_config['n_attributes']
    classification_attributes = [i + n_groups for i in range(n_attrs)]
    
    f1_model, e1_model, info = pu_learn_sar_em(x1_train, s1_train,                    
                                               classification_attributes=classification_attributes,
                                               propensity_attributes=[],
                                               classification_model=classification_model)
    f2_model, e2_model, info = pu_learn_sar_em(x2_train, s2_train, 
                                               classification_attributes=classification_attributes,
                                               propensity_attributes=[],
                                               classification_model=classification_model)
    
    pred_rel_prior, pred_g1_prior, pred_g2_prior = eval_relative_prior(x_test, 
                                                                   f1_model, 
                                                                   f2_model)
    return pred_rel_prior, pred_g1_prior, pred_g2_prior, (f1_model, f2_model)

def scar_km2_rel_prior(x1_train, x2_train, s1_train, s2_train):
    if len(s1_train) > 3200:
        sample = np.random.choice(x1_train.shape[0], 3200)
        x1_train_ram = x1_train[sample]
        s1_train_ram = s1_train[sample]
    else:
        x1_train_ram = x1_train
        s1_train_ram = s1_train
        
    (_, c1_est) = ramaswamy(x1_train_ram[:,2:3], 
                            x1_train_ram[:,2:3][np.squeeze(s1_train_ram == 1)])

    if len(s2_train) > 3200:
        sample = np.random.choice(x2_train.shape[0], 3200)
        x2_train_ram = x2_train[sample]
        s2_train_ram = s2_train[sample]
    else:
        x2_train_ram = x2_train
        s2_train_ram = s2_train
    (_, c2_est) = ramaswamy(x2_train_ram[:,2:3], 
                            x2_train_ram[:,2:3][np.squeeze(s2_train_ram == 1)])
    # Output of ramaswamy is the p(positive component) or p(y=1) in mixture
    pred_g1_prior = c1_est
    pred_g2_prior = c2_est
    pred_rel_prior = c1_est/c2_est
    return pred_rel_prior, pred_g1_prior, pred_g2_prior,  (None, None)