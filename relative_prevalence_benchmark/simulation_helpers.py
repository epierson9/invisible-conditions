import numpy as np
import pdb
from scipy.stats import percentileofscore

def generate_g(g_config):
    # Return set  of examples x, where first n_groups features
    # are a one-hot-encoding of the group
    # and the remaining feature are randomly sampled from a multivariate Gaussian
    std = g_config['std'] * np.eye(g_config['n_attributes'])
    x = np.random.multivariate_normal(g_config['mean'], std, g_config['n_samples'])
    x = np.concatenate((np.zeros((g_config['n_samples'], g_config['n_groups'])), x), axis=1)
    x[:,g_config['group_feat']-1] = 1
    return x

def separable_decision_rule_1d(x):
    thresh = 0
    y = (x < thresh).astype(int)
    return y

def inseparable_decision_rule_1d(x):
    b = -2
    ys = []
    for x_i in x:
        p_y = 1/(1 + np.exp(-x_i*b))
        y_i = int(np.random.random() < p_y)
        ys.append(y_i)
    return np.expand_dims(np.array(ys), axis=1)


def inseparable_decision_rule_nd(x, beta, scale, decision_rule_diff):
    n_dims = x.shape[1]
    
    ortho_normal = np.expand_dims(1*np.ones(n_dims), 0)
    features = np.dot(ortho_normal, x.T)
    normalization = np.dot(ortho_normal, ortho_normal.T)[0][0]
    features = features/normalization
    p_y = scale * (1/(1 + np.exp(-features*beta))) + decision_rule_diff
    samples = np.random.random(p_y.shape)
    y = np.less(samples, p_y)[0].astype(int)
    return y

def separable_decision_rule_nd(x, beta, scale, decision_rule_diff=0):
    n_dims = x.shape[1] 
    
    ortho_normal = np.expand_dims(1*np.ones(n_dims), 0)
    features = np.dot(ortho_normal, x.T)
    normalization = np.dot(ortho_normal, ortho_normal.T)[0][0]
    features = features/normalization
    y = (features > 0).astype(int)
    return y

def create_gap_nd(x, y, g_config):
    # drop examples close to the boundary
    n_dims = x.shape[1] - g_config['n_groups']
    ortho_normal = np.expand_dims(1*np.ones(n_dims), 0)
    features = np.dot(ortho_normal, x[:,g_config['n_groups']:].T)
    normalization = np.dot(ortho_normal, ortho_normal.T)[0][0]
    features = features/normalization
    
    # Drop 40th - 60th percentile values
    anchor = percentileofscore(features[0], 0)
    lower_pctle, upper_pctle = max(anchor-40, 10), min(anchor+40, 90)
    lower_thresh = np.percentile(features, lower_pctle)
    upper_thresh = np.percentile(features, upper_pctle)
    drop_idxs = np.where(np.logical_and(features[0]>=lower_thresh, features[0]<=upper_thresh))
    x = np.delete(x, drop_idxs[0], axis=0)
    y = np.delete(y, drop_idxs[0])
    return x, y

def generate_y(x, g_config, decision_rule, beta=1, scale=1, decision_rule_diff=0):
    return decision_rule(x[:,g_config['n_groups']:], beta, scale, decision_rule_diff)

def generate_s_scar(y, labeling_freq):
    y_pos_idxs = np.where(y == 1)[0]
    labeled_idxs = np.random.choice(y_pos_idxs, int(len(y_pos_idxs)*labeling_freq), replace=False)
    s = np.zeros(y.shape)
    s[labeled_idxs] = 1
    return s

def create_gap_1d(x, y, g_config):
    thresh = 0
    epsilon = 1
    # Breaks when there's more than one attribute
    greater_than = x[:,g_config['n_groups']] > (thresh - epsilon)
    less_than = x[:,g_config['n_groups']] < (thresh + epsilon)
    drop_idxs = np.array([a and b for a,b in zip(greater_than, less_than)])
    x = x[~drop_idxs]
    y = y[~drop_idxs]
    return x, y