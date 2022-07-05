import torch
from torch import nn
import numpy as np
import os
import pdb
from tqdm import tqdm
import hashlib


from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import average_precision_score, roc_auc_score
from paths import MODEL_DIR

def load_model(model_fname, expmt_config):
    f_model =  RelativeEstimator(expmt_config['n_groups'], expmt_config['n_attributes'])
    f_model.load_state_dict(torch.load(MODEL_DIR + model_fname))
    return f_model

def BCELoss_class_weighted(weights):

    def loss(input, target):
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)
    
    return loss

def get_loss(s_true, s_pred):
    # s_true is on CPU
    # s_pred is on cuda
    criterion = torch.nn.BCELoss()
    loss = criterion(s_pred, torch.Tensor(s_true).cuda())  
    return loss.item()

def get_model_fname(expmt_config):
    m_name = ""
    for key in sorted(expmt_config.keys()):
        m_name += key + '_' + str(expmt_config[key]) + "_"
    m_name = hashlib.md5(m_name.encode('utf-8')).hexdigest()
    return m_name

def apply_group_weights(loss, model, x):
    group_indicators = x[:,:model.n_groups]
    sample_weights = torch.mm(group_indicators, torch.unsqueeze(model.group_weights, 1).cuda())
    loss = torch.dot(loss, torch.squeeze(sample_weights)) / len(x)
    return loss

def train_relative_estimator(x_train, s_train, x_val, s_val, 
                             expmt_config, save_model=False,
                             save_preds=False):

    save_dir = MODEL_DIR + get_model_fname(expmt_config)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
            
    # Choose underlying model
    if expmt_config['estimator_type'] == 'logreg':
        model = RelativeEstimator(expmt_config['n_groups'], expmt_config['n_attributes'],
                                  group_weights=expmt_config['group_weights'])
    elif expmt_config['estimator_type'] == 'deep':
        model = DeepRelativeEstimator(expmt_config['n_groups'], expmt_config['n_attributes'])
    elif expmt_config['estimator_type'] == 'unconstrained':
        model = UnconstrainedEstimator(expmt_config['n_groups'], expmt_config['n_attributes'])

    # Choose optimizer
    if expmt_config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, weight_decay=.01)
    elif expmt_config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    
    lamda = expmt_config['lamda']
    n_epochs = expmt_config['n_epochs']

    model.cuda('cuda:0')
    model.train()

    losses = []
    val_losses = []
    val_aucs = []
    logged_epochs = []
    log_interval = 500
    
    x_tensor = torch.Tensor(x_train).cuda()
    s_tensor = torch.Tensor(s_train).cuda()
    x_val_tensor = torch.Tensor(x_val).cuda()
    s_val_tensor = torch.Tensor(s_val).cuda()
    
    all_idxs = np.array(list(range(len(x_tensor))))
    
    n_batches = expmt_config['n_batches']
    n_to_pad = n_batches - (len(all_idxs) % n_batches)
    all_idxs = np.pad(all_idxs, (0, n_to_pad))
    batch_idxs_list = np.array(all_idxs).reshape((n_batches, -1))

    criterion = torch.nn.BCELoss(reduction='none')
    criterion.cuda('cuda:0')
    val_idxs = np.random.choice(list(range(len(x_val_tensor))), len(x_val_tensor))

    for epoch in tqdm(range(n_epochs)):
        for i, batch_idxs in enumerate(batch_idxs_list): 
            s_pred = model(x_tensor[batch_idxs])
            loss = criterion(torch.squeeze(s_pred), s_tensor[batch_idxs])  
            loss = apply_group_weights(loss, model, x_tensor[batch_idxs])

            l1_regularization = 0.
            params = model.get_regularizable_parameters()
            l1_regularization += params.abs().sum()
            loss += lamda * l1_regularization
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()                

        if epoch % log_interval == 0 and epoch !=0:
            with torch.no_grad():
                model.eval()
                s_val_pred = model(x_val_tensor[val_idxs])
                val_loss = criterion(torch.squeeze(s_val_pred), s_val_tensor[val_idxs]) 
                val_loss = apply_group_weights(val_loss, model, x_val_tensor[val_idxs])
                val_auc = roc_auc_score(s_val_tensor.cpu().numpy()[val_idxs], s_val_pred.detach().cpu().numpy().flatten())
                val_losses.append(val_loss.item())
                val_aucs.append(val_auc)
                logged_epochs.append(epoch)

                epoch_dir = save_dir + '/' + str(epoch) + '/'
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)

                if save_model:
                    torch.save(model.state_dict(), epoch_dir + 'model')

                if save_preds:
                    val_preds = model(x_val_tensor)
                    np.save(epoch_dir + 'val_preds', val_preds.detach().cpu().numpy())
                    np.save(epoch_dir + 's_val', s_val)

                    train_preds = model(x_tensor[batch_idxs_list[0]])
                    np.save(epoch_dir + 'train_preds', train_preds.detach().cpu().numpy())
                    np.save(epoch_dir + 's_train', s_tensor[batch_idxs_list[0]].cpu().numpy())

                    train_loss = get_loss(s_tensor[batch_idxs_list[0]].cpu(), train_preds.flatten())
                    val_loss = get_loss(s_val_tensor.cpu(), val_preds.flatten())
                    
                    train_loss = apply_group_weights(train_loss, model, x_tensor[batch_idxs_list[0]])
                    val_loss = apply_group_weights(val_loss, model, x_val_tensor)
                                                                                
                    np.save(epoch_dir + 'train_loss', train_loss)
                    np.save(epoch_dir + 'val_loss', val_loss)
        
    best_epoch = logged_epochs[np.argmax(val_aucs)]
    model_path = save_dir + '/' + str(best_epoch) + '/' + 'model'
    model.load_state_dict(torch.load(model_path))
    
    
    if save_model:
        epoch_ = logged_epochs[np.argmin(val_losses)]
        epoch_dir = save_dir + '/' + str(epoch) + '/'
        torch.save(model.state_dict(), save_dir + 'model_final_' + str(n_epochs))
        print(save_dir + 'model_final_' + str(n_epochs))
    s_preds = model(x_val_tensor[val_idxs])
    val_loss = get_loss(s_val[val_idxs], s_preds[:,0])
    s_preds = torch.squeeze(s_preds).detach().cpu()
    auc = roc_auc_score(s_val[val_idxs], s_preds)
    auprc = average_precision_score(s_val[val_idxs], s_preds)
    info = {'auprc': auprc,
            'auc': auc, 
            'val_loss': val_loss, 
            'best_epoch': best_epoch, 
            'val_losses': tuple(val_losses),
            'val_aucs': tuple(val_aucs),
            'logged_epochs': tuple(logged_epochs)}
    
    return model, losses, info

class RelativeEstimator(nn.Module):
    def __init__(self, n_groups, n_attributes, group_weights=[]):
        super().__init__()
        self.n_groups = n_groups
        self.n_attributes =  n_attributes
        self.real_c_g_coeffs = torch.Tensor([[1], [1]]).cuda()
        self.real_c_g_coeffs = []
                
        self.p_y_coeffs = nn.Linear(n_attributes, 1, bias=True)
        self.p_y_coeffs.weight.data.fill_(0)

        self.c_g_coeffs = nn.Linear(n_groups, 1, bias=False)
        
        torch.nn.init.kaiming_normal_(self.c_g_coeffs.weight)
        
        self.group_weights = torch.ones(self.n_groups)
        if len(group_weights):
            self.group_weights =  torch.Tensor(group_weights)
        
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # x : a N x (n_groups + n_features) matrix, where the first 
        # n_groups features is a one-hot encoding of the patient's group
        # and the subsequent n_features features are the patient's attributes
        
        group_indicators = x[:,:self.n_groups]
        features = x[:,self.n_groups:]
        if not len(self.real_c_g_coeffs):
            c_pred = self.sigmoid(self.c_g_coeffs(group_indicators))
        else:
            c_pred = torch.mm(group_indicators, self.real_c_g_coeffs)
       
        p_y_pred = self.estimate_p_y(features)
        return c_pred * p_y_pred
    
    
    def estimate_p_y(self, features):
        
        p_y_pred = self.p_y_coeffs(features)
        p_y_pred = self.sigmoid(p_y_pred)
        return p_y_pred
    
    def estimate_ratio(self, g1_attributes, g2_attributes):
        p_y_g1 = self.estimate_p_y(g1_attributes)
        p_y_g2 = self.estimate_p_y(g2_attributes)
        
        g1_prevalence = torch.mean(p_y_g1)
        g2_prevalence = torch.mean(p_y_g2)
        
        relative_prevalence = g1_prevalence / g2_prevalence
        
        return relative_prevalence.item(), g1_prevalence.item(), g2_prevalence.item()
    
    def get_regularizable_parameters(self):
        params = list(self.p_y_coeffs.parameters())[0]
        return params
    
    def get_c_g(self):
        if not len(self.real_c_g_coeffs):
            c_pred = self.sigmoid(self.c_g_coeffs.weight)
            return c_pred
        return self.real_c_g_coeffs
    
class UnconstrainedEstimator(nn.Module):
    def __init__(self, n_groups, n_attributes):
        super().__init__()
        self.n_groups = n_groups
        self.coeffs = nn.Linear(n_attributes*(n_groups+1), 1, bias=True)
        # Can improve convergence to initialize model to all 0s:
        # self.coeffs.weight.data.fill_(0)
        self.group_weights = torch.ones(self.n_groups)
        self.sigmoid = nn.Sigmoid()
        torch.nn.init.kaiming_normal_(self.coeffs.weight)
        
    def forward(self, x):
        # x is n_attributes + n_groups features 
        # for group_indicator in the first n_group indicator columns
        # append a version of x 
        transformed_x = []
        transformed_x.append(x[:,self.n_groups:])
        for i in range(self.n_groups):
            transformed_x.append(torch.unsqueeze(x[:,i], 1)*x[:,self.n_groups:])

        transformed_x = torch.cat(transformed_x, axis=1)
        p_s_pred = self.coeffs(transformed_x)
        p_s_pred = self.sigmoid(p_s_pred)
        return p_s_pred
    
    def get_regularizable_parameters(self):
        params = list(self.coeffs.parameters())[0]
        return params
            
    
class UnconstrainedPerGroupEstimator(nn.Module):
    def __init__(self, n_groups, n_attributes, group_weights=[]):
        super().__init__()
        self.n_groups = n_groups
        self.coeffs = torch.nn.Parameter(torch.zeros(n_groups, n_attributes))
        self.group_weights = torch.ones(self.n_groups)
        if len(group_weights):
            self.group_weights =  torch.Tensor(group_weights)
        
    def forward(self, x):
        group_feats = x[:,:self.n_groups]
        # Get weights for each example
        coeffs = torch.mm(group_feats, self.coeffs)
        # Multiple each example with each example's weights
        feats = x[:,self.n_groups:]
        bs = feats.shape[0]
        n_feats = feats.shape[1]
        p_s_pred = torch.bmm(feats.reshape(bs, 1, n_feats), coeffs.reshape(bs, n_feats, 1))        
        p_s_pred = torch.squeeze(p_s_pred)
        # Preserve only the p_s_pred where example is multiplied by it's own weight
        return p_s_pred

    def get_regularizable_parameters(self):
        return self.coeffs.data.flatten()

    
class DeepRelativeEstimator(nn.Module):
    def __init__(self, n_groups, n_attributes):
        super().__init__()
        self.n_groups = n_groups
        self.n_attributes =  n_attributes
        self.real_c_g_coeffs = torch.Tensor([[1], [1]]).cuda()
        self.real_c_g_coeffs = []
                
        self.c_g_coeffs = nn.Linear(n_groups, 1, bias=False)
        self.fc1 = nn.Linear(n_attributes, 512, bias=True)
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.fc3 = nn.Linear(256, 64, bias=True)
        self.fc4 = nn.Linear(64, 1, bias=True)
    
        torch.nn.init.kaiming_normal_(self.c_g_coeffs.weight)
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            torch.nn.init.kaiming_normal_(layer.weight)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # x : a N x (n_groups + n_features) matrix, where the first 
        # n_groups features is a one-hot encoding of the patient's group
        # and the subsequent n_features features are the patient's attributes
        
        group_indicators = x[:,:self.n_groups]
        features = x[:,self.n_groups:]
        if not len(self.real_c_g_coeffs):
            c_pred = self.sigmoid(self.c_g_coeffs(group_indicators))
        else:
            c_pred = torch.mm(group_indicators, self.real_c_g_coeffs)
       
        p_y_pred = self.estimate_p_y(features)
        
        return c_pred * p_y_pred
    
    def estimate_p_y(self, features):
        x = self.fc1(features)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        p_y_pred = self.sigmoid(x)
        return p_y_pred
    
    def estimate_ratio(self, g1_attributes, g2_attributes):
        p_y_g1 = self.estimate_p_y(g1_attributes)
        p_y_g2 = self.estimate_p_y(g2_attributes)
        
        g1_prevalence = torch.mean(p_y_g1)
        g2_prevalence = torch.mean(p_y_g2)
        
        relative_prevalence = g1_prevalence / g2_prevalence
        
        return relative_prevalence.item(), g1_prevalence.item(), g2_prevalence.item()
    
    def get_regularizable_parameters(self):
        params = []
        for layer in [self.fc1, self.fc2, self.fc3, self.fc3]:
            params.extend(list(layer.parameters())[0].flatten())
        return torch.stack(params)
    
    def get_c_g(self):
        if not len(self.real_c_g_coeffs):
            c_pred = self.sigmoid(self.c_g_coeffs.weight)
            return c_pred
        return self.real_c_g_coeffs
