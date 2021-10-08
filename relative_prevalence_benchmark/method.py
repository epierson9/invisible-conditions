import torch
from torch import nn
import numpy as np
import os
import pdb
from tqdm import tqdm

def get_loss(s_true, s_pred):
    criterion = torch.nn.BCELoss()
    loss = criterion(s_pred, torch.Tensor(s_true).cuda())  
    return loss.item()

def train_relative_estimator(x_train, s_train, expmt_config, n_epochs=7000, save=False):
    # train model
    if save:
        save_dir = './model_ckpts/c1_' + str(expmt_config['labeling_frequency_g1']) + '_c2_' 
        save_dir += str(expmt_config['labeling_frequency_g2']) + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    model = RelativeEstimator(expmt_config['n_groups'], expmt_config['n_attributes'])
    lamda = expmt_config['lamda']
    #optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters())

    model.cuda('cuda:0')
    model.train()
    
    losses = []
    x_tensor = torch.Tensor(x_train).cuda()
    s_tensor = torch.Tensor(s_train).cuda()
    all_idxs = np.array(list(range(len(x_tensor))))
#     n_examples = len(all_idxs)
#     batch_size = 13000
#     all_idxs = np.concatenate([all_idxs,
#                               [0 for i in range(batch_size - (n_examples % batch_size))]])
#     batch_idxs_list = np.array(all_idxs).reshape((-1, batch_size))
    n_batches = 1
    batch_idxs_list = np.array(all_idxs).reshape((n_batches, -1))
                                     
                    
#     n_total = len(s_train)
#     n_pos = np.sum(s_train)
#     n_neg = n_total - n_pos
#     weight = torch.tensor([n_total/n_neg, n_total/n_pos])
#     weight_ = weight[s_tensor.data.view(-1).long()].view_as(s_tensor).cuda()
    #criterion = torch.nn.BCEWithLogitsLoss(weight=weight_)
    criterion = torch.nn.BCELoss()

    criterion.cuda('cuda:0')

    for epoch in tqdm(range(n_epochs)):
        for i, batch_idxs in enumerate(batch_idxs_list): 
#             batch_idxs = np.trim_zeros(batch_idxs)
            s_pred = model(x_tensor[batch_idxs])
            loss = criterion(torch.squeeze(s_pred), s_tensor[batch_idxs])      
            
            l1_regularization = 0.
            for param in model.parameters():
                l1_regularization += param.abs().sum()
            loss += lamda * l1_regularization
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 4000 == 0 and epoch !=0 and save:
            torch.save(model.state_dict(), save_dir + 'model_' + str(epoch))
    if save:
        torch.save(model.state_dict(), save_dir + 'model_final_' + str(n_epochs))
    return model, losses

class RelativeEstimator(nn.Module):
    def __init__(self, n_groups, n_attributes):
        super().__init__()
        self.n_groups = n_groups
        self.n_attributes =  n_attributes
        self.real_c_g_coeffs = []
#         self.real_c_g_coeffs = torch.Tensor([[1], [1]]).cuda()
    
        self.p_y_coeffs = nn.Linear(n_attributes, 1, bias=True)
        self.c_g_coeffs = nn.Linear(n_groups, 1, bias=True)
        torch.nn.init.xavier_uniform(self.p_y_coeffs.weight)
        torch.nn.init.xavier_uniform(self.c_g_coeffs.weight)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x : a N x (n_groups + n_features) matrix, where the first 
        # n_groups features is a one-hot encoding of the patient's group
        # and the subsequent n_features features are the patient's attributes
        
        group_features = x[:,:self.n_groups]
        class_attributes = x[:,self.n_groups:]
        if not len(self.real_c_g_coeffs):
            c_pred = self.sigmoid(self.c_g_coeffs(group_features))
        else:
            c_pred = torch.mm(group_features, self.real_c_g_coeffs)
       
        p_y_pred = self.p_y_coeffs(class_attributes)
        p_y_pred = self.sigmoid(p_y_pred)
        
        return c_pred * p_y_pred
    
    def estimate_p_y(self, x):
        group_features = x[:,:self.n_groups]
        class_attributes = x[:,self.n_groups:]
        p_y_pred = self.p_y_coeffs(class_attributes)
        p_y_pred = self.sigmoid(p_y_pred)
        return p_y_pred
    
    def estimate_ratio(self, g1_attributes, g2_attributes):
        # find indices for g2 
        n_g1 = len(g1_attributes)
        n_g2 = len(g2_attributes)

        p_y_g1 = self.sigmoid(self.p_y_coeffs(g1_attributes))
        p_y_g2 = self.sigmoid(self.p_y_coeffs(g2_attributes))
        
        # sum and divide by 
        g1_prevalence = torch.sum(p_y_g1)/n_g1
        g2_prevalence = torch.sum(p_y_g2)/n_g2
        relative_prevalence = g1_prevalence / g2_prevalence
        return relative_prevalence.item(), g1_prevalence.item(), g2_prevalence.item()
    
    def get_c_g(self):
        if not len(self.real_c_g_coeffs):
            c_pred = self.sigmoid(self.c_g_coeffs.weight)
            return c_pred
        return self.real_c_g_coeffs