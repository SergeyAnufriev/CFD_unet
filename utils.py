import torch.nn as nn
from sklearn.model_selection import train_test_split
import os
import glob

'''dir_files = r'/content/drive/MyDrive/cfd_data/uzipped_files/graphical_data/'''

def split_data(dir_files:list)->(list,list):

    files = sorted(filter(os.path.isfile,glob.glob(dir_files + '*txt')))
    train_idx, test_idx = train_test_split(list(range(int(len(files)/3))),test_size=0.2)
    train_samples = []
    test_samples  = []
    for x in train_idx:
        train_samples += [x*3,x*3+1,x*3+2]
    for x in test_idx:
        test_samples  +=  [x*3,x*3+1,x*3+2]
    train_files = [files[x] for x in train_samples]
    test_files  = [files[x] for x in test_samples]
    return train_files,test_files


def weight_init(type_):
    '''Initialise linear layers weights'''
    '''Returns weight function for model.apply(weight function)'''
    def weight(m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if type_ == 'normal':
                m.weight.data.normal_(0.0, 0.007)
            elif type_ == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif type_ == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif type_ == 'kaiming_normal':
                nn.init.kaiming_normal(m.weight)
            elif type_ == 'ones':
                nn.init.trunc_normal_(m.weight)
            else:
                m.weight.data.fill_(0)
            if m.bias is not None:
                m.bias.data.fill_(0)
    return weight