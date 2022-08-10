"""The module purpose is to load simulation data in _n.txt,_c.txt files into
pytorch geometric dataset object https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html,
which will be used to create data batches, where single entity withing batch includes one CFD simulation,which
is described by input graph and target graph node feature matrix """


import torch
from torch_geometric.data import Dataset
import glob
from torch.nn.functional import one_hot
from torch_geometric.data import Data
import pandas as pd



'''1e662_193_51_17291_744_149_n.txt

U_X = 1.93  (number after first _ /100)
U_Y = 0.51  ( number after second _ /100)
P_ambient = 17291 Pa ( 3d _)
alpha_deg = 7.44 (4th _ /100)
cav_ num  = 1.49 (last number /100)

'''
dir_files = r'C:\Users\zcemg08\PycharmProjects\gitlearning\data_samples\data_samples\graphical_data\*txt'

files = glob.glob(dir_files)

''' input 
   0 - internal - velocities yes
   1 - excit 
   2 - airfoil surface 
   3 - inlet'''


''' output 
   0 - internal 
   1 - excit 
   2 - airfoil surface 
   3 - inlet
   '''


def read_file(dir:str)->list:
    with open(dir, 'r') as fd:
        lines = []
        for line in fd:
            lines.append(line)
    return lines


def connectivity_data(dir_c:str, dict_:dict)->torch.long:
    '''Input: connectivity file location and Node translation dictionary
    Output: adj matrix in CCO format - torch long tensor'''

    lines_c = read_file(dir_c)
    up_   = []
    down_ = []

    for l in lines_c:
        line = l.split('  ')
        '''renumber all nodes'''
        i = dict_[int(line[0])]
        j = dict_[int(line[1])]
        k = dict_[int(line[2])]

        up_   += [i, j, i, k, j, k]
        down_ += [j, i, k, i, k, j]

    CCO_matrix = [up_, down_]

    return torch.unique(torch.tensor(CCO_matrix, dtype=torch.long),dim=1) ### remove repeating edges


def edge_features(CCO_matrix:torch.long,X_input:torch.float)->torch.float32:
    '''Input: CCO_matrix, X_input
     output: edge features including : displacement vector and its euclidean norm'''

    x_y_in   = torch.index_select(X_input,0,CCO_matrix[0,:])[:,:2]
    x_y_out  = torch.index_select(X_input,0,CCO_matrix[1,:])[:,:2]
    u_ij     = x_y_in - x_y_out
    norm_    = torch.norm(u_ij,dim=1)

    #return torch.cat([u_ij,norm_.unsqueeze(1)],dim=1)
    return norm_.unsqueeze(1)

def velocities_cav(dir:str,split_by:str)->torch.float32:
    '''Input file name:
    Output: u_x,u_y,cav'''
    data   = dir.split(split_by)[-1].split('_')
    u_x    = float(data[1])/100
    u_y    = float(data[2])/100
    p_amb  = float(data[3])
    cav    = float(data[-2])/100
    return u_x,u_y,p_amb,cav

def read_file_n(dir_,split_by):

    df = pd.read_csv(dir_,sep =' ',header=None)
    df = df.drop(labels=list(range(1, 15, 2)),axis=1)
    df.columns = ['node_num','node_type','x','y','P','v_x','v_y','cav']
    df = df.astype({"P": float, "cav": float})

    '''all nodes have the same starting velocity'''
    df['u_x'], df['u_y'],df['p_amb'],df['cav'] = velocities_cav(dir_,split_by)

    #df['P'] = df['P']-df['p_amb']

    '''no slip condition'''
    df.loc[df['node_type']==2,'u_x'] = 0
    df.loc[df['node_type']==2,'u_y'] = 0

    return df

def node_data(df):

    node_dict_ = {x:y for x,y in zip(df['node_num'],list(range(len(df))))}
    coord_velocities = torch.tensor(df[['x','y','u_x','u_y']].values,dtype=torch.float32)
    nodes_types = one_hot(torch.tensor(df['node_type']))
    X_input = torch.cat([coord_velocities,nodes_types],dim=1)
    X_output = torch.tensor(df[['P','v_x','v_y']].values,dtype=torch.float32)

    return X_input,X_output,node_dict_


class dataset_graph_(Dataset):
    '''class initiates by providing directory to simulation data in _n.txt,_c.txt files'''
    def __init__(self, files:list,split_by:str, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.files = files
        self.split_by = split_by

    def len(self):
        return int(len(self.files) / 3)

    def get(self, idx:int):
        '''The function takes simulation number and returns model input in graph format and
                  target variable in matrix form'''
        c = idx*3
        n = c + 2

        X_input, X_output, node_dict_ = node_data(read_file_n(self.files[n],self.split_by))
        coo_matrix                    = connectivity_data(self.files[c], node_dict_)
        edge_feat                     = edge_features(coo_matrix,X_input)

        Graph = Data(x=X_input, edge_index=coo_matrix,edge_weight=edge_feat,y=X_output)

        return Graph

