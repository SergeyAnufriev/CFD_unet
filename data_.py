"""The module purpose is to load simulation data in _n.txt,_c.txt files into
pytorch geometric dataset object https://pytorch-geometric.readthedocs.io/en/latest/notes/create_dataset.html,
which will be used to create data batches, where single entity withing batch includes one CFD simulation,which
is described by input graph and target graph node feature matrix """


import torch
from torch_geometric.data import Dataset
import glob
from torch.nn.functional import one_hot
from torch_geometric.data import Data



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
    data = dir.split(split_by)[-1].split('_')
    u_x  = float(data[1])/100
    u_y  = float(data[2])/100
    #cav  = float(data[-2])/100

    #return torch.tensor([u_x,u_y,cav],dtype=torch.float32)
    return torch.tensor([u_x,u_y],dtype=torch.float32) # make input velocities dimentionless by dividing |v_inp|

def node_data(dir:str,split_by):
    '''Input: Node file directory
    Out: Node feat vector X_input, X_output, and node translation dictionary'''
    files2 = read_file(dir)
    counter = 0
    node_dict_ = {}
    node_type = []

    output_nodes_data = torch.zeros(len(files2),3)                 # v_x,v_y,P
    input_coord       = torch.zeros(len(files2),2)                 # x,y
    u_x_u_y           = velocities_cav(dir,split_by).repeat(len(files2), 1) # make input velocities dimless


    for i, line in enumerate(files2):
        line_ = line.split('  ')
        node_number = int(line_[0])
        node_dict_[node_number] = counter
        counter += 1
        node_type.append(int(line_[1]))

        '''no slip condition'''
        if int(line_[1]) == 2:
            u_x_u_y[i,:] = torch.zeros(2,1,dtype=torch.float32)

        '''float_data = x,y,P,v_x,v_y'''
        x, y, P, v_x, v_y      = [float(line_[x]) for x in range(2, 7)]

        output_nodes_data[i,:] = torch.tensor([P,v_x,v_y],dtype=torch.float32)  # make velocities and pressure dimless
        input_coord[i,:]       = torch.tensor([x,y],dtype=torch.float32) # normilise input coord by min max to [-1,1]

    node_type_tensor = one_hot(torch.tensor(node_type, dtype=torch.long))

    '''Input graph nodes data'''
    X_input  = torch.cat([input_coord,u_x_u_y,node_type_tensor], dim=1)  # x,y,u_x,u_y,cav,node_type

    '''Output graph nodes data'''
    X_output = output_nodes_data   # P,v_x,v_y

    return X_input, X_output, node_dict_


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

        X_input, X_output, node_dict_ = node_data(self.files[n],self.split_by)
        coo_matrix                    = connectivity_data(self.files[c], node_dict_)
        edge_feat                     = edge_features(coo_matrix,X_input)

        Graph = Data(x=X_input, edge_index=coo_matrix,edge_weight=edge_feat,y=X_output)

        return Graph

