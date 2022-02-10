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


def read_file(dir):
    with open(dir, 'r') as fd:
        lines = []
        for line in fd:
            lines.append(line)
    return lines


def connectivity_data(dir_c, dict_):
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


def edge_features(CCO_matrix,X_input):
    '''Input: CCO_matrix, X_input
     output: edge features including : displacement vector and its euclidean norm'''

    x_y_in   = torch.index_select(X_input,0,CCO_matrix[0,:])[:,:2]
    x_y_out  = torch.index_select(X_input,0,CCO_matrix[1,:])[:,:2]
    u_ij     = x_y_in - x_y_out
    norm_    = torch.norm(u_ij,dim=1)

    return torch.cat([u_ij,norm_.unsqueeze(1)],dim=1)


def velocities_cav(dir):
    '''Input file name:
    Output: u_x,u_y,cav'''
    data = dir.split('\\')[-1].split('_')
    u_x  = float(data[1])/100
    u_y  = float(data[2])/100
    cav  = float(data[-2])/100

    return torch.tensor([u_x,u_y,cav],dtype=torch.float32)


def node_data(dir):
    '''Input: Node file directory
    Out: Node feat vector X_input, X_output, and node translation dictionary'''
    files2 = read_file(dir)
    counter = 0
    node_dict_ = {}
    node_type = []

    output_nodes_data = torch.zeros(len(files2),3)                 # v_x,v_y,P
    input_coord       = torch.zeros(len(files2),2)                 # x,y
    u_x_u_y_cav       = velocities_cav(dir).repeat(len(files2), 1) # values the same for all input graph nodes


    for i, line in enumerate(files2):
        line_ = line.split('  ')
        node_number = int(line_[0])
        node_dict_[node_number] = counter
        counter += 1
        node_type.append(int(line_[1]))
        '''float_data = x,y,P,v_x,v_y'''
        x, y, P, v_x, v_y      = [float(line_[x]) for x in range(2, 7)]

        output_nodes_data[i,:] = torch.tensor([P,v_x,v_y],dtype=torch.float32)
        input_coord[i,:]       = torch.tensor([x,y],dtype=torch.float32)

    node_type_tensor = one_hot(torch.tensor(node_type, dtype=torch.long))

    '''Input graph nodes data'''
    X_input  = torch.cat([input_coord,u_x_u_y_cav,node_type_tensor], dim=1)  # x,y,u_x,u_y,cav,node_type

    '''Output graph nodes data'''
    X_output = output_nodes_data   # P,v_x,v_y

    return X_input, X_output, node_dict_


class dataset_graph_(Dataset):

    def __init__(self, dir, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)
        self.files = glob.glob(dir)

    def len(self):
        return int(len(self.files) / 3)

    def get(self, idx):
        c = idx*3
        n = c + 2

        X_input, X_output, node_dict_ = node_data(self.files[n])
        coo_matrix                    = connectivity_data(self.files[c], node_dict_)
        edge_feat                     = edge_features(coo_matrix,X_input)

        Graph = Data(x=X_input, edge_index=coo_matrix,edge_attr=edge_feat)

        return Graph,X_output

