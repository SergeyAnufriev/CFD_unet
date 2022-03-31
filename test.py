'''The file purpose is to check model forward pass'''


from torch_geometric.data import DataLoader
from data_ import dataset_graph_
from torch_geometric.nn import GraphUNet
from torch.nn import L1Loss

### test git

dir_    = r'C:\Users\zcemg08\PycharmProjects\CFD_unet\dataset2\graphical_data\*txt'
dataset = dataset_graph_(dir_)
dataset.split_by ='\\'

loader  = DataLoader(dataset, batch_size=2, shuffle=True)

model   = GraphUNet(in_channels=9,hidden_channels=128,out_channels=3,depth=2)

for graph,target in loader:
    break


output = model(graph.x,graph.edge_index)


print(graph.edge_weight.shape)

print(graph.batch)
print('output shape',output.shape)
print('target shape',target.shape)


l = L1Loss()
new_target = target.view(output.shape)

print('new target shape',new_target.shape)
loss = l(output,new_target)

print(loss)
