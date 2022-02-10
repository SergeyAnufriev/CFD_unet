from torch_geometric.data import DataLoader
from data_ import dataset_graph_
from torch_geometric.nn import GraphUNet
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
from torch.nn import L1Loss

### test git 

dir_    = r'C:\Users\zcemg08\PycharmProjects\gitlearning\data_samples\data_samples\graphical_data\*txt'
dataset = dataset_graph_(dir_)

loader  = DataLoader(dataset, batch_size=2, shuffle=True)

model   = GraphUNet(in_channels=9,hidden_channels=128,out_channels=3,depth=2)

for graph,target in loader:
    break


output = model(graph.x,graph.edge_index)
print(graph.batch)
print('output shape',output.shape)
print('target shape',target.shape)


l = L1Loss()
new_target = target.view(output.shape)

print('new target shape',new_target.shape)
loss = l(output,new_target)

print(loss)
'''

graph ,X_out  = dataset.get(12)

X_feat  = graph.x[:,:2]
pos_    =  {x:(float(X_feat[x,0].numpy()),float(X_feat[x,1].numpy())) for x in range(len(X_feat))}




g = to_networkx(graph,to_undirected=True)



colors = X_out[:,2].numpy()

cmap=plt.cm.viridis
vmin = min(colors)
vmax = max(colors)

nx.draw(g,pos=pos_, with_labels=False,node_size=0.1,node_color=colors, vmin=vmin, vmax=vmax)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
sm._A = []
plt.colorbar(sm)
plt.show()
'''
