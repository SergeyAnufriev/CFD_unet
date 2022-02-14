# CFD_unet
Unet application to CFD data


Data description 

The data set consists of files with $_n.txt$, $_c.txt$ and $_f.txt$, which represent CFD simulation.
The file names specify simulation conditions.
For example file $1e662_193_51_17291_744_149_n.txt$ specifies airfoil upstream conditions such as 

Upstream velocity in x direction, v_x = 1.93

Upstream velocity in y direction, v_y = 0.51

Upstream pressure                 P_{\inf} = 17291


Files with $_n.txt$ extention store CFD nodes information, where each rows gives nodes information.

|index|node\_num|node\_type|x|y|P|u\_x|u\_y|cav|
|---|---|---|---|---|---|---|---|---|
|0|0|2|0\.115|0\.0001817|15622\.5|0\.0|0\.0|1\.0|
|1|1|1|0\.5|0\.25|16542\.9|1\.97403|-0\.33367|1\.0|
|2|2|1|0\.5|-0\.25|16541\.9|1\.95862|-0\.315459|1\.0|
|3|3|3|-0\.5|-0\.25|16493\.2|1\.97203|-0\.333595|1\.0|

(Table 1)

x,y       - node's coordinate 

P,u_x,u_y - pressure and velocities in x and y direction

node_type - type of node including 
   0 - internal 
   1 - excit 
   2 - airfoil surface 
   3 - inlet

Files with $_c.txt$ extention store mesh (Figure 1) triangles connectivity data. 
The mesh consists of triangular elements,with each table row shows, triangles nodes numbers.
These node numbers correspond the above table node_num column. 


|index|0|2|4|
|---|---|---|---|
|0|880|3215|7058|
|1|1500|2946|7406|
|2|2946|3395|7406|
|3|1035|2688|7806|

(Table 2)

![alt text](https://github.com/SergeyAnufriev/CFD_unet/blob/main/images/airfoil-no-special-mesh.png)

(Figure 1)


Prediction task:

Given upstream conditions v_x,v_y,P_inf (file names) and mesh geometry ($_c.txt$) with coordinates x,y columns in ($_n.txt$)
predict P,u_x and u_y columns in ($_n.txt$). 


Proposed method: 

Graph u_net https://arxiv.org/pdf/1905.05178.pdf


https://github.com/SergeyAnufriev/CFD_unet/blob/main/images/graph_unet.PNG

![alt_text](https://github.com/SergeyAnufriev/CFD_unet/blob/main/images/graph_unet.PNG)

(Figure 2)

Graph unet takes Graph represented by adjecency matrix A and node feature vector X and 
outputs graph with the same adjecency matrix A but with node feature vector Y


In this case input node feature vector X represents:

1) upstream conditions equal for all graph nodes: u_x,u_y,P_inf (simulation file name)
2) node type node_type (Table 1)
3) node coordinate x,y (Table 1)

Graph adjecency matrix A obtained by:
1) Mesh triangles conectivety (Table 2)
2) Edge feature displacement vector |v_ij| between nodes and their norm |v_ij|
where v_ij = [x_i-x_j,y_i-y_j] as in 
https://arxiv.org/abs/2010.03409 (p 13, type Euclidean)

The output feature vector Y represents:
1) velocity in x direction u_x (Table 1)
2) velocity in y direction u_y (Table 2)
3) local pressure P (Table 2)

