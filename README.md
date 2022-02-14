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



![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)


