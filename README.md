# CFD_unet
Unet application to CFD data


Data description 

Files with $_n.txt$ extention store CFD nodes information, where the columns are ordered by:

|index|node\_num|node\_type|x|y|P|u\_x|u\_y|cav|
|---|---|---|---|---|---|---|---|---|
|0|0|2|0\.115|0\.0001817|15622\.5|0\.0|0\.0|1\.0|
|1|1|1|0\.5|0\.25|16542\.9|1\.97403|-0\.33367|1\.0|
|2|2|1|0\.5|-0\.25|16541\.9|1\.95862|-0\.315459|1\.0|
|3|3|3|-0\.5|-0\.25|16493\.2|1\.97203|-0\.333595|1\.0|
|4|4|3|-0\.5|0\.25|16545\.1|1\.97199|-0\.333572|1\.0|


Proposed model