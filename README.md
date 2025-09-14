# HeRaKT
There is the project for paper "HeRaKT: Heterogeneous bidirectional modeling for response-aware knowledge tracing".

# Experimental environment
ubuntu=20.04.6  
python=3.9.20  
cuda=11.8

Please download the environment before running the project.
```
pip install -r requirements.txt
```

# Dataset
Due to Github upload rules, please download the dataset and place it to the path `\datasets`. The dataset we mainly selected is: 
[ASSIST2009](https://drive.google.com/drive/folders/19Uv_elM5xfV5Ocv4WieRRIkcKsKa3ZE_), 
[ASSIST2012](https://www.dropbox.com/scl/fo/sqxklwowothcun25vhufi/AC97TB9B2GXnKxUZSv-Cin8/data/assist12_3?dl=0&rlkey=u9wvepwxiuabjeg3f22tk57k5&subfolder_nav_tracking=1)
 and 
[EdNet](https://drive.google.com/drive/folders/19Uv_elM5xfV5Ocv4WieRRIkcKsKa3ZE_)
There are other datasets for reference only 
[ASSIST2017](https://drive.google.com/drive/folders/19Uv_elM5xfV5Ocv4WieRRIkcKsKa3ZE_), 
[JUNYI](https://github.com/DMiC-Lab-HFUT/DAGKT/tree/main/data/junyi_3)
 and 
[CSEDM](https://github.com/DMiC-Lab-HFUT/DAGKT/tree/main/data/csedm)

# Command
Run the local project. 
```
python main.py --dataset assist12_3 --model herakt --num_epochs 200 --batch_size 64 --device_id 2
```