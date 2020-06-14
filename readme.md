This project is for scene text detection.

# Installation
## Clone the repo
```
git clone https://yomhub@dev.azure.com/yomhub/FasterRCNN/_git/FasterRCNN
```
Denote the root directory path of datasets by `${ds_root}` and project path by `${pj_root}`.  

Link dataset folder to prooject root:
```
ln -s ${ds_root} ${pj_root}/mydataset
```

Install conda environment
```
conda create --name <env> --file conda_env.txt
```

## Prerequisites
 (Only tested on) Ubuntu 16.04 and windows with:
* tensorflow-gpu >= 2.1    
* numpy    
* matplotlib



