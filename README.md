# Scene Text Detection via Decoupled Feature Pyramid Networks
This code is for review only.
## Requirements:  
tensorflow >=1.13  
python3  
Polygon  
sklearn  
Shapely  
Pillow  
numpy  
scipy
cython

## Checkpoints
Checkpoints for Total-Text are avaiable at [Total-Text](https://drive.google.com/drive/folders/1eFWIPgO_MTB98wzHttv3lfe7CwUq07HX?usp=sharing), the checkpoints are pre-trained on ICDAR MLT 2017, ICDAR 2015, ICDAR 2013, and finetuned on Total-Text. Unzip it and move the three files to the ./checkpoints folder.

Checkpoints for ICDAR 2015 are avaiable at [ICDAR 2015](https://drive.google.com/drive/folders/1_R6L1g_39YX6fW_w62ngTNw6k4IRUa3B?usp=sharing), the checkpoints are pre-trained on ICDAR MLT 2017, ICDAR 2015, ICDAR 2013, and finetuned on ICDAR 2015. Unzip it and move the three files to the ./checkpoints folder.


## Make
before runing the code, please:  
cd lanms; make; cd ../   
cd polynms; make; cd ../  

## Run  
run the code in three ways:  
### 1. sh run.sh       (with default parameters)  
### 2. if you want to test ICDAR 2015: 
       sh run_IC15.sh [gpu_list] 
### 3. if you want to test Total-Text: 
       sh run_Total.sh [gpu_list] 

## Parameters:  
    link_method: four methods for linking: "RegLink", "Box", "Mask", "DBSCAN"  
    test_data_path: the path of your images.  
    gpu_list: gpu id
    checkpoint_path: path of checkpoint.  
    output_dir: path of results, and it will create three folders:
              1: [link_method] : result txts of curve;
              2: [link_method]_box: result txts of quadrilateral;
              3: [link_method]_draw: images of results.

## =========================================
Notably, tensorflow will cost a lot of time at the first image, so if you want to test the speed of our method, please test at least two images. And please add 'TF_CUDNN_USE_AUTOTUNE=0' before python command (has been add in run.sh and sh run_link_method.sh).  

## Others:
### "DBSCAN" takes a lot of time for calculating IoU of each pair of boxes, so we implement GPU IoU (in polynms), but it has some bugs, and the H-mean is slightly lowwer than CPU method.
    1: eg, sh run_link_method.sh 0,1 DBSCAN
    using GPU for calculating IoU, the second GPU is used for calculating IoU.
    2: eg, sh run_link_method.sh 0,0 DBSCAN
    using one GPU, but it may cause OOM, "invalid device ordinal", or some other bugs.
    3: eg, sh run_link_method.sh 0 DBSCAN
    using CPU  for calculating IoU, but it is quite slow.
