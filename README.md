# Comparison of image processing approach and machine vision methods in counting apples

## Introduction

This repository was created for the Group Project for the msc robotics engineering of University of Bristol.

This repository contains programs for image processing approach and machine vision.

Please click here to see the report and instructions. [Document](https://1drv.ms/f/s!AkUkwfqMyJzClTsm5lHBdoG_hD2d?e=jhv9K0)

## About project

Apple counting in orchards traditionally relies on manual labour, which is time-consuming and error prone. Computer vision technology provides a new possibility for the automated management of orchards, through image algorithms to achieve automatic detection and counting of apples, thereby improving efficiency and reducing labour costs. This project compares the detection performance of image processing approach methods based on color extraction and edge detection with the YOLOv5 algorithm in an orchard environment. The image processing approach method has low computational cost but poor robustness to light and complex background, while YOLOv5 shows high accuracy and strong robustness under occlusion, overlapping and complex background conditions through multi-layer feature extraction and optimization techniques. The experimental results verify the advantages of YOLOv5 and explore its applicability under the condition of limited hardware resources, which provides an important reference for the automation of orchard management.

## Steps

1. data reprocess

2. split dataset

3. cuda test

   

   ```
   conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2
   conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly
   ```

   Enter the following code into the python interpreter to verify the installation of torch.

   ```
   import torch
   print(torch.__version__)
   print(torch.cuda.is_available)
   ```

   

4. Download yolo environment 

   ```
   git clone https://github.com/ultralytics/yolov5.git 
   cd yolov5
   ```

5. Test and score the test set

6. Compare the test results with the ground_truth file of the test set

7. Find the optimal and worst results

8. Apple precision evaluation

   

## Authors

This project was created by part of the group of students from robotics student MSC at the University of Bristol.

- Zhiyi Liu
- Jelly Jinzhe Liu
- Junfeng Ren
- Yuqi Zhang
- Yuhao Gu

## Acknowledgments

We would like to thank our project professor, Dr. Jisi Chen, Dr. Wenhao Zhang , for his guidance and support throughout the project.
