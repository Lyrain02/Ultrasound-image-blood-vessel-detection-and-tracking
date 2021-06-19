# Ultrasound-image-blood-vessel-detection-and-tracking
This is a course project from Digital Image Processing. It's aimed at detecting and tracing vessel in a given ultrasound video. This project is designed for specific ultrasound video and frames captured from that.

# 超声图像血管检测与跟踪
## 简介
本仓库为实现超声图像血管检测与跟踪建立了轮廓识别与动态追踪两个主要模型，并对其进行了进一步的精化调整。轮廓识别模型针对静态图片，对血管轮廓进行检测；动态追踪模型针对动态视频，对血管变换进行追踪。  
在轮廓识别模型中，首先针对特定帧图像，运用高斯模糊、二值化、形态学处理、轮廓检测等技术，建立粗糙的血管识别模型。紧接着，利用目标血管特殊的位置信息，进一步优化模型，增强模型的适应性。随后，针对运行结果不理想的具体图像，从形态学处理角度入手，对模型进行进一步的优化，确定最终的轮廓识别模型。  
在动态追踪模型中，利用opencv2自带的处理视频的函数，获取视频中的每一帧图像，并对每一帧图像使用轮廓识别模型，获取识别结果，从而实现对血管轮廓变换的动态追踪。  
## 模型
一、 轮廓识别模型  
  主模型  
        res_1.py    
  精化模型  
        refine_1.py    
        refine_2.py  
        refine_3.py [final]   

二、 动态追踪模型  
  主模型  
        res_2.py  
## 素材
images      用于res_1, refine_1, refine_2模型  
img2         用于refine_3模型        
video         用于res_2模型
