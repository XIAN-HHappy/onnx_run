# ONNX Runtime

## 项目配置  
### 1、作者开发环境：  Windows10 （vs 2019 or vs 2022）
### 2、项目依赖
* opencv 3.4.15   
      安装方法：自行安装配置项目依赖。  
* 如模型运行平台 GPU :     
- [x] onnxruntime-win-x64-gpu-1.3.0      
- [x] cuda_10.1.105_418.96_win10       
- [x] cudnn-10.1-windows10-x64-v7.6.5.32   
      安装方法：自行安装配置项目依赖。        
* 如模型运行平台 CPU :    
- [x] Microsoft.ML.OnnxRuntime.1.10.0 & Microsoft.ML.OnnxRuntime.MKLML.1.6.0     
      安装方法：     
          进入 visual studio 编辑环境，选择：工具 -> NuGet 包管理器 -> 管理解决方案的 NuGet程序包，    
      然后选择“浏览”，在搜索栏分别输入 “Microsoft.ML.OnnxRuntime.1.10.0”、“Microsoft.ML.OnnxRuntime.MKLML.1.6.0”，进行安装。  

## 项目依赖工具包   
* [项目依赖工具包下载地址(百度网盘 Password: s46l )](https://pan.baidu.com/s/18KhFz5_ea-HCcRWqzHXwhg)   

## ONNX 模型
* [handpose_X 模型下载地址(百度网盘 Password: obxa )](https://pan.baidu.com/s/13oo9t4DGKu26yQ9Th393Ag)      

## 项目使用方法  
### 1、如GPU运行自行确定软件已安装及硬件环境符合。
* 注意：另外通过 main.cpp 中定义  USE_CUDA ，确定是运行在GPU or CPU 硬件上。
### 2、配置项目依赖库。   
### 3、配置模型本地路径。  
### 4、配置样本路径。  
### 5、运行项目。  
