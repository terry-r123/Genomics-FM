# Geomics-FM
This repository includes the 'Genomics-FM: Universal Foundation Model for Versatile and Data-Efficient Functional Genomic Analysis' project.  

## 1. Environment setup



#### 1.1 Create and activate a new virtual environment
For example, you can use python>=3.6
```
conda create -n dnalm python=3.8
conda activate dnalm
```



#### 1.2 Install the package and other requirements

(Required)
Personally, I recommend fresh students use some stable version of Pytorch to match 3090 GPU cluster such as 1.7.1, which has been verified.
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/terry-r123/Genomics-FM.git
cd Genomics-FM
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```
#### 1.3 Install the apex requirements (Optional, install apex for fp16 training)

change to a desired directory by `cd PATH_NAME`

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

if you meet some errors, you can try the following:
first delete and uninstall the original apex, then
```
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
# Maybe you also need to comment  raise RuntimeError("Cuda extensions are being compiled with a version of Cuda that does...." with 'pass' in the 52nd line in setup.py
# Maybe vi ~/.bashrc and Add "export TORCH_CUDA_ARCH_LIST="8.0", then source ~/.bashrc 
python setup.py install --cuda_ext --cpp_ext
pip3 install -v --no-cache-dir ./
```
#### 1.4 checkpoint
google drive link: https://drive.google.com/drive/folders/17-MQdeD9G-F7uiuHWWMdP50MmgJvpJb9?usp=drive_link
