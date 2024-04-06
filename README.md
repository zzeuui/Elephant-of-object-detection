# Setting
### 1. docker
```
$ docker pull nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
$ docker run -it --gpus all --name container_name -v /path/to/workspace:/root -v /path/to/dataset:/root/dataset nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04

# apt-get update
# apt-get install sudo
```

### 2. anaconda
```
# sudo apt install curl -y
# curl --output anaconda.sh https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
# sha256sum anaconda.sh
# bash anaconda.sh
# sudo apt-get install vim
# vim ~/.bashrc
# source ~/.bashrc
# conda -V

# exit

$ docker exec -it container_name bash
```

### 3. Pytorch
```
# conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
# python
>>> import torch
>>> torch.cuda.is_available()
True
```

### 4. Detectron2
```
# sudo apt-get install git
# git clone https://github.com/facebookresearch/detectron2.git
# python -m pip install -e detectron2
# cd detectron2
# mv configs/ detectron2
```

### 5. Test
```
# git clone https://github.com/Vastlab/Elephant-of-object-detection.git
# cd Elephant-of-object-detection

# vim main.py
...
change the path on 162 line to your dataset path.
maybe it is '/root/dataset/something/..'
..
```

in my case. my gpu is small..
```
# vim ~/detectron2/detectron2/engine/defaults.py
...
change the "opts" on 134 line to "--opts"
...

# python main.py --num-gpus 1 --config-file training_configs/faster_rcnn_R_50_FPN.yaml --opts SOLVER.IMS_PER_BATCH 4
...
can check the "SOLVER.IMS_PER_BATCH" in 562 line of detectron2/detectron2/config/defaults.py 
...
```

# The Overlooked Elephant of Object Detection: Open Set
This repository contains the code for the evaluation approach proposed in the paper [The Overlooked Elephant of Object Detection: Open Set](https://openaccess.thecvf.com/content_WACV_2020/papers/Dhamija_The_Overlooked_Elephant_of_Object_Detection_Open_Set_WACV_2020_paper.pdf)

Our paper may be cited with the following bibtex
```
@inproceedings{dhamija2020overlooked,
  title={The Overlooked Elephant of Object Detection: Open Set},
  author={Dhamija, Akshay and Gunther, Manuel and Ventura, Jonathan and Boult, Terrance},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={1021--1030},
  year={2020}
}
```

Evaluation for the wilderness impact curve is now supported using detectron2

### Dataset is expected under `datasets/` in the following structure 
For Pascal VOC:
```
VOC20{07,12}/
  JPEGImages/
```

For MSCOCO:

```
coco/
  {train,val}2017/
    # image files that are mentioned in the corresponding json
```

### Training models for the evaluation
In order to run the evaluation please prepare a model trained with the protocol files in this repo.

You may use the following command to train a FasterRCNN model:

```
python main.py --num-gpus 8 --config-file training_configs/faster_rcnn_R_50_FPN.yaml
```

For convenience models trained with the config files in this repo have been provided at: https://vast.uccs.edu/~adhamija/Papers/Elephant/pretrained_models/

### Running the evaluation script

Please ensure your config is correctly set to load the models trained above. You might want to set the `OUTPUT_DIR` detectron2 config

The following command may be used to run the complete evaluation

```python main.py --num-gpus 2 --config-file training_configs/faster_rcnn_R_50_FPN.yaml --resume --eval-only```
