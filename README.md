# Lanenet-Lane-Detection (pytorch version)
  
## Introduction   
Use pytorch to implement a Deep Neural Network for real time lane detection mainly based on the IEEE IV conference paper "Towards End-to-End Lane Detection: an Instance Segmentation Approach".You can refer to their paper for details https://arxiv.org/abs/1802.05591. This model consists of ENet encoder, ENet decoder for binary semantic segmentation and ENet decoder for instance semantic segmentation using discriminative loss function.  

The main network architecture is:  
![NetWork_Architecture](./data/source_image/network_architecture.png)
## Configure the environment
```
conda create -n pytorch1.7.0 python=3.7
conda activate pytorch1.7.0
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt
```
## Generate Tusimple training set/validation set
First, download tusimple dataset [here](https://github.com/TuSimple/tusimple-benchmark/issues/3) through the kagge website.  

path/to/your/unzipped/file should like this:  
```
|--train_set
|----clips
|----label_data_0313.json
|----label_data_0531.json
|----label_data_0601.json
|--test_set
|----...
|--test_label.json
``` 

Then, run the following command to generate the training folder  and the train.txt,val.txt file.   
```
python tusimple_transform.py --src_dir ./TUSimple/train_set --val True
```

 

## Training the model    

Using tusimple folder with ENet/Focal loss:   
```
python train.py --dataset ./TUSimple/train_set/training
```
You cloud also replace  ENet and Cross Entropy with  DeepLabv3+ or Cross Entropy loss through following commands
```
python train.py --dataset ./TUSimple/train_set/training --loss_type CrossEntropyLoss
```
```
python train.py --dataset ./TUSimple/train_set/training --model_type DeepLabv3+
```
If you want to change focal loss to cross entropy loss, do not forget to adjust the hyper-parameter of instance loss and binary loss in ./model/lanenet/train_lanenet.py    

## Testing result    
A pretrained trained model by myself is located in ./log (only trained in 25 epochs)      
Test the model:    
```
python test.py --img ./data/tusimple_test_image/0.jpg --model_type ENet --model ./log/best_model.pth
```
The testing result is here:    
![Input test image](./data/source_image/input.jpg)    
![Output binary image](./data/source_image/binary_output.jpg)    
![Output instance image](./data/source_image/instance_output.jpg)    

## Evalution    
```python
python eval.py --dataset ./TUSimple/train_set/training --model_type ENet --model ./log/best_model.pth
```



## Reference:  
The project is modified from https://github.com/IrohXu/lanenet-lane-detection-pytorch.

