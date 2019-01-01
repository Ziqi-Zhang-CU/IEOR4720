# IEOR4720 Team O
The datasets we used are not included in this repo as they exceed the uploading limit.(UCF, Shanghai Tech) Please let me know if you need them to test the performance. To test the code, one should produce the json file (train.json, val.json, test.json) using the script UCF_json_generate.ipynb or Shanghai_json_generate.ipynb first, and produce ground truth data using the script UCF_make_dataset.ipynb or Shanghai_make_dataset.ipynb. Then modify the root of files accordingly. Finally, run train file (train_CSRNet.py, train_ResNet.py, train_ResNet_vgg.py) by "python train.py train.json val.json 0 0". After that, a model will be generated according to the name one gave in the util.py. Finally, can test the model on testing data by running val_testingdata.ipynb.

When replicating the paper about CSRNet, we referred to the source:
https://github.com/leeyeehoo/CSRNet-pytorch/tree/master

When constructing the Resnet, we referred to the source:
https://github.com/LiMeng95/pytorch_hand_classifier/blob/master/network/resnet.py


Train_Val_Performance: 
Analyze how performance is improved during the training process and when the best model is saved.

val_Testingdata: 
Test model on testing dataset. Print out density map for sanity check. 

model_Resnet.py: 
The Resnet model. ResNet 18 and ResNet 34 could be constructed using it. 

model_Resnet_vgg.py: 
A ResNet-like CSRNet model

model_CSRNet.py: 
CSRNet model

train_Resnet.py: 
Train ResNet 18 and ResNet 34 

train_Resnet_vgg.py: 
Train ResNet-like CSRNet model

train_CSRNet.py: 
Train CSRNet model
