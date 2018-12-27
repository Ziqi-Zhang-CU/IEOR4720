# IEOR4720 Team O
The datasets we used are not included in this repo as they exceed the uploading limit.(UCF, Shanghai Tech) Please let me know if it is needed to be uploaded. To test the code, one should produce the json file using the script json_generate first, and produce ground truth data using the script make_dataset. Then modify the root of files accordingly. Finally, run train file by "python train.py train.json val.json 0 0". After that, a model will be generated according to the name one gave in the util.py. Finally, can test the model on testing data by running val_testing file.

When replicating the paper about CSRNet, we referred to the source:
https://github.com/leeyeehoo/CSRNet-pytorch/tree/master

When constructing the Resnet, we referred to the source:
https://github.com/LiMeng95/pytorch_hand_classifier/blob/master/network/resnet.py
