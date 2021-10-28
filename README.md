# Pruned DSFD

This is pruned version of DSFD.

# Setup
- tensorflow==1.15
# Training
## Training Method
> python pruned_train.py 

## How to change learning rate
* Open pruned_train_config.py, change lr_value and lr_decay to modify piecewise learning rate. 

## Save ckpt file
* Learned models are saved in a ckpt file. The storage path can be set by changing the model_path of the config file. It is recommended to change the storage path every time when you train with changed the hyperparameter.
* The learned models are stored in the model_path path each epoch.
* Model_path/optimized folder stores models with the lowest validation loss.

## fine tuning pre-trained model
* Let's prune unnecessary connections of the existing standard model by adding pruning loss.
* Enter the ckpt path of the previously learned standard model in the pre-trained_model.
e.g.) config.MODEL.pretrained_model='./face_detector/optimized/epoch_5_val_loss_3.ckpt'

* Example) 0 - 999 step : 0.001 / 1000 - 9999 step: 0.005 / 10000 - step: 0.0005
config.TRAIN.lr_value_every_step = [0.001,0.005,0.0005]
config.TRAIN.lr_decay_every_step = [1000,10000]

## Modifying the pruning loss
* Open pruned_train_config.py
1. Enter the path of the ckpt file of the standard model  in the pre-trained_model.
2. Adjust the ratio of other loss by changing the pruned_alpha.
3. Set the ratio per layer of changed pruned_ratio.
* Reduce batch size if memory error occurs and does not return
* Yes.)
config.TRAIN.batch_size = 4
config.TRAIN.pruned_alpha = 1.
config.TRAIN.pruned_ratio = [0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

# How to execute AP calculation code

1. Produce pb file. The pb file contains the graphs and parameters only necessary for inference.
> tools/auto_freeze.py --ckpt_path= 'ckpt file path to test' output_path= 'folder to store the detector.pb file'

2. (about 40 minutes) Rotate the face detector for all validation images to extract results, and save the results as a text file.
> python tools/wider.py --model='pbfile storage path' --data_dir='WIDER FACE validation image storage path' --result='result='result='result storage path'

3. Calculating AP scores.

> cd tools/ap_eval && python setup.py build_ext --inplace

> python tools/ap_eval/evaluation.py --pred='Store results'

# Code for MAC Calculation
Open mac_config.py to modify the path to the ckpt file path to test the path to the train_model.

> python mac.py

### Reference
This is modification of this repo:
[DSFD-tensorflow](https://github.com/610265158/DSFD-tensorflow)
