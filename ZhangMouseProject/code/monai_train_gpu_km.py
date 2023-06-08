import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from monai.handlers.utils import from_engine
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    RandCropByLabelClassesd,
    EnsureTyped,
    NormalizeIntensityd
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


import torch
import glob

# initialize params
mainDir = '/data/zhang_mri/data'
lr = 1e-4
numephs = 100
modelname = '' #set model name
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:3") #set which GPU to use

# train images
train_images = sorted(glob.glob(mainDir + '/images/train/' + 'id*.nii.gz'))
train_segs = sorted(glob.glob(mainDir + '/labels/train/' + 'id*.nii.gz'))


# validation images
val_images = sorted(glob.glob(mainDir + '/images/val/' + 'id*.nii.gz'))
val_segs = sorted(glob.glob(mainDir + '/labels/val/' + 'id*.nii.gz'))

# test images
# test_images = sorted(glob.glob(mainDir + '/images/test/' + 'volume*.nii.gz'))
# test_segs = sorted(glob.glob(mainDir + '/labels/test/' + 'segmentation*.nii.gz'))

print('Number of training images per epoch:', len(train_images))
print('Number of validation images per epoch:', len(val_images))
# print('Number of test images per epoch:', len(test_images))

# Creation of data directories for data_loader

train_dicts = [{'image': image_name, 'label': label_name}
               for image_name, label_name in zip(train_images, train_segs)]

val_dicts = [{'image': image_name, 'label': label_name}
             for image_name, label_name in zip(val_images, val_segs)]

# test_dicts = [{'image': image_name, 'label': label_name}
#               for image_name, label_name in zip(test_images, test_segs)]

train_transforms = [
    LoadImaged(keys=['image', 'label']),
    AddChanneld(keys=['image', 'label']),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True),  # intensity (z-score normalization using mean and std)
    RandCropByLabelClassesd(keys=['image', 'label'], label_key="label", spatial_size=[256,16,256], ratios=[1, 1],
                            num_classes=2, num_samples=8),
    RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=1),
    RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=0),
    RandFlipd(keys=['image', 'label'], prob=0.5, spatial_axis=2),
    ToTensord(keys=['image', 'label']), #converts image from NIfTI to Tensor?  If so, why here and not before?
    EnsureTyped(keys=["image", "label"]), #ensures data is PyTorch Tensor or numpy array
]

val_transforms = [
    LoadImaged(keys=['image', 'label']),
    AddChanneld(keys=['image', 'label']),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    NormalizeIntensityd(keys=["image"], nonzero=True),  # intensity
    ToTensord(keys=['image', 'label']),
    EnsureTyped(keys=["image", "label"]),
]

train_transforms = Compose(train_transforms) # from https://docs.monai.io/en/stable/_modules/monai/transforms/compose.html:
                            # "provides the ability to chain a series of callables together in a sequential manner." What???
val_transforms = Compose(val_transforms)

# create a training data loader
check_train = CacheDataset(data=train_dicts, transform=train_transforms) # from https://docs.monai.io/en/stable/data.html:
                                                                        # caches results of all non-randomized transforms
                                                                        # (up through NormalizeIntensityd in our case)
                                                                        # to save time re-running those on each epoch
# we set batch size to 1 because it iterates on patients, which all have 9 crops
# ^ why 9 crops?? Does original get saved too, or would it actually be 8 crops or however many you set for in RandCropbyLabelClassesd?
train_loader = DataLoader(check_train, batch_size=1, shuffle=True, num_workers=1)
#  What does this mean from documentation:      num_workers: how many subprocesses to use for data loading.
#        ``0`` means that the data will be loaded in the main process.  (default: ``0``)
# in DataLoader, first arg is dataset from which to load data.  But doesn't check_train only load data prior to randomized transformations?

# create a validation data loader
check_val = CacheDataset(data=val_dicts, transform=val_transforms)
# we set further validation params for sliding window below in training loop
val_loader = DataLoader(check_val, batch_size=1, num_workers=1)


model = UNETR(   #UNet Transformers
    in_channels=1,
    out_channels=2,
    img_size=(256,16,256), # is this where we can change to 2D?
    feature_size=16, # dimension of network feature size
    hidden_size=768, #dimension of hidden layer - similar to num of nodes/neurons?
    mlp_dim=3072, #dimension of feedforward layer - does this ever need to change?
    num_heads=12, #num of attention heads.  What are attention heads???
    pos_embed="perceptron", # position embedding layer type
    norm_name="instance", # feature normalization type and arguments
    res_block=True, # residual block is used
    dropout_rate=0.2, # faction of input units to drop
).to(device)
# model.load_state_dict(torch.load(os.path.join(mainDir,"models","UNETR_2bs-8crop_96-96-32_1-1-1_1e4_500ep.pth")))
# model.to(device)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True) #include_background defaults to False, softmax converts vector of numbers into vector of probabilities
torch.backends.cudnn.benchmark = True
            # ?????  cudnn autotuner will look for optimal set of algorithms for configuration - fast when input sizes don't vary, slow when they do
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)  #implements AdamW algorithm as optimizer

# start a typical PyTorch training
post_label = AsDiscrete(to_onehot=True, num_classes=2)
post_pred = AsDiscrete(argmax=True, to_onehot=True, num_classes=2)
# These transform output to discrete values.
# In post_label, just converts to one-hot format,
# in post_pred also executes argmax function on input data before transform (finds class with largest predicted probability)
# I think this is where I kept getting obsolete errors, but not sure how to change.
# documentation here https://docs.monai.io/en/stable/_modules/monai/transforms/post/array.html says to do "to_onehot = 2"
# and not have num_classes as a separate arguement, but I got errors doing that. May have been before fixing
# num of classes earlier in code, but the error said something about "expecting an int"

dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False) #computes average dice score between two tensors
val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
# writer = SummaryWriter()
for epoch in range(numephs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{numephs}")
    model.train() #sets mode to train but actual training done in loop below? https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device))
        optimizer.zero_grad() # sets gradients of all optimized tensors to zero  (so in other words it won't compound losses? Each image's loss is calculated independently)
                # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
        outputs = model(inputs) # I'm assuming this actually runs the model on the input data
        loss = loss_function(outputs, labels)
        loss.backward() # accumulation of gradients happens here?  computes dloss/dx for every parameter x
        optimizer.step() # updtaes value of x using gradient from loss.backward
            # so if loss.backward gives x.grad += dloss/dx
            # optimizer.step() performs calculation like (for SGD)  x += -lr*x.grad
                # so loss.backward calculates gradient and optimizer updates weights based on lr and gradient - right?
        epoch_loss += loss.item()
        epoch_len = len(check_train) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")


    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad(): # disables gradient calculation to save memory consumption
            for val_data in val_loader:
                # print('starting val sliding window')
                val_images, val_labels = val_data["image"], val_data["label"]
                roi_size = [256,16,256]
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model, sw_device=device, device=torch.device("cpu"))
                        # looks at different parts of image same size as cropped training images, calculates probability for each patch, blends to create outputs?
                        # is batch size equal to number of patches for each image????
                        # roi_size is spatial window size for inferences, sw_batch_size is the batch size to run window slices
                        # when roi_size is larger than inputs' spatial size, input images are padded during inference then final output image is cropped to original input size
                        # we don't set overlap size.  Default seems to be 0.25?
                        #       need to be sure overlap*roi_size*output_size/input_size is an integer (for each spatial dimension).
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)] # applies argmax on output probabilities, converts to one-hot
                val_labels = [post_label(i) for i in decollate_batch(val_labels)] # converts ground truth labels to one-hot
                # compute metric for current iteration
                #dice_metric(y_pred=val_outputs, y=val_labels)
                value = dice_metric(y_pred=val_outputs, y=val_labels) # compares output labels to ground truth labels using dice
                value = value.detach().cpu().numpy()  # "detaches tensor from the current computational graph" - what????
                print(value)
                dice_metric.reset()
                # if np.isnan(value):
                #     continue
                metric_values.append(value)
            # metric = dice_metric.aggregate().item()
            metric = np.nanmean(np.array(metric_values))
            print(metric)


            # Save best model
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(mainDir,"models",modelname+".pth"))
                    # Just SAVES model with best validation metric, right?  Validation doesn't affect weights for next epoch?
                print("saved new best metric model")

            # Logger bar
            print(
                "current epoch: {} Validation dice: {:.4f} Best Validation dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )

            # writer.add_scalar("Mean_epoch_loss", epoch_loss, epoch + 1)
            # writer.add_scalar("Validation_dice", metric, epoch + 1)

print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")


# How to make 2D?
# How to save transformations to augment dataset?