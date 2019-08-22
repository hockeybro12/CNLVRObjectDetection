import os
import sys
import random
import math
import re
import time
import numpy as np
import matplotlib
from skimage.transform import resize
import skimage.io

from tqdm import tqdm

# uncomment this if your Matterport Mask_RCNN is not in this path
#sys.path.append(ROOT_DIR)
from mrcnn import utils
from mrcnn.config import Config
import mrcnn.model as modellib

# parameters that you can change

# True if you want to do inference (get performance on validation set), False if you want to Train
inference_only = False

ROOT_DIR = ''
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class CNLVRConfig(Config):

    # Give the configuration a recognizable name
    NAME = "cnlvr"

    # color -> (black, blue, yellow)
    # shape -> (triangle, square, circle)
    # so 3 * 3 = 9 classes
    NUM_CLASSES = 9 + 1

    GPU_COUNT = 1
    IMAGES_PER_GPU = 10

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 64
    IMAGE_MAX_DIM = 256

    PRE_NMS_LIMIT = 4000

    IMAGE_RESIZE_MODE = 'none'

    # have more of these so that you can identify the smaller images
    RPN_TRAIN_ANCHORS_PER_IMAGE = 512

    # use smaller anchors as the objects are small
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)

    #RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # scale up the image so that we can identify the smaller objects in the images better
    IMAGE_MIN_SCALE = 2.0

    TRAIN_ROIS_PER_IMAGE = 32

class InferenceConfig(CNLVRConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class CNLVRDataset(utils.Dataset):


    # load all the images from the json file with their annotations
    def load_cnlvr(self, count=10, image_width=50, image_height=200, validation=False):

        # load the list which has all the paths
        if validation:
            scatter_paths_list = list(np.load('generation/validation_scatter_image_path.npy'))
            tower_paths_list = list(np.load('generation/validation_tower_image_path.npy'))
        else:
            scatter_paths_list = list(np.load('generation/scatter_image_path.npy'))
            tower_paths_list = list(np.load('generation/tower_image_path.npy'))

        # combine them and shuffle them so we have a mix of tower and scatter images
        paths_list = scatter_paths_list + tower_paths_list
        np.random.shuffle(paths_list)

        
        # add the class names to the class
        class_names = {'yellow circle': 1, 'yellow square': 2, "yellow triangle": 3, "black circle": 4, "black square": 5, "black triangle": 6, "blue circle": 7, "blue square": 8, "blue triangle": 9}
        class_names_list = list(class_names.keys())
        
        
        #for key, val in class_names.items():
        for key, val in sorted(class_names.items(), key=lambda x: x[1]):
            self.add_class("cnlvr", int(val), key)

        if count > len(paths_list):
            count = len(paths_list)

        print("Count is: " + str(count))

        base_image_path = 'generation/'

        # add all the images info to the path
        for i in range(count):
            self.add_image(
                    "cnlvr", image_id=i,
                    path=base_image_path + paths_list[i] + '.png',
                    width=100,
                    height=400,
                    annotations=base_image_path + paths_list[i] + '_mask.npy',
                    class_index=base_image_path + paths_list[i] + '_mask_index.npy')
            
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        return info["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]

        image = resize(image, (64, 256))
        image = 255 * image
        # Convert to integer data type pixels.
        image = image.astype(np.uint8)

        return image
        
        
    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        image_info = self.image_info[image_id]
        # get the annotations we had set before
        annotations_path = image_info["annotations"]
        class_index_path = image_info["class_index"]
        
        mask = np.load(annotations_path)
        class_ids = np.load(class_index_path)

        return mask.astype(np.bool), class_ids.astype(np.int32)

# Training dataset
dataset_train = CNLVRDataset()
dataset_train.load_cnlvr(count=100000, validation=False)
dataset_train.prepare()

# Validation dataset
dataset_val = CNLVRDataset()
dataset_val.load_cnlvr(count=500, validation=True)
dataset_val.prepare()

print("Dataset loaded")

if not inference_only:

    config = CNLVRConfig()

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.

    model.train(dataset_train, dataset_val, 
                learning_rate=config.LEARNING_RATE, 
                epochs=10, 
                layers='heads')
else:

    # we are in inference mode
    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Test on a random image
    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config, 
                               image_id, use_mini_mask=False)

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = np.random.choice(dataset_val.image_ids, len(dataset_val.image_ids))
    APs = []
    for image_id in tqdm(image_ids):
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)

        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]


        # Compute AP
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])

        APs.append(AP)
        
    print("mAP: ", np.mean(APs))










