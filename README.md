# CNLVRObjectDetection
Trains a MASK-RCNN model to do Object Detection on the CNLVR Dataset

## Requirements
Same as the [Matterport Mask RCNN](https://github.com/matterport/Mask_RCNN)


## Generating Data

To train a MASK-RCNN, you need images and their respective masks. As NLVR is a synthetic dataset, we can modify the files by [Alane Suhr](https://github.com/alsuhr-c/nlvr-baselines) to generate the masks. As those files already draw the images, we can easily generate the masks on top of them and save them.

In order to generate one image with it's mask and save it in `filename`, you can add this code to the end of `generator/image_generator.py`. You must also comment out everything after line 228.
```
# choose your type here
current_image = generate_image(ImageType.SCATTER)
if not current_image:
    continue
# where you want the image and it's mask to be saved
filename = 'sample_image'
# generate the mask and save it
current_image.png(box_size=BOX_SIZE, filename=filename)

```

Otherwise, you can use our image generator process which will generate all the data you need to train the model and save it in the folders `generated_images`, `validation_generated_images`, `generated_images_tower`, and `validation_generated_images_tower`. If you want to generate validation data, set the `generating_validation` flag to True. If you want to run generate a single image and it's mask, comment out anything after line 228 and paste in the code above there.

Run: `python generator/image_generator.py`. 


## Training the MASK RCNN model

To train the model, do:
`python3 cnlvr_maskrcnn.py`

This will train the model for 10 epochs (can be modified by changing the epochs parameter) and save the results in the logs directory.

We basically just load the image with masks, resize them, and feed them into the MASK RCNN model with appropriate parameters. It's an easy task, so it trains pretty quick.

If you want to do inference only (after training), set the `inference_only` flag to True.