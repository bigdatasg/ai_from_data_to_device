# Module 3: Model Deployment with IoT
## Overview
This module walks through a process in deploying and using a machine learning model inside an IoT prototype board (i.e. Raspberry Pi).

## What you'll learn
In this lab, you will learn how to do the following:
- Prepare the model so it can be used inside IoT prototype board (i.e. Raspberry Pi).
- Consuming the model using tflite format
- Converting to and using an Edge TPU model 
- Integrating with Camera and SenseHat

## Prerequisites

### Skills
- Basic proficiency in Python

### Equipment/Software
- Raspberry Pi Model 3 (available at the test stations)
- Coral Accelerator (available at the test stations)

## Task 01: Prepare Workspace
Note: This assumes you have now access to the prepared raspberry pi setup.
1. Open the terminal
![](assets/pi_terminal.png)
2. Create a temporary workspace by executing the following in the terminal:
```
rm -rf /home/pi/edgeiot_workspace && 
mkdir /home/pi/edgeiot_workspace && 
cd /home/pi/edgeiot_workspace
```
3. Clone the workshop repository by running the command below:
```
git clone https://github.com/bigdatasg/ai_from_data_to_device.git &&
cd /home/pi/edgeiot_workspace/ai_from_data_to_device/coral_workshop
```

## Task 02: Compile Tensorflow Lite models for the Edge TPU
1. Open PI's web browser and navigate to the following address:
```
https://coral.withgoogle.com/web-compiler/
```
2. Upload the following tensorflow lite model files and convert them 1 at a time using the above web compiler. Follow the steps in the site on how to upload.
```
test_data/mobilenet_v1_0.25_128_quant.tflite
test_data/inception_v4_299_quant.tflite
```
Download the compiled models.    
Close the web browser.    

3. Move the downloaded files to the following folder:
```
/home/pi/edgeiot_workspace/ai_from_data_to_device/coral_workshop/test_data
```

## Task 03: Classify images using the compiled Edge TPU models
1. Ensure you're back to the workspace folder:
```
cd /home/pi/edgeiot_workspace/ai_from_data_to_device/coral_workshop
```
2. Run the following command to classify image using mobilenet:
```
python3 classify_image.py --model test_data/mobilenet_v1_0.25_128_quant_*_edgetpu.tflite --label test_data/imagenet_labels.txt --image images/cat-image.jpg
```
3. Run the following command to classify another image using mobilenet:
```
python3 classify_image.py --model test_data/mobilenet_v1_0.25_128_quant_*_edgetpu.tflite --label test_data/imagenet_labels.txt --image images/merlion-park-tower.jpg
```
4. Run the following command to classify the same image using inception:
```
python3 classify_image.py --model test_data/inception_v4_299_quant_*_edgetpu.tflite --label test_data/imagenet_labels.txt --image images/merlion-park-tower.jpg
```

## Task 04: Perform transfer learning on Edge TPU
1. Download dataset by running the following:
```
DEMO_DIR=/tmp

wget -P ${DEMO_DIR} http://download.tensorflow.org/example_images/flower_photos.tgz

tar zxf ${DEMO_DIR}/flower_photos.tgz -C ${DEMO_DIR}
```
2. Download an embedding extractor:
```
wget -P ${DEMO_DIR} https://dl.google.com/coral/canned_models/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite
```
3. Start on-device transfer learning
```
python3 classification_transfer_learning.py \
--extractor ${DEMO_DIR}/mobilenet_v1_1.0_224_quant_embedding_extractor_edgetpu.tflite \
--data ${DEMO_DIR}/flower_photos \
--output ${DEMO_DIR}/flower_model.tflite \
--test_ratio 0.95
```

## Task 05: Use pre-trained object detection model
1. Ensure that webcam is connected to the raspberry pi
2. Run the following:
```
 python3 coral_live_object_detection.py --model test_data/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --label test_data/imagenet_labels.txt
```
3. Show the camera some objects to identify
4. Once done, press ctrl+c to end the live detection.

## Congratulations! You've completed the Coral EdgeTPU (IoT) Workshop!


