# test_detection
**For car and van detection used models:** 

-FasterRCNN - not quite modern but simple to use (build in torchvision). Classes: car and truck

-resnext50 - most accurate on imagenet from available models in torchvision. Used to specify "vans" subclass from "truck" detection class.

**run command: 
**
python annotate.py --video=Traffic_Monitoring_1.mp4

**Possible steps to improve annotations:
**
-change frequency of detection

-get rid of of obscured objects

-use more modern NN from benchmarks 
https://paperswithcode.com/sota/image-classification-on-imagenet
https://paperswithcode.com/sota/object-detection-on-coco

