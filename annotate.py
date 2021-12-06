import cv2
import matplotlib.pyplot as plt
#from tqdm.notebook import tqdm
import tqdm
import json
import os
import numpy as np
import torch
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ms_coco_classnames import cl_dict as coco_names
from imagenet_classes import imagenet1000_cls
import argparse

ap = argparse.ArgumentParser()
#ap.add_argument("-v", "--video", required=True, 
#	help="path to input video file")

ap.add_argument("-v", "--video", default='Traffic_Monitoring_1.mp4',
	help="path to input video file")
args = vars(ap.parse_args())

#cap=cv2.VideoCapture('Traffic_Monitoring_1.mp4')
cap=cv2.VideoCapture(args["video"])
ret, frame=cap.read()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                    min_size=800)
model = model.eval().to(device)

resnext50_32x4d = torchvision.models.resnext50_32x4d(pretrained=True)
resnext50_32x4d=resnext50_32x4d.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
])

transform_classifier = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])


COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # add a batch dimension
    outputs = model(image) # get the predictions on the image

    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score and necessary classes
    labels=outputs[0]['labels'].cpu().numpy()
    #condition=(np.where((labels ==3)) or np.where((labels ==8)))
    #condition=np.where(pred_scores >detection_threshold)
    condition=np.intersect1d(np.where(np.logical_or((labels ==3) , (labels ==8))), np.where(pred_scores >0.6))
    """
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    scores=pred_scores[pred_scores >= detection_threshold]#.astype(np.int32)
    labels=outputs[0]['labels'][pred_scores >= detection_threshold].cpu().numpy()
    """
    boxes = pred_bboxes[condition].astype(np.int32)
    scores=pred_scores[condition]#.astype(np.int32)
    labels=labels[condition]
    #pred_classes=pred_classes[pred_scores >= detection_threshold]
    pred_classes=[coco_names[i] for i in labels]
    #return boxes, pred_classes, outputs[0]['labels'], scores #
    return boxes, pred_classes, labels, scores #
    
def draw_boxes(boxes, classes, labels, image):
    # read the image with OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image
    
fps = cap.get(cv2.CAP_PROP_FPS)
frames_nmb=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with open('annot_example.json', 'r') as f:
    annot_data=json.load(f)

h, w = frame.shape[:2]

annot_data['size']['height']=h
annot_data['size']['width']=w


fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps=int(cap.get(cv2.CAP_PROP_FPS))
out_fps=1
writer = cv2.VideoWriter('video_vis.mp4', fourcc, out_fps, (w, h))
obj_id=0
#while cap.isOpened():
#while i<900:
for i in range(0, frames_nmb):
    ret, frame=cap.read()
    if not ret: break
    if i%fps==0:
        annot_data['objects']=[]
        boxes, classes, labels, scores = predict(frame, model, device, 0.6)
        img_vis = draw_boxes(boxes, classes, labels, frame)
        for box, class_name, score, label in zip(boxes, classes, scores, labels):
            #x1, y1, x2, y2=box
            x1, y1, x2, y2=map(int, box)
            if class_name=='car':
                cur_obj={'id': obj_id, 'classId': 2956171, 'classTitle': 'car', \
                         'points': {'exterior': [[x1, y1], [x2, y2]], 'interior': []}}
                annot_data['objects'].append(cur_obj)
            if class_name=='truck':
                truck_crop=frame[y1:y2,x1:x2, :]
                im = Image.fromarray(np.uint8(truck_crop))
                img_t = transform_classifier(im)
                batch_t = torch.unsqueeze(img_t, 0)
                output=resnext50_32x4d(batch_t)
                ind=output.argmax().item()
                #output.shape
                name=imagenet1000_cls[ind]
                if name in ['moving van', 'minivan', 'police van, police wagon, paddy wagon, patrol wagon, wagon, black Maria']:
                    cur_obj={'id': obj_id, 'classId': 2956175, 'classTitle': 'van', \
                             'points': {'exterior': [[x1, y1], [x2, y2]], 'interior': []}}
                    annot_data['objects'].append(cur_obj)
                    #print('van detected!')
            obj_id+=1
        writer.write(img_vis)
        with open('annot_{}.json'.format(i), 'w') as f:
            json.dump(annot_data, f)
writer.release()
