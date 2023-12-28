---
layout: single
title: "Fast & Faster R-CNN 정리"
toc: true
toc_sticky: true
category: Detection
---

R-CNN이 다른 모델들에 비해 높은 mAP를 기록했고 이후 Fast-RCNN, Faster R-CNN, Mask R-CNN 등 여러 모델들이 R-CNN을 develop하여 object detection 분야에서 더 많은 발전을 이루어냈다. 이번 포스팅에서는 R-CNN 이후의 모델인 Fast R-CNN과 Faster R-CNN에 대해 간단하게 정리해보자.

# Spatial Pyramid Pooling Net(SPPNet)
Fast & Faster R-CNN을 더 잘 이해하기위해 먼저, SPPNet에 대해 알아보자.

R-CNN은 selective search와 CNN, SVM을 통해 object detection을 수행했는데 region proposals, feature extraction, detection이 모두 다른 네트워크에서 수행되어 학습 및 추론 시간이 느리다는 단점을 가지고있다.
또한, 2000개의 region proposals이 CNN에 입력되기 위해서는 모든 region이 고정된 크기의 벡터여야하는데 이를 위해 crop이나 warp를 사용해 사이즈를 조절했다. Crop/warp를 사용할 경우 실제 region과는 조금은 다른 이미지가 만들어지기때문에 성능에도 영향이 있다.

SPPNet은 이러한 문제점을 개선해서 2000개의 region proposals 이미지를 전부 CNN에 통과시키는 것이 아니라 CNN은 원본 이미지만 통과시켜 피처맵을 만들고 selective search로 나온 region을 이 피처맵과 맵핑시켜 학습하는 방법을 사용했다. 하지만, 피처맵과 맵핑된 region을 고정된 크기의 벡터로 flatten시켜야 분류기에 넣어 detection을 수행할 수 있는데 모든 region의 크기가 제각각이기때문에 이를 고정된 크기의 벡터로 만드는 것이 어려웠다.

SPPNet에서는 이를 해결하기위해 SPP Layer를 만들어서 Flatten을 시키기전에 어떤 크기의 region이 들어와도 분류기에 입력하기 전에 고정된 크기의 벡터로 변환했다. 아래 그림처럼 다양한 크기의 피처맵이 들어오면 해당 피처맵을 여러개의 분면으로 쪼개고 이를 합쳐서 고정된 크기의 벡터로 만들어 분류기에 입력으로 사용하는 것이다.

<br> 
<div align="center">
  <p>
  <img width="500" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/52ba1d69-7923-400d-bcf8-624bd5cca7ba">
  </p>
</div>

<br>

이를 통해 성능의 개선 뿐만 아니라 detection 수행 시간도 크게 단축됐다.

# Fast R-CNN
Fast R-CNN은 SPPNet을 조금 더 보완시켜 SPP layer를 RoI pooling layer로 바꾼 네트워크라고 할 수 있다. 또한, 위에서 언급한 R-CNN의 문제를 여러가지 방법을 통해 해결했는데 먼저 SVM을 softmax 네트워크로 변환시켰고
classification과 regression을 혼합하여 사용하는 multi-task loss 함수를 통해 end-to-end network(RoI proposals 제외)를 구축했다.

SPPNet이 다양한 크기의 피처맵을 미리 정해놓은 여러개의 분면으로 쪼개 고정된 크기의 벡터로 만들었다면, Fast R-CNN은 RoI pooling을 통해 어떤 크기의 피처맵이 들어와도 모두 고정된 크기로 max pooling하는 기법을 사용한다.
보통 7 x 7 크기로 pooling 시키는데 만약 피처맵의 크기가 7의 배수가 아니라면 보간이나 이미지를 resizing하여 크기를 맞춘다.

손실함수로는 multi task loss를 사용하고 특히 regression loss에는 smooth $L_1$을 적용해서 R-CNN과 SPPNet에서 사용된 $L_2$ loss보다 outliers에 덜 민감하도록 하고, 
loss가 1보다 작으면 loss를 더 작게해서 큰 loss에 더 집중할 수 있도록 한다. $\lambda$는 classification과 box regression loss의 balance를 맞추는 용도로 사용된다.

<br> 
<div align="center">
  <p>
  <img width="500" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/a12312fb-2d42-4850-bf1a-078d715be168">
  </p>
  <p>
  <img width="500" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/ecf36288-8f1f-4017-89a8-b251ef7459c7">
  </p>
  <p>
  <img width="450" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/6b7d4f67-7234-4a08-9a28-a5fb794dc05a">
  </p>
</div>
<br>

Fast R-CNN의 학습 과정을 정리하면 다음과 같다.

1) 원본 이미지를 CNN에 통과시켜 feature extraction을 수행
   
2) selective search를 통해 나온 2000개의 region proposals을 원본 이미지의 피처맵과 맵핑

3) 맵핑된 다양한 크기의 region proposals 피처맵에 7x7 RoI pooling 적용

4) RoI pooling을 통해 나온 output으로 detection 수행(multi task loss)


<br> 
<div align="center">
  <p>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/19f94db4-a3da-4c4f-b07a-b4af21e67fa2">
  <br>
  <img width="300" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/95ca23fc-703a-4540-9348-901ee76d56e8">\
  </p>
</div>
<br>

결과적으로 Fast R-CNN은 속도나 정확도 측면 모두에서 기존 R-CNN보다 크게 개선된 mAP를 기록했다.

# Faster R-CNN
Fast R-CNN은 CNN, RoI pooling, 분류기를 결합해 특징 추출과 detection을 하나의 네트워크에서 수행하여 R-CNN보다 훨씬 빠른 속도로 detection이 가능했지만 여전히 region proposals에는 selective search 알고리즘을 사용하여 완벽한 end-to-end network를
구축하진 못했고 one stage model보다 속도 측면에서 좋지 않은 performance를 보였다. Faster R-CNN은 이러한 문제를 해결하기위해 Fast R-CNN에 RPN(Region Proposals Network)을 결합하여 selective search를 대체한
완벽한 end-to-end network를 구축했다.

End-to-End Network가 구축되면 classification, box regression 뿐만 아니라 region에 대한 back propagation도 가능해지기때문에 2000개로 한정된 selective search 알고리즘보다 더 효율적인 proposals이 가능하다.
그렇다면, 이미지에서 어떻게 selective search와 같이 region을 예상하여 제안할 수 있을까?

여기에서 사용된 개념이 anchor box이다. Anchor box는 한 픽셀당 고정된 여러 스케일의 boxes를 사용하여 region proposals을 수행한다. 아래 그림처럼 각각의 피처맵의 한 픽셀당 여러 ratio와 크기를 가진 k개의 anchor box가 있다고 할 때 anchor box마다 object의 존재여부 scores와($2k$ scores) 4개의 box 좌표($4k$ scores)가 output으로 도출된다. Anchor boxes가 ground truth box와 일치한다의 기준은 IoU가 가장 높은 anchor나 0.7이상인 anchor를 positive, 0.3이하를 negative로 분류하고 그 외의 애매한 box는 학습에서 제외시킨다.

<br> 
<div align="center">
  <p>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/a37bfcfd-7fb1-44a9-83da-afe1129efbab">
  </p>
</div>
<br>

손실 함수는 Fast R-CNN처럼 multi task loss를 사용하지만 anchor box와 관련된 계수들이 추가되었다. $p_i$는 anchor box내 객체가 object일 확률이고, $p_{i}^{\star}$는 해당 object가 ground trurh와 일치하면 positive로 1, 일치하지 않으면 negative 0으로 취급한다. Box regression은 anchor box가 클래스를 올바르게 예측한 즉, positive에 대해서만 수행하고 $t_i$는 anchor box와 모델이 예측한 bbox와의 차이, $t_{i}^{\star}$는 anchor box와 ground truth간의 차이이다. Faster R-CNN의 손실함수에서 특이한 점은 예측 bounding box와 ground truth box의 차이를 anchor box와 각각의 prediction box와의 차이를 사용하여 계산한다는 것이다. 

이 방법은 anchor box를 참고해서 anchor를 기준으로 GT와 predicted box의 차이가 비슷할수록 두 박스의 거리가 가까울 것이라는 아이디어이다. 객체의 존재여부를 하나도 모르는 bbox를 생성하고 조정하는 것보다 positive anchor를 참고하여 bbox를 GT에 가깝게 조금씩 조정하는 것이 더 효율적인 방법이 된다. $N_{cls}$는 positive와 negative anchor의 비율을 동일하게 가져가기 위한 정규화 파라미터고 $N_{reg}$는 박스 갯수를 정규화한 값이다.

<br> 
<div align="center">
  <p>
  <img width="450" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/9911a43e-0cde-4a93-8700-cd1a62dacf80">
  <br>
  <img width="500" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/6db411c0-3104-4626-bcbc-d353de5b57e6">
  </p>
</div>
<br>

Faster R-CNN의 학습 과정을 정리하면 다음과 같다.

1) 원본 이미지를 CNN을 통과시켜 특징 추출

2) 각각의 피처맵에 대해 픽셀당 여러 스케일을 가진 anchor box를 그리고 각각의 boxes를 RoI pooling으로 고정된 크기의 벡터로 변환

3) Anchor boxes에 대해 object 존재여부와 ground truth와의 일치여부를 계산하고, 일치한다면 anchor box, predicted box, ground truth box를 통해 box regression 수행

<br> 
<div align="center">
  <p>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/129d93c3-c0d4-4c8b-a7f2-e78b70484aef">
  <br>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/bcc38a5d-40ec-4d16-9b0a-138a76217953">
  </p>
</div>
<br>

Faster R-CNN은 selective search 보다 anchor boxes를 활용한 RPN을 사용했을 때 성능이 크게 향상됐고, Fast R-CNN보다 개선된 모델임을 알 수 있다.

# 구현
Pytorch로 Faster R-CNN model을 사용해보자. 데이터는 Roboflow에서 제공하는 [soccer dataset](https://universe.roboflow.com/yinguo/soccer-data)을 사용했다. Class는 공을 소유한 사람과 소유하지 않은 사람으로 구분된다.

```python
# engine 라이브러리 사용을 위한 git clone
!git clone https://github.com/pytorch/vision.git
%cd vision
!git checkout v0.3.0

!cp references/detection/utils.py ../
!cp references/detection/transforms.py ../
!cp references/detection/coco_eval.py ../
!cp references/detection/engine.py ../
!cp references/detection/coco_utils.py ../
```

```python
!pip install -q torch==1.13.0 torchvision==0.14.0 # 런타임 다시 시작
```

```python
import json
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from pycocotools.coco import COCO
from PIL import Image
import time
import transforms as T
```

```python
DATA_PATH = '/content/drive/MyDrive/논문실습/data/'
TR_DATA_PATH = '/content/drive/MyDrive/논문실습/data/coco_format/train/'
VAL_DATA_PATH = '/content/drive/MyDrive/논문실습/data/coco_format/valid/'
TEST_DATA_PATH = '/content/drive/MyDrive/논문실습/data/coco_format/test/'
TR_LAB_PATH = '/content/drive/MyDrive/논문실습/data/coco_format/train/_annotations.json'
VAL_LAB_PATH = '/content/drive/MyDrive/논문실습/data/coco_format/valid/_annotations.json'
TEST_LAB_PATH = '/content/drive/MyDrive/논문실습/data/coco_format/test/_annotations.json'
```

```python
# Custom Datset 정의
class SoccerDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, label_path, transforms):
        self.data_path = data_path
        self.label_path = label_path
        self.transforms = transforms

        self.imgs = [x for x in sorted(os.listdir(data_path)) if '.jpg' in x]
        self.labs = COCO(label_path) # COCO를 사용하여 라벨을 쉽게 불러올 수 있다

        # 이미지 전체 id
        all_img_ids = self.labs.getImgIds() # 이미지 id 전체 가져오기
        self.img_ids = []

        for idx in all_img_ids:
            annotations_ids = self.labs.getAnnIds(imgIds=idx, iscrowd=False) # 해당 img_id에 일치하는 annotation들
            if len(annotations_ids) == 0: # 만약 list가 0이면 해당 image에는 annotation이 없는 것
                print(idx)
            else:
                self.img_ids.append(idx)
    
    def __getitem__(self, idx):
        image = self.load_image(idx)
        lab = self.load_annotatinos(idx)
        
        boxes = torch.tensor(lab[:,:4])
        labels = torch.tensor(lab[:,4], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # width * height
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64) # 해당 라벨들은 군집화되어있지 않음. 따라서 box수만큼 0표시

        target = {} # engine 라이브러리를 사용하기위해선 항상 아래 형식의 target을 만들어야한다.
        target['boxes'] = boxes
        target['labels'] = labels
        target["image_id"] = torch.tensor(self.img_ids[idx])
        target['area'] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            # 이렇게 넘어온 key, value 값으로 다시 리턴해줌
            # bboxes, labels라는 key로 transformed dict 리턴
            transformed = self.transforms(image = image, bboxes = boxes, labels=labels)
            image = transformed['image']
            target['boxes'] = torch.tensor(transformed['bboxes']) # 변환된 box로 정의
            target['labels'] = torch.tensor(transformed['labels'])
            
        return T.ToTensor()(image, target) # ToTensor는 이미지를 정규화하고 (C,H,W) 형식으로 만듬
    
    def load_image(self, img_idx):
        image_info = self.labs.loadImgs(self.img_ids[img_idx])[0] # 일치하는 이미지 정보를 가져옴
        img = cv2.imread(self.data_path + image_info['file_name'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def load_annotatinos(self, img_idx):
        annot_ids = self.labs.getAnnIds(imgIds=self.img_ids[img_idx], iscrowd=False) # 해당 img id와 일치하는 annot_id
        annots = np.zeros((0,5)) # box좌표와 category_id를 넣을 array

        if len(annot_ids) == 0:
            print('No annotations in this image')

        coco_annots = self.labs.loadAnns(annot_ids) # 라벨이 존재하면 모든 라벨 정보 저장
        for idx, x in enumerate(coco_annots):
            annot = np.zeros((1,5))
            annot[0,:4] = x['bbox']
            annot[0,4] =  x['category_id'] # 학습시 loss_box_reg가 0인거면 라벨링이 잘못된것
            annots = np.append(annots, annot, axis = 0)
        
        # w,h를 x2,y2형식으로 변환
        annots[:,2] = annots[:,0] + annots[:,2]
        annots[:,3] = annots[:,1] + annots[:,3]
        #print(annots)
        return annots

    def __len__(self):
        return len(self.imgs)
```

```python
import albumentations as A

def get_transforms(train):
    transforms = []
    if train:
        transforms.append(A.HorizontalFlip(0.5))
        transforms.append(A.VerticalFlip(0.5))
    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    # label_fields는 호출할 때 입력한 key와 맞아야함. key이름을 labels로 했으니까 label_field로 labels
    # 이미 x2,y2형식으로 바꿨으니까 coco가 아닌 pascal 형식으로 리턴
```

```python
# 모델 정의
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, AnchorGenerator

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights = 'DEFAULT') # 사전학습된 가중치 그대로 사용
num_classes = 3 # 1: has ball, 2: no ball, 0: background

# 분류기에 사용할 입력 정보
in_features = model.roi_heads.box_predictor.cls_score.in_features

# 모델의 헤드 부분 교체
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
```

```python
# forward 잘 되는지 테스트
a = SoccerDataset(TR_DATA_PATH, TR_LAB_PATH, get_transforms(train=True))
dl = torch.utils.data.DataLoader(a, batch_size=2, shuffle=True,
  collate_fn=utils.collate_fn)
# # 학습 시
images,targets = next(iter(dl))

output = model(images,targets)
output
```

```python
from engine import train_one_epoch, evaluate
import utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_dataset = SoccerDataset(TR_DATA_PATH, TR_LAB_PATH, get_transforms(train=True))
val_dataset = SoccerDataset(VAL_DATA_PATH, VAL_LAB_PATH, get_transforms(train=False))

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2,
                                                collate_fn = utils.collate_fn)

val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=2,
                                                collate_fn = utils.collate_fn)

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                 step_size=3,
#                                                 gamma=0.8)
lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=lambda lr: 0.9 ** lr)

num_epochs = 15

#start = time.time()
for epoch in range(14, num_epochs+1):
    # iteration10 마다 결과 출력
    train_one_epoch(model, optimizer, train_data_loader, device, epoch, print_freq=10)
    # 학습률 업데이트.
    lr_scheduler.step()
    
    evaluate(model, val_data_loader, device=device)

print("학습 종료 ", (time.time() - start) // 60, ' 시간 소요')
torch.save(model.state_dict(),f'{WEIGHTS_PATH}faster_rcnn_{num_epochs}.pt')
```

```python
# 모델 평가. 테스트용으로 하나만
i, t = val_dataset[0]
model.to(device)
model.eval()
with torch.no_grad():
    prediction = model([i.to(device)])[0]

i = np.array(i.permute((1, 2, 0)) * 255).astype(np.uint8).copy()
for idx, x in enumerate(prediction['boxes']):
  x = np.array(x.cpu(), dtype = int)
  cv2.rectangle(i, (x[0], x[1]), (x[2],x[3]), color = (0,255,0), thickness = 2)
  cv2.putText(i, str(prediction['labels'][idx].tolist()), (x[0],x[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color = (255,0,0), thickness= 3)
plt.imshow(i)
```
<br> 
<div align="center">
  <p>
  <img width="500" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/dfbf7705-0dd8-4cfe-9c59-12e5540a1b84">
  </p>
  <p>
    nms 적용 전 이미지
  </p>

</div>

<br>

```python
from torchvision.ops import nms

selected_idx = nms(prediction['boxes'], prediction['scores'], iou_threshold = 0.1)
selected_boxes = torch.tensor(prediction['boxes'])[selected_idx]
selected_labels = torch.tensor(prediction['labels'])[selected_idx]

i, t = val_dataset[0]
i = np.array(i.permute((1, 2, 0)) * 255).astype(np.uint8).copy()
for idx,x in enumerate(selected_boxes):
  x = np.array(x.cpu(), dtype = int)
  cv2.rectangle(i, (x[0], x[1]), (x[2],x[3]), color = (0,255,0), thickness = 2)
  cv2.putText(i, str(selected_labels[idx].tolist()), (x[0],x[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color = (255,0,0), thickness= 3)
plt.imshow(i)
```

<br> 
<div align="center">
  <p>
  <img width="500" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/3b617a59-e9ea-4e2f-b3ff-b40096277e4a">
  </p>
  <p>
    nms 적용 후 이미지
  </p>

</div>

<br>

```python
# 데이터 정의
test_dataset = SoccerDataset(TEST_DATA_PATH, TEST_LAB_PATH, get_transforms(train=False))

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False,
                                                collate_fn = utils.collate_fn)
evaluate(model, test_data_loader, device=device)
```

위의 이미지를 통해, Faster RCNN model은 일반 RCNN 모델([RCNN 포스팅 참고](https://hyeonseung0103.github.io/detection/RCNN/))보다 localization을 잘 수행하고 분류 성능도 나쁘지않다는 것을 알 수 있다. Epochs를 15 정도만 했는데도 test set의 mAP50이 약 0.65였고, map@0.5:0.95는 0.407이었다. 테스트 목적이 아니라 실제로 성능을 높이기위해 에포크 수를 늘린다면 더 좋은 성능을 기록할 것이다. 

RCNN은 selective search와 detection이 다른 네트워크에서 이루어지기때문에 한 에포크당 약 20분의 시간이 걸렸는데 Faster RCNN은은 3분 30초 정도로 RCNN보다 5배이상 빨랐다. 이를 통해 Faster RCNN은 anchor box를 기반으로 하나의 네트워크에서 region proposals, classification, box regression을 수행할 수 있기때문에 RCNN보다 훨씬 빠르면서 구현이 쉬운 모델이라는 것을 몸소 느낄 수 있었다.

# Reference
- [SPPNet](https://arxiv.org/pdf/1406.4729v4.pdf)
- [Fast R-CNN](https://arxiv.org/pdf/1504.08083.pdf)
- [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf)
