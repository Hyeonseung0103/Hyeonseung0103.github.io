---
layout: single
title: "RetinaNet 정리"
toc: true
toc_sticky: true
category: Detection
---

RetinaNet은 one stage detector의 대표주자인 YOLO, SSD보다 높은 성능을 기록하면서 Faster R-CNN보다 빠른 수행시간을 기록한 모델이다. 특히, 작은 object에 대한 detection 능력도 뛰어난데 이번 포스팅에서는 이 RetinaNet에 대해 간단히 정리해보자.

# Focal Loss의 필요성
Classification에서는 cross entropy를 손실함수로 많이 사용한다. 하지만, class가 imbalance할 때는 cross entropy가 상대적으로 잘 맞추고있는 class임에도 단순히 데이터 수가 많아서 loss의 많은 부분을 차지할 수 있다는 문제가 발생한다. Two stage detector에서는 RPN을 통해 객체가 있을만한 높은 확률 순으로 필터링을 수행한 후 탐지를 할 수 있지만 one stage detector에서는 모든 region(e.g. anchor box)에 대해 탐지를 수행해야하기때문에 class imbalance 문제가 더 도드라진다.

예를 들어, detection이 쉬운 데이터를 easy examples, 어려운 데이터를 hard examples이라고 할 때 배경과 같이 흔한 easy examples이 10,000개 자전거와 같이 예측하고자 하는 hard examples이 50개 라고 하자. 만약, Easy examples이 평균적으로 loss가 0.1이고 hard example이 1이라면 에러의 총합은 easy examples이 1,000(0.1 * 10,000), hard examples이 50(1 * 50)으로 이미 잘 맞추고있는 easy examples의 에러가 더 크게 취급된다. 결국, 우리가 잘 예측해야하는 것은 hard examples이기때문에 cross entropy를 사용하면 이와 같이 데이터의 분포는 고려되지 않은채 학습이 진행되어 학습이 불안정할 수 있다.

기존에는 augmentation이나 데이터셋 샘플링으로 이를 보완하려고 했지만 너무 많은 리소스가 필요하기때문에 RetinaNet에서는 Focal loss를 사용하여 이를 해결했다. 
Focal loss는 cross entropy($CE(p,y) = -\Sigma y_i ln p_i$) 공식에 가중치를 적용하는 방식이고 해당 클래스에 대한 확률이 높을수록(객체가 존재한다고 확신할수록) $\gamma$를 조절해 loss를 더 낮게하여 오히려 잘 예측하지 못한 클래스에 더 집중하도록 한다.

$$ FL(p_t) = -\Sigma y_i (1-p_t)^{\gamma} log(p_t) $$

<br> 
<div align="center">
  <p>
  <img width="600" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/088cde99-0ccb-4930-83db-e529162b5962">
  </p>
</div>

<br>

이 Focal loss를 활용해서 Cross entropy를 손실함수로 사용했을때보다 더 좋은 정확도를 기록했다.

# Feature Pyramid Network(FPN)
CNN에서는 층이 깊어질수록 추상적인 정보만 남아서 앞단의 세밀한 이미지 정보를 기억하기 어렵다는 문제가 있다. FPN은 이러한 문제를 해결하기 위한 기법으로 각 층의 피처맵을 예측에 사용할 피처맵과 결합하여 이미지 정보를 최대한 유지시키는 아이디어다.

<br> 
<div align="center">
  <p>
  <img width="500" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/862606a5-8611-40f8-b77f-cdd671cd5ddf">
  </p>
</div>
<br>

Backbone에서 bottom-up(사이즈는 줄이고, 채널은 늘림)으로 추출한 피처맵을 top-down(사이즈를 2배로 키우고, 채널은 그대로)으로 upsampling한 피처맵과 결합하여 이 결합한 피처맵을 예측에 사용하는 것이다.
해당 피처맵에서 계산된 손실을 모두 반영하여 loss를 계산한다. 이 방법은 여러 layer의 피처맵을 예측에 사용함으로써 단일 피처맵을 사용하는 것보다 다양한 이미지 정보를 사용할 수 있다는 장점이 있다. 
또한, 각 layer의 피처맵마다 grid에 9개의 anchor box가 할당되고 anchor는 k개의 클래스 확률값과 4개의 box regression 좌표를 가진다.

<br> 
<div align="center">
  <p>
  <img width="700" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/6c78e185-9eb2-403b-96e8-f7986da10975">
  </p>
</div>
<br>

Faster R-CNN에 FPN을 적용했을 때 성능이 향상했고, RetinaNet의 성능이 one stage detector뿐만 아니라 two satge detector인 Faster R-CNN과 비교해도 가장 높은 것을 알 수 있다.

# 구현
Pytorch로 RetinaNet 모델을 사용해보자([코드 참고](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/retinanet.py)). 데이터셋 및 파일 경로 설정은 [Fast & Faster RCNN 포스팅 구현 파트 참고](https://hyeonseung0103.github.io/detection/Fast_and_Faster_RCNN/).

```python
# 모델 정의
import torchvision
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection import _utils as det_utils
from functools import partial
from torch import nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.retinanet_resnet50_fpn_v2(pretrained = True)

num_classes = 3 # has ball, no ball, background

in_channels = model.backbone.out_channels
num_anchors = model.anchor_generator.num_anchors_per_location()[0]
norm_layer = partial(nn.GroupNorm, 32)

model.head = RetinaNetHead(in_channels, num_anchors, num_classes, norm_layer)
model = model.to(device)
```

```python
# 데이터 정의
train_dataset = SoccerDataset(TR_DATA_PATH, TR_LAB_PATH, get_transforms(train=True))
val_dataset = SoccerDataset(VAL_DATA_PATH, VAL_LAB_PATH, get_transforms(train=False))

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True,
                                                collate_fn = utils.collate_fn)

val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False,
                                                collate_fn = utils.collate_fn)
```


```python
num_epochs = 30
val_loss_tmp = 10000
best_epoch_tmp = 1
early_stopping_cnt = 0
early_stop = 7

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001,
                            momentum=0.9, weight_decay=0.0005)

# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                 step_size=3,
#                                                 gamma=0.9)

#lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=lambda lr: 0.95 ** lr)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.0001)

print('----------------------train start--------------------------')

for epoch in range(1, num_epochs+1):
  start = time.time()
  model.train()
  epoch_loss = 0
  prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

  for images, targets in prog_bar:
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    loss_dict = model(images, targets)

    optimizer.zero_grad()
    loss = sum(loss for loss in loss_dict.values())
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  print(f'epoch : {epoch}, Loss : {epoch_loss}, time : {time.time() - start}')

  with torch.no_grad():
    epoch_val_loss = 0
    val_start = time.time()
    for images, targets in val_data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        val_loss_dict = model(images, targets)
        epoch_val_loss += sum(loss for loss in val_loss_dict.values())

    print(f'Val Loss : {epoch_val_loss}, time : {time.time() - val_start}')
    if epoch_val_loss < val_loss_tmp:
        early_stopping_cnt = 0
        best_epoch_tmp = epoch
        val_loss_tmp = epoch_val_loss
        torch.save(model.state_dict(),f'{WEIGHTS_PATH}retinanet_{num_epochs}.pt')
    else:
        early_stopping_cnt += 1
    print(f'현재까지 best 모델은 Epochs {best_epoch_tmp}번째 모델입니다.')

  if early_stopping_cnt == early_stop:
    print(f'{early_stop}번 동안 validation 성능 개선이 없어 학습을 조기 종료합니다.')
    break
```


```python
# 데이터 정의
test_dataset = SoccerDataset(TEST_DATA_PATH, TEST_LAB_PATH, get_transforms(train=False))

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False,
                                                collate_fn = utils.collate_fn)
```


```python
evaluate(model, test_data_loader, device=device) # mAP@0.5:0.95 0.635, mAP@0.5 0.893
```


```python
from torchvision.ops import nms
i, t = test_dataset[10]
model.eval()
with torch.no_grad():
    prediction = model([i.to(device)])[0]

selected_idx = nms(prediction['boxes'], prediction['scores'], iou_threshold = 0.5)
selected_boxes = torch.tensor(prediction['boxes'])[selected_idx]
selected_labels = torch.tensor(prediction['labels'])[selected_idx]
selected_scores = torch.tensor(prediction['scores'])[selected_idx]

i, t = test_dataset[10]
i = np.array(i.permute((1, 2, 0)) * 255).astype(np.uint8).copy()
for idx,x in enumerate(selected_boxes):
  if selected_scores[idx] > 0.9:
    x = np.array(x.cpu(), dtype = int)
    cv2.rectangle(i, (x[0], x[1]), (x[2],x[3]), color = (0,255,0), thickness = 2)
    cv2.putText(i, str(selected_labels[idx].tolist()), (x[0],x[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color = (255,0,0), thickness= 3)
plt.imshow(i)
```

<br> 
<div align="center">
  <p>
  <img width="500" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/3c307ee2-bff6-44d7-8df5-05ee509e411e">
  </p>
  <p>nms 적용 후 RetinaNet 모델의 test 이미지 결과(confidence threshold 0.9)</p>
</div>
<br>

<br> 
<div align="center">
  <p>
  <img width="500" alt="image" src="https://github.com/Hyeonseung0103/Hyeonseung0103.github.io/assets/97672187/ab693c59-3161-4f15-a572-a1b4a0ecf4d1">
  </p>
  <p>nms 적용 후 SSDLite 모델의 test 이미지 결과(confidence threshold 0.9)</p>
</div>
<br>

위의 이미지를 보면, RetinaNet은 confidence score가 0.9이상인 박스들을 추출하면 원하는 객체를 올바르게 탐지하는 반면 SSD는 5개의 객체 중 3개의 객체만 탐지한다. 또한, RetinaNet은 해당 객체들 중 볼을 소유하고 있는 객체의 클래스를 1이라고 올바르게 예측했다.

RetinaNet은 Focal Loss와 FPN을 활용하여 당시 one stage detector에서 좋은 성능을 보였던 YOLO와 SSD보다 뛰어난 정확도를 가졌고, two stage detector인 Faster R-CNN보다도 높은 정확도를 기록했다고 논문에 언급되었다. 

실제로 구현을 해보니 RetinaNet은 SSD 0.546, Faster R-CNN 0.407, YOLOv1 0.34보다 높은 0.635의 mAP를 기록했다. 비록 YOLO와 SSD보다는 학습 시간(custom dataset 기준 한 에포크당 20초)이 느리긴하지만 한 에포크당 학습 시간이 Faster R-CNN(한 에포크당 3분 30초)보다 약 1.4배 정도 더 빠른 2분 30초의 시간이 소요됐다. 



# Reference
- [RetinaNet Paper](https://arxiv.org/pdf/1708.02002.pdf)
