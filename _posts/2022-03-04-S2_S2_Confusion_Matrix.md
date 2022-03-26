---
layout: single
title: "Section2 Sprint2 Note 223 Confusion Matrix, ROC curve, AUC"
category: Section2
toc: true
toc_sticky: true
---

분류 모델의 성능을 평가하는 데에는 accuracy외에도 여러가지 지표가 있다. 이번 포스팅에서는 다양한 지표들과 데이터에 따라 더 중요한 지표에 대해 알아보자.

### Confusion Matrix(혼동 행렬)
TP, TN, FP, FN으로 이루어진 매트릭스로 각 요소는 분류 모델의 성능을 평가하는 지표로 활용될 수 있다. 행렬의 요소는 예측값이 실제값과 일치하는 지의 여부를 나타내고 일치하고, 맞고 틀린 갯수가 나타나 있다.
상황에 따라 더 중요한 지표가 다르고, 파이썬 패키지에 따라 행과 열의 위치가 조금씩 다를 수 있다.

TP: 실제로 참인 것을, 예측도 참이라고 한 경우

TN: 실제로 거짓인 것을, 예측도 거짓이라고 한 경우

FP: 실제로 거짓인 것을, 예측은 참이라고 한 경우

FN: 실제로 참인 것을, 예측은 거짓이라고 한 경우

```python
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
#원래 confusion matrix
fig, ax = plt.subplots()
pcm = plot_confusion_matrix(pipe, X_val, y_val, #꼭 파이프가 아니라 그냥 분류 모델 넣어도 됨.
                            cmap=plt.cm.Blues,
                            ax=ax);
plt.title('Confusion matrix, threshold: 0.5')
plt.show()

#임계값을 0.4로 조정
threshold = 0.4
y_pred_proba = pipe.predict_proba(X_val)[:, 1]
y_pred = y_pred_proba > threshold
confusion_matrix(y_val, y_pred) # threhosld를 0.5에서 0.4로 바꾸니까 TP, FP값이 증가하고 FN,TN 값이 감소한다.
#이로 인해 recall 값이 증가하고, precision 값이 줄어들 것이다.
```

### Precision(정밀도)과 recall(재현율)

Accuracy에 한계: 데이터가 불균형하다면(특정 범주에 많은 데이터가 있는 것) 해당 범주로만 모두 찍어도 정확도가 어느 정도 나온다.

ex) 광주사람 90명 , 서울사람 10명이 모여 있는 곳에서 모든 사람을 광주사람이라고 찍어도 90/100 = 90%의 정확도가 나온다.
하지만 실제로는 서울사람에 대해서 하나도 맞추지 못했고, 만약 이 모델에 서울 사람의 데이터가 들어오면 더 맞추지 못할 것이다.

이를 방지하기 위해 Accuracy 외에도 정밀도와 재현율을 지표로 사용해야 한다. 맞았는지 틀렸는지 보다 진짜 문제를 해결하기 위한 예측이 잘 되었는지 볼 수 있는 지표가 필요하다.

![image](https://user-images.githubusercontent.com/97672187/156867417-6e1d247d-8ef8-4f4d-b3b2-bf791cf52075.png){: .align-center}

이미지 출처: https://towardsdatascience.com/accuracy-precision-recall-or-f1-331fb37c5cb9

정밀도는 내가 참이라고 예측한 값 중에 실제로 참인 값을 말한다. 위의 공식을 보면 FP가 분모에 있는 것을 알 수 있고, 이 FP를 줄이면 정밀도가 올라갈 것이다.

재현율은 실제로 참인 것들 중에 내가 참으로 예측한 값을 말한다. 위의 공식을 보면 FN이 분모에 있는 것을 알 수 있고, 이 FN을 줄이면 재현율이 올라갈 것이다.

두 지표 모두 높은 것이 좋은 것이지만, 정밀도와 재현율은 trade off 관계에 있다. 따라서 상황에 따라 어떤 지표를 더 높일 것인지 연구자가 결정해야 한다.

ex1) 정밀도가 재현율보다 더 중요한 상황: 은행에서 사람에게 돈을 대출해 줄 때 그 사람이 돈을 갚을 것인지(1), 안 갚을 것인지(0)

-> 갚을 것이라고 예측했는데(Positive) 실제로는 갚지 못한(False) 경우가 많다면 FP 값이 클 것이다.

-> 갚지 않을 것이라고 예측했는데(Negative) 실제로 갚은(False) 경우가 많다면 FN 값이 클 것이다.

위 두가지 경우 중 FP의 값을 줄이는 것이 더 risk가 적다. 갚지 않을 것이라고 예측했는데 갚은 것은 금전적인 손해는 발생하지 않으니까. 따라서 다음과 같은 상황에서는 FP값을 줄이는, 즉,
recall보다 precision 값을 더 올리는 게 중요할 것이다.

ex2) 재현율이 정밀도보다 중요한 상황: 불량품(1), 정상품(0) 판별 여부.

-> 불량품이라고 예측 했는데(Positive) 실제로 정상품으로(False) 예측한 경우가 많은 경우. FP. Precision

-> 정상품이라고 예측 했는데(Negative) 실제로 불량품(False)으로 예측한 경우가 많은 경우. FN, Recall

위 두 가지 경우에서는 정상품이라고 내보냈는데 불량품이 생기면 훨씬 리스크가 크다. 따라서 FP보다는 FN의 값을 더 줄여야 하고, 결국 Precision보다 recall이 더 중요하다.

정밀도와 재현율은 이진 분류 뿐만 아니라 다중 분류도 가능하다.

### Threshold(임계치)
분류모델에서 확률값을 분류하는 기준이 되는 수치이다. 보통 분류모델에서는 0.5를 자동으로 임계치로 활용하여 타겟을 분류하지만, 이 임계치를 조정함으로써 클래스를 다르게 분류할 수 있고,
결국 클래스가 다르게 분류되면 precision과 recall 값도 바뀌게 된다. 이진 분류에서는 이 임계치 이상의 값을 1, 작은 값을 0으로 사용한다.

임계치를 높일수록, Positive(1)보다 Negative(0)이 많아진다. Negative가 많아지면 FN 값이 증가하기 때문에 recall 값이 낮아지고, Positive가 줄어들어 FP 값이 감소하기 때문에 trade off에 의해
precision 값이 올라간다.

임계치를 낮출수록, Positive 값이 많아지므로, FP가 많아져 precision이 감소하고, FN에 비해 TP가 많아지기 때문에 trade off 관계에 의해 recall이 증가한다.

상황에 따라 적절한 임계치를 사용하는 것이 좋다.

### F-beta Score
두 분류 모델의 Precision과 Recall 값을 알고 있더라도 어떤 모델이 더 좋은 성능을 내는지 직관적으로 파악하기 어렵다. 따라서 여기서 사용하는 것이 F-beta score이다. 

![image](https://user-images.githubusercontent.com/97672187/156741437-f9293b63-3d98-46ce-b82e-1cf89bf2b9a4.png){: .align-center}

beta는 precision이나 recall 값에 부여되는 가중치로, b = 1이면 precision과 recall에 동일한 가중치를 부여하게 되고, b > 1이면 recall에, b < 1이면 precision에 더 큰 가중치를 부여한다.

우리가 자주 사용하는 F1 score는 beta가 1인 값이고, precision과 recall에 동일한 가중치를 부여해서 ![image](https://user-images.githubusercontent.com/97672187/156741816-4a770f24-bfef-4c2c-9251-256bfc275927.png)
가 된다.

Threshold와 F-beta score는 모두 Precision과 Recall에 영향을 미친다는 공통점이 있지만, threshold는 확률값을 분류하는 기준, F-beta score는 모델의 성능을 평가하는 지표라는 점에서 차이가 있다.

### ROC curve(Receiver Operating Characteristic)와 AUC(Area Under the Curve)
가장 좋은 임계치란, Precision과 recall을 가장 높게 하는 임계치일 것이다. 하지만, 한 번에 이 임계치가 두 지표를 가장 높게 한다는 것을 알기 힘들다. 따라서 ROC curve와 AUC를 사용하여
가장 좋은 임계치를 찾을 수 있다.

ROC에서 사용되는 개념은 2가지이다.

![image](https://user-images.githubusercontent.com/97672187/156743849-f8dee67b-c3a9-4113-ad05-504776103223.png){: .align-center}

이미지 출처: https://ichi.pro/ko/roc-gogseon-mich-auce-daehan-choboja-gaideu-70748239405639

1) TPR은 recall을 의미한다. 즉, 실제로 참인 것 중에 내가 참으로 예측한 값의 비율이다.

2) FPR은 위양성률(양성을 위배하는)이다. recall의 모든 지표를 거꾸로 한 것이고, 실제로 거짓인 것 중에 내가 참이라고 예측한 값의 비율이다.

![image](https://user-images.githubusercontent.com/97672187/156744802-45f486ba-8057-4a87-88f9-2759ca0111d6.png){: .align-center}

이미지 출처: https://losskatsu.github.io/machine-learning/stat-roc-curve/#3-roc-%EC%BB%A4%EB%B8%8C

ROC curve는 이 TPR과 FPR의 관계를 나타내는 곡선으로, TPR이 높고, FPR이 낮을수록 좋은 모델이 된다. 여러 threshold에 따라 각각 다른 tpr, fpr을 가지므로 ROC curve가 가장 좌측 상단에 있게 하는 threshold를 찾으면 해당 모델에서 성능이
가장 우수한 모델이라고 할 수 있다.

AUC는 이 ROC curve의 밑의 면적을 나타낸다. ROC curve는 TPR과 FPR의 관계를 나타낸 curve인데 threshold가 변화함에 따라 TPR이 크게 증가하고, FPR이 조금 증가한다면 이 면적이
넓어지게 될 것이다.(위의 노란색 curve 처럼). 반면에 TPR이 조금 증가하는데, FPR이 크게 증가한다면 곡선이 위로보다는 오른쪽으로 그려져 버릴 것이기 때문에 면적이 점점 줄어든다.
이렇게 AUC는 TPR(민감도)와 FPR(1-특이도)의 증가 관계를 수치(0.5~1) 로 파악할 수 있기 때문에 분류 모델에서 자주 사용된다. 정답이 1인것을(TPR) 정답이 0인데 1이라고(FPR) 잘못 예측한 것보다(FPR) 더 잘 예측할 확률. = 결국 타겟을 잘 구분할 수 있는 확률이 됨.

AUC가 0.5라면 TPR과 FPR의 증가하는 양이 똑같아서 원점에서 정확히 대각선으로 그려지게 될 것이다. 이는 곧 특이도가 1일 때(다 0으로 찍은 것) 민감도는 0이 되고, 민감도가 1일 때(다 1로 찍은 것) 특이도가 0이 되는 것을 의미한다.
AUC가 0.5 = 특이도와 민감도의 합이 항상 1인 관계(TPR = 1, FPR = 1-0(특이도) =1 결국 똑같이 증가. 반대도 마찬가지) = 두 값이 정확하게 trade off 되는 관계.

ex) AUC = 1일 때(두 개의 곡선이 전혀 겹치지 않는다. 두 클래스를 완벽하게 구분할 수 있다.)

![image](https://user-images.githubusercontent.com/97672187/160045360-dd3357d4-8576-4699-884d-1966945a0e59.png){: .align-center}

AUC = 0.7일 때 (두 개의 곡선이 좀 겹쳐있다. threshold를 조정해서 어떤 값을 더 잘 분류하게 할 것인지 정할 수 있다. 두 클래스를 잘 구별할 확률이 70%라는 뜻.)

![image](https://user-images.githubusercontent.com/97672187/160045809-40775103-db74-48b0-916b-16a943abb2f8.png){: .align-center}

AUC = 0.5 일 때(두개의 곡선이 완전 겹쳐있다. 구별할 수 있는 능력이 아예 없다. 위의 예시처럼 민감도1 or 특이도 1로 아예 한 쪽 클래스로만 분류를 해버리니까.)

![image](https://user-images.githubusercontent.com/97672187/160045935-c676e2b5-00e8-4532-847e-2846245151c4.png){: .align-center}

출처: https://bioinformaticsandme.tistory.com/328

따라서 AUC 값이 클수록 더 좋은 모델이 된다. 만약 0.5 미만이면 labeling이나 algorithm이 잘못됐을 가능성이 크다.

```python
from sklearn.metrics import roc_auc_score, roc_curve

pipe = make_pipeline(
  OrdinalEncoder(),
  SimpleImputer(),
  RandomForestClassifier(n_estimators = 100, n_jobs=-1, random_state= 42, max_depth= 8, oob_score=True, min_samples_leaf = 5, min_samples_split= 2)
)

pipe.fit(X_train,y_train)
y_pred_proba_train = pipe.predict_proba(X_train)[:, 1] 
y_pred_proba = pipe.predict_proba(X_val)[:, 1] 
fprs, tprs, th = roc_curve(y_val, y_pred_proba)

# ROC Curve
plt.plot(fprs , tprs, label='ROC')

# 최적의 threshold
# np.argmax는 최댓값의 index를 리턴해준다.
# tpr과 fpr의 차이가 클수록 AUC값이 클 것.
idx = np.argmax(tpr - fpr)
best_threshold = th[idx]

print('threshold:', best_threshold)

#AUC Score
from sklearn.metrics import roc_auc_score
auc_score = roc_auc_score(y_val, y_pred_proba)
auc_score

```

ROC curve는 이진 분류 문제에서 사용할 수 있는데, 다중분류에서도 사용하려면 각 클래스를 이진 분류처럼 변환해서 사용해야 한다. (하나 vs 전체)
Class A,B,C -> Class A vs B,C , Class B vs A,C....

### 백신접종 문제에서의 Precision, Recall
백신 접종(1), 접종X(0)

백신을 맞았는데 안 맞았다고 예측하면, 백신을 맞은 사람이 자기가 맞았다고 제대로 다시 얘기해준다면 큰 문제가 되지 않는다.

하지만, 백신을 안 맞았는데 맞았다고 예측하여, 백신에 관련된 정보를 해당 사람들에게 제공하지 않는다면 전염병에서 정말 중요한 예방접종이 제대로 이루어지지 않기 때문에 문제가 된다고 판단했다.

그러나 맞았는데 안 맞았다고 예측하는 경우가 너무 많으면, 백신을 접종하라고 권유하는 시간이나, 이미 맞은 사람을 접종시키기 위해 가지고 있는 백신의 재고등의 문제도 있기 때문에, precision을 너무 높여서 recall을 떨어뜨리는 것도 좋진 않다. 적절하게 균형을 이뤄야할 것 같다.

따라서 백신을 안 맞았는데 맞았다고 예측한 FP를 낮추는게 이 문제에선 더 중요할 것 같고, 임계치를 조금 낮춰서 recall 보다 precision을 조금 더 올리는 것이 더 적합할 것 같다





