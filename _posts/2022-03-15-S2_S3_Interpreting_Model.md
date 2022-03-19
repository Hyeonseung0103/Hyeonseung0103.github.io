---
layout: single
title: "Section2 Sprint3 Note 234 PDP, SHAP, Interpreting ML Model"
category: Section2
---

선형 회귀에서는 회귀 계수를 통해 변수가 타겟에 영향을 미치는 정도를 파악할 수 있었다. 하지만, 좋은 성능을 내기 위해 단순한 선형 회귀가 아닌 복잡한 모델을 사용하면 모델을 해석하기가 어려워진다.
즉, 어떤 변수가 타겟에 얼만큼 영향을 미치는지 설명하기가 어려워진다는 뜻이다. 이 문제는 PDP, SHAP 개념을 통해 해소할 수 있다.

## Note 234
### Model Dependence vs Model Agnostic
트리 모델에서 제공하는 feature_importances는 트리 모델에서만 사용할 수 있다. 또한, 선형 회귀의 회귀계수는 선형 회귀의 경우에만 해석이 가능하다.
이처럼 모델을 해석할 때 특정 모델에만 국한되어서 사용할 수 있는 해석 방법을 Model Dependence 하다고 한다.

반면, 지난 시간에 배운 permutation importance, drop-column importance는 모델에 상관없이 모두 사용할 수 있는 방법이다. 어떤 모델이건 저 방법을 통해 변수의 중요도를 파악할 수 있게 된다.
이렇게 모델을 해석할 때 모델에 상관없이 사용될 수 있는 해석 방법을 Model Agnostic 하다고 한다.

### Partial Dependence Plots(PDP)
모델의 복잡도와 해석은 trade off 관계에 있다. 복잡한 모델일수록 모델을 해석하기가 힘들어진다. 하지만, Partial Dependence Plot을 사용하면, 모델이 복잡하더라고 변수와 타겟과의 관계를
파악할 수 있으므로 모델을 해석할 수 있게 된다.

PDP는 모델이 예측한 타겟값을 0 또는 지정한 값으로 잡아놓고, 해당 피처의 값이 변화할 때 예측 타겟값(실제 타겟값이 아니고)이 어떻게 변화는 가를 확인할 수 있는 그래프이다. 

1) 랜덤하게 관측치를 선택

2) 그 관측치에 대해서 특정 변수 외에 모든 변수의 값은 고정시키고, 특정 변수의 값은 랜덤한 값으로 바꿈(아예 랜덤은 아니고 정해진 grid 내에서의 값임)

3) 특정 변수의 값이 바꼈을 때 모델이 예측한 새로운 타겟값이 도출됨

4) 이 관측치에서 또 다른 랜덤한 값 부여

5) 새로운 타겟값이 도출됨

6) 위의 과정을 반복하여 모든 관측치를 사용해서 특정 변수의 따른 타겟의 변화 곡선들(ICE curve)을 평균냄

7) 최종 PDP 생성

ICE는 랜덤한 관측치에서 예측한 타겟값의 분포고, PDP는 모든 ICE curve를 평균낸 것. 결국 모든 관측치를 사용하게 된다.

![image](https://user-images.githubusercontent.com/97672187/158373189-c31ee196-184a-478e-b887-0f8f91c5f666.png)

위의 과정을 보면 한 관측치에 대해 랜덤한 값이 들어감으로써 예측되는 타겟값이 바뀌게 될텐데 이 값들을 선으로 연결한 것이 하나의 ICE(Individual Conditional Expectation) curve이다.
다른 관측치에서는 또 다른 ICE curve가 생길 것이고, PDP는 이 모든 ICE curve를 평균 낸 것이다.

PDP는 복잡한 모델을 해석할 수 있게 하고, 기존 데이터에 없었던 모르는 데이터도 랜덤한 값으로 사용하여 외삽을 할 수 있다는 장점이 있다. 하지만, 데이터의 분포가 적은 구간에서 도출된 
예측 타겟값과의 상관관계는 신뢰할 수 어렵고, 독립 변수들끼리 독립성을 띄어야 한다는 가정이 단점이다.

독립성을 띄어야 다른 관측치들이 그대로 있고 특정 변수가 바껴도 다른 변수들이랑 상관없이 값의 예측이 가능할 테니까. 하지만 이렇게 데이터끼리 독립인 경우는 거의 없다. 
예를 들어 키와 몸무게가 독립변수, 사람의 보폭 크기를 종속변수라고 할 때 키를 특정변수로 하고 PDP를 그린다고 하자. 그럼 몸무게를 그대로 두고 키에 랜덤한 값을 넣을텐데. 
랜덤하게 선택된 관측치의 몸무게가 50kg 인데 키를 랜덤하게 2m를 넣으면 현실적으로 말이 안 되는 값이다. 따라서 독립성 가정 때문에 현실적이지 못한 데이터를 계속 만들어낸다면 오히려 
신뢰성이 떨어진다. 이를 해소하기 위해 조건부로 랜덤하게 값을 부여하는 Acculated Local Effect Plots도 있다고 한다.

또한, 최대 2개의 변수와 예측 타겟값과의 관계를 나타낼 수 있다. 이때는 2개의 변수의 상관성이 예측값에 영향을 미치는 정도를 파악할 수 있게 된다.

```python
#1. 변수 하나로만 PDP 그리기
from pdpbox.pdp import pdp_isolate, pdp_plot

feature = 'CRIM'

isolated = pdp_isolate(
    model = model,
    dataset = X_test,
    model_features = X_test.columns,
    feature = feature,
    grid_type = 'percentile',
    num_grid_points=10 # numeric 변수에서 랜덤하게 넣을 값의 갯수?(10개의 구간의 값을 랜덤하게 부여한다.)
)
pdp_plot(isolated, feature_name=feature , plot_pts_dist = True);
#plt.xlim((20000,150000)); #값의 범위를 제한해서 특정 구간을 보고 싶을 때.
# plot_pts_dist = 젤 밑에 데이터의 분포 표시(이상치나 노이즈를 확인, 데이터 불균형 확인) 이를 통해 이 값을 신뢰할지 말지 결정할 수도 있다.
```

![image](https://user-images.githubusercontent.com/97672187/158375996-03277824-7a07-4a4e-b9ea-b40c2f939e91.png)


```python
#2. 두 변수를 사용하여 PDP 그리기
from pdpbox.pdp import pdp_interact, pdp_interact_plot

features = ['CHAS', 'DIS']

interaction = pdp_interact(
    model = model,
    dataset = X_test,
    model_features = X_test.columns,
    features = features
)
pdp_interact_plot(interaction, plot_type = 'grid',
                  feature_names = features)
plt.xticks([1, 2], ['with Charse River', 'without Charse River',]); # 범주형 변수에서 encoding된 값을 원래대로 표현하기 위해서 xticks로 직접 바꿈.

```

![image](https://user-images.githubusercontent.com/97672187/158376106-5746df0c-0322-47c7-804b-00ff3ccf58d9.png)

히트맵 안에 값이 클수록 타겟에 영향을 더 많이 미치는 것.


### SHAP
SHAP는 각 관측치에서 변수들이 타겟에 얼마나 영향을 주는가를 볼 수 있다. 선형 회귀 모델이 아니라 복잡한 모델이라면 변수들이 타겟에 영향을 미치는 정도를 알기가 힘든데, SHAP value를 사용하면
각 관측치에 마다 변수들이 타겟에 영향을 주는 정도를 파악할 수 있다. 변수의 영향도를 본다는 점에서 PDP와 유사한 것 같지만, 랜덤한 관측치를 통해 변화되는 모델의 전체(Global) 예측 타겟값을 
확인하고 기존에 없던 데이터도 랜덤한 값으로 사용하는 PDP와는 달리 SHAP는 모델 전체가 아니라 특정 관측치에서(Local) 예측한 타겟값에 독립변수들이 얼마나 영향을 미치는지 확인하고, 실제로 있는
데이터만을 사용한다는 점에서 차이가 있다. 각 관측치마다 예측값이 다르고 독립변수가 기여하는 정도가 다르기 때문에 다양한 Shap value가 계산 되고 이 계산된 Shap value들의 분포를 통해
모델의 예측 타겟값에 얼마나 영향을 미치는지, 어느 방향으로 영향을 미치는지를 확인할 수가 있다.

PDP처럼 모델에 상관없이 변수의 중요도를 파악하고 타겟과의 관계를 해석할 수 있다는 장점이 있지만, 역시나 독립성이 가정되어야 하고, PDP처럼 Global이 아닌 Local의 관계를 나타내기 때문에
사용한 관측치가 너무 작다면 이 관계를 신뢰할 수 없다는 단점이 있다.

```python
# force plot
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test.iloc[:300])

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, X_test.iloc[:300])
```
![image](https://user-images.githubusercontent.com/97672187/158379972-eae93691-3250-4baa-86a2-dc36416f2450.png)

```python
shap.summary_plot(shap_values, X_test.iloc[:300])
#빨간색 점이 오른쪽에 있어야 양의 상관관계. 파란색 점이 오른쪽이면 음의 상관관계.
#하지만, 점들이 오른쪽, 왼쪽에 골고루 섞여있으면 상관관계가 없는 것.

```

![image](https://user-images.githubusercontent.com/97672187/158380101-56805a02-5971-4dd1-87c5-c6cd7efa2885.png)


```python
shap.summary_plot(shap_values, X_train.iloc[:300], plot_type='bar') #bar는 shap value를 평균내서 기여도를 파악하는 것.
```

![image](https://user-images.githubusercontent.com/97672187/158380251-84220490-64ac-4a1d-b45d-726382a135da.png)

```python
shap.summary_plot(shap_values, X_test.iloc[:300], plot_type="violin")

```

![image](https://user-images.githubusercontent.com/97672187/158380289-ab955491-fa16-445a-b8ef-060e3deef1bf.png)



### 모델을 해석하는 방법들

서로 관련이 있는 모든 특성들에 대한 전역적인(Global) 설명

-Feature Importances

-Drop-Column Importances

-Permutaton Importances

타겟과 관련이 있는 개별 특성들에 대한 전역적인 설명

- Partial Dependence plots

개별 관측치에 대한 지역적인(local) 설명

- Shapley Values

