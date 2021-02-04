---
layout: post
title:  "Visualization에서 color의 중요성"
date:   2021-02-04 22:25:45
author: Hoon
categories: Visualization
use_math: true
---

성공적인 시각화 사례들을 보면 시각화의 의도에 맞게 색깔 조합이 이루어진 것을 볼 수 있습니다. 의도에 맞게 잘 사용한 색깔 조합은 시각화를 통해 청중들에게 표현하고자 하는 정보를 더욱 쉽게 전달할 수 있다고 생각합니다. 반면 필요 이상으로 너무 많은 색깔을 사용하거나, 상황에 맞지 않는 색깔 조합을 사용하면 오히려 전달력이 떨어지는 경우가 있습니다.

이처럼 시각화에서 생각보다 색깔이 중요한 요소라고 생각되어 본 포스트를 작성 했습니다.

-----

### 1. color palette의 종류

시각화에서 주로 쓰이는 주요 palette들은 다음과 같습니다.

* *Qualitative palettes*
* *Sequential palettes*
* *Diverging palettes*

상황에 따라 적절한 palette를 쓰면 시각화가 더욱 돋보일 수 있다고 생각합니다.

#### 1-1. Qualitative palettes

Qualitative palettes는 시각화 하고자 하는 변수가 카테고리일 때 사용합니다.

![palplot.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/Visualization/Visualization_color/palplot.PNG?raw=true)

Qualitative palettes 에서 각 색깔들은 하나의 그룹에 지정되기 때문에 너무 많으면 혼동을 줄 수 있다. 가급적 10개 이하의 색으로 구성하는 것이 좋다.

![lineplot.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/Visualization/Visualization_color/lineplot.PNG?raw=true)

pie chart의 경우에도 너무 조각이 작아지면서 많아지면 그 조각들을 others 로 묶어서 그려주는 것이 훨씬 시각적으로 깔끔합니다.

![piechart.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/Visualization/Visualization_color/piechart.PNG?raw=true)

위의 좌측 pie chart를 보면 색깔들이 너무 많아지고 심지어 색이 겹쳐서 그려집니다. 이는 좋은 현상이 아니기 때문에 위와 같이 pie chart의 조각들이 너무 많아 지는 경우 하위 몇 조각들을 others로 묶은 후 시각화 해주는 것이 바람직합니다.

이처럼 Qualitative palettes 에서는 색깔 자체로 카테고리들을 구분해주는 것이 일반적이지만 상황에 따라 밝기를 이용해 구분할 수도 있습니다.

![moveing_average_lineplot.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/Visualization/Visualization_color/moveing_average_lineplot.PNG?raw=true)

위의 그래프를 보면 5월동안 일일 기온 lineplot과 1주일을 기준으로 한 이동평균선 lineplot을 동시에 그렸습니다. 이렇듯 서로 연관이 있는 두 항목을 동시에 시각화할 때는 색깔의 차이가 아닌 밝기의 차이를 이용해 그려주면 더욱 효과적인 시각화입니다. 여기서 주의해야할 점은 둘의 밝기가 너무 많이 차이가 나면 안됩니다. 자칫 잘못하면 진한 색이 더욱 중요하다라는 인식을 줄 수 있기 때문입니다.

#### 1-2. Sequential palettes vs Diverging palettes

Sequential palettes 와 Diverging palettes 는 서로 연관이 있기 때문에 같이 설명하는게 좋을거라 판단 했습니다. 둘의 사용법에 대해 언급하기 전에 우선 각각 어떤 palettes 인지 설명 하고 넘어가겠습니다.

Sequential palettes 는 하나의 main 색깔을 바탕으로 아주 밝은 색깔부터 진한 색깔까지를 표현하는 palettes 입니다.

![sequential_palettes.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/Visualization/Visualization_color/sequential_palettes.PNG?raw=true)

Diverging palettes 는 2개 이상(레인보우처럼 다양한 색깔도 가능합니다)의 색깔을 이용해 양쪽 끝은 그 색깔들이 매우 진하고 중간은 하얀색에 가까울정도로 밝기가 매우 약한 palettes 입니다.

![diverging_palette.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/Visualization/Visualization_color/diverging_palette.PNG?raw=true)

두 palettes 모두 변수가 추세가 있는 수치형이거나 명목형 변수여도 순서가 있는 변수인 경우 사용한다는 공통점이 있습니다. 

하지만 Diverging palettes 의 경우 0을 중심으로 상대적으로 비교되는 경우에 쓰는 것이 좋고, Sequential palettes 는 상대되는 개념들의 표현이 아닌 자체적으로 대소 비교시에 사용하는 것이 바람직합니다.

예를 들면 Diverging palettes 는 다음과 같이 heatmap 을 시각화 할 때 사용하는 것이 적절합니다. 

![heatmap.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/Visualization/Visualization_color/heatmap.PNG?raw=true)



Sequential palettes 는 다음과 같이 하나의 변수내에서 대소 비교를 할 때 유용합니다.

![map.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/Visualization/Visualization_color/map.PNG?raw=true)

----

### 2. Discrete palettes vs Continous palettes

 Sequential palettes 와 Diverging palettes 는 두 가지 방법으로 표현이 가능합니다. 



>  **Discrete palettes 의 장점**

* Continous palettes 에 비해서 색을 구분하기가 용이
* 시각화 하는 데이터를 더 잘 표현하도록 범위를 나눌 수 있다
  - Continous palettes는 outlier 들이 많을 시 좁은 범위에 데이터들이 몰릴 가능성



> **Continous palettes 의 장점**

* 데이터의 exact 한 값을 색깔로 표현이 가능하다.
  - Discrete palettes 는 서로 다른 값들이 같은 범위에 존재한다는 문제점 발생



Discrete palettes 와 Continous palettes 중 시각화의 목적에 맞게 사용하는 것이 

----

### 3. Tools for choosing colors

구글링을 통해 색깔들을 찾아봐도 좋지만 이런 수고를 덜어줄 수 있는 다양한 사이트들이 존재합니다.

#### [Data Color Picker](https://learnui.design/tools/data-color-picker.html)

![color_picker.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/Visualization/Visualization_color/color_picker.PNG?raw=true)

위 사이트는 Sequential palettes 와 Diverging palattes 를 생성할 때 유용합니다.

#### [Color Thief](https://lokeshdhakar.com/projects/color-thief/)

![color_thief.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/Visualization/Visualization_color/color_thief.PNG?raw=true)

위 사이트는 특정 사진을 입력하면 사진의 색감들을 바탕으로 Qualitative palettes 을 추천해주는 재미있는 사이트 입니다. 

#### [Viz Palette](https://projects.susielu.com/viz-palette)

![VIZ_palette.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/Visualization/Visualization_color/VIZ_palette.PNG?raw=true)

위 사이트는 최종적인 시각화 결과 어떻게 보일지에 대해 판단할 때 유용한 사이트 입니다.

-----

#### References

* medium: [How to Choose Colors for Your Data Visualizations](https://medium.com/nightingale/how-to-choose-the-colors-for-your-data-visualizations-50b2557fa335)
* everyday analytics: [When to Use Sequential and Diverging Palettes](https://everydayanalytics.ca/2017/03/when-to-use-sequential-and-diverging-palettes.html)