---
layout: post
title:  "matplot에서 subplots를 그리는 다양한 방법"
date:   2021-02-09 22:47:45
author: Hoon
categories: Visualization
use_math: true
---

시각화를 하다보면 한번에 두 개 이상의 그래프를 동시에 그리면 좋은 경우가 종종 있는데 대부분의 경우에는`plt.subplots(nrows, ncols)` 를 이용하는 편입니다.

하지만 `matplotlib` 을 이용해서 subplots을 그리는 방법들은 생각보다 다양합니다. 그 중에서도 제가 자주 사용하는 방법인 `subplots`와 `gridspec`, `add_subplot`에 대해 설명 해보겠습니다.

----

### 1. subplots

동일한 크기의 사격형들로 subplots를 그리고 싶으면 `plt.subplots(nrows, ncols)`를 이용하면 됩니다.

~~~python
x = np.linspace(0,1,50)
y1 = np.cos(4*np.pi*x)

fig, axes = plt.subplots(2, 3, figsize=(9,6))

axes[0][2].plot(x,y1)
plt.show()
~~~

![subplots_1.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/Visualization/subplots/subplots_1.PNG?raw=true)

그리고 보니 x축과 y축들이 너무 붙어 있어서 답답한 느낌을 줍니다. 이런 경우 `plt.tight_layout()`을 이용해주면 쉽게 해결이 가능합니다.

~~~python
x = np.linspace(0,1,50)
y1 = np.cos(4*np.pi*x)

fig, axes = plt.subplots(2, 3, figsize=(9, 6))

axes[0][0].plot(x,y1)
plt.tight_layout()
plt.show()
~~~

![subplots_2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/Visualization/subplots/subplots_2.PNG?raw=true)

`plt.subplots()`는 위와 같이 동일한 크기의 사각형을 그릴 때 편리합니다.

----

### 2. gridspec, add_subplot

모든 subplot들의 사이즈를 동일하게 그리면 다양한 종류의 시각화를 동시에 표현할 때 불편합니다. 예를 들어 추세를 보여주는 lineplot과 piechart를 똑같은 크기의 subplot을 이용해 시각화를 하면 매우 어색한 모양의 그림이 나올 것 같습니다.

`gridspec` 과 `add_subplot` 을 이용하면 마치 list에 인덱싱을 하는 것과 같은 원리로 접근을 하면 되기 때문에 이해하기 쉽습니다.

~~~python
fig = plt.figure(figsize=(9, 6))

gs = fig.add_gridspec(2, 3)

ax = [None for _ in range(5)]

ax[0] = fig.add_subplot(gs[0, :]) 
ax[0].set_title('gs[0, :]')

ax[1] = fig.add_subplot(gs[1, 0])
ax[1].set_title('gs[1, 0]')

ax[2] = fig.add_subplot(gs[1, 1])
ax[2].set_title('gs[1, 1]')

ax[3] = fig.add_subplot(gs[1, -1])
ax[3].set_title('gs[1, -1]')

for ix in range(4):
    ax[ix].set_xticks([])
    ax[ix].set_yticks([])

plt.tight_layout()
plt.show()
~~~

![gridspec_1.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/Visualization/subplots/gridspec_1.PNG?raw=true)

다음은 `gridspec`과 `add_subplots`를 이용해 하나의 jointplot이 아닌 두 개의 jointplot을 동시에 시각화 한 예시 입니다.

~~~python
fig = plt.figure(figsize=(12,7))

widths = [4, 4, 1]
heights = [1, 4]

spec = fig.add_gridspec(ncols=3, nrows=2, width_ratios=widths, height_ratios=heights)

axs = {}
for i in range(len(heights)*len(widths)):
    axs[i] = fig.add_subplot(spec[i//len(widths), i%len(widths)])


sns.scatterplot("T", "TARGET", data=data, hue="month",
                palette=['dodgerblue','salmon'],ax=axs[3], alpha=0.6)
for i, s in enumerate(months):
  sns.regplot("T", "TARGET", data=data.loc[data["month"]==s], 
                scatter=False, ax=axs[3])

sns.scatterplot("RH", "TARGET", data=data, hue="month",
                palette=['dodgerblue','salmon'],ax=axs[4], alpha=0.6)
for i, s in enumerate(months):
  sns.regplot("RH", "TARGET", data=data.loc[data["month"]==s], 
                scatter=False, ax=axs[4])


sns.kdeplot("T", data=data, hue="month", ax=axs[0],
            palette=['dodgerblue','salmon'], legend=False, fill=True, zorder=1)
axs[0].set_xlim(axs[3].get_xlim())
axs[0].set_xlabel('')
axs[0].set_xticklabels([])
axs[0].spines["left"].set_visible(False)
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)


sns.kdeplot(y="TARGET", data=data, hue="month", ax=axs[5], 
            palette=['dodgerblue','salmon'], legend=False, fill=True, zorder=1)
axs[5].set_ylim(axs[3].get_ylim())
axs[5].set_ylabel('')
axs[5].set_yticklabels([])
axs[5].spines["bottom"].set_visible(False)
axs[5].spines["top"].set_visible(False)
axs[5].spines["right"].set_visible(False)

axs[2].axis("off")

sns.kdeplot("RH", data=data, hue="month", ax=axs[1],
            palette=['dodgerblue','salmon'],legend=False, fill=True, zorder=1)
axs[1].set_xlim(axs[4].get_xlim())
axs[1].set_xlabel('')
axs[1].set_xticklabels([])
axs[1].spines["left"].set_visible(False)
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)

hist_range_max = max(axs[0].get_ylim()[-1], axs[1].get_ylim()[-1], axs[5].get_xlim()[-1])
for i in range(len(widths)-1):
    axs[i].set_ylim(0, hist_range_max)
axs[5].set_xlim(0, hist_range_max)

axs[1].set_yticklabels([])
axs[1].set_ylabel('')
axs[4].set_yticklabels([])
axs[4].set_ylabel('')

for i in range(len(heights)*len(widths)):
    axs[i].grid("on", color="lightgray", zorder=0)
fig.tight_layout()
plt.show()
~~~

![gridspec_2.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/Visualization/subplots/gridspec_2.PNG?raw=true)

이와 같이 subplots의 크기를 다르게 조정할 수 있다는 점을 다양하게 활용할 수 있습니다.

-----

### References

* Pega Devlog: [Seaborn with Matplotlib (2)](https://jehyunlee.github.io/2020/10/03/Python-DS-35-seaborn_matplotlib2/)