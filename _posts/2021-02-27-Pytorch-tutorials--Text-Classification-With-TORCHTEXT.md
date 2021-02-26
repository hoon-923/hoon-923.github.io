---
layout: post
title:  "Pytorch tutorials- Text Classification With TORCHTEXT"
date:   2021-02-27 01:27:15
author: Hoon
categories: 딥러닝
use_math: true
---

----

### Introduction

이번 튜토리얼에서는 `torchtext`에 포함되어 있는 text classification dataset의 사용 방법을 설명합니다. 이 데이터셋을 다음을 포함합니다.

~~~python
- AG_NEWS,
- SogouNews,
- DBpedia,
- YelpReviewPolarity,
- YelpReviewFull,
- YahooAnswers,
- AmazonReviewPolarity,
- AmazonReviewFull
~~~

이 중 AG_NEWS를 이용해 분류를 위한 지도 학습 알고리즘을 훈련하는 방법을 보여줍니다.

-----

### Load data with ngrams

문장이나 문서를 bag-of-words model로 나타낼 때 일반적으로 unigram이 사용됩니다. 하지만 문서 전체 분류에서는 unigram보다 bigram이 정보력이 더 좋습니다. 밑의 예제 문장을 보면 쉽게 이해가 됩니다.

*어제 경기에서 손흥민 선수의 플레이는 별로 좋지 않았습니다.*

이 문장을 unigram별로 보면

~~~python
tokenize('어제 경기에서 손흥민 선수의 플레이는 좋지 않았습니다')    
# 어제, 경기, 에서, 손흥민, 선수, 의, 플레이, 는, 좋지, 않았습니다
~~~

여기서 unigram 단위로 보면 위 문장에서 '좋지'만 보고 긍정인지 부정인지 알기 어렵습니다. 반면 bigram으로 둘씩 묶어서 보면 '좋지 않았습니다'를 이용해 더 많이 정보를 활용할 수 있습니다. 3개 이상씩 묶어서 보는 방법은 **ngrams**라고 합니다.

`TextClassification` 데이터셋은 ngrams method를 지원해서 ngrams를 직접 지정할 수 있습니다. 튜토리얼에서는 ngrams를 2로 설정함으로서 dataset에 있는 예제 텍스트는 single words에 bigrams string을 더한 list가 됩니다. 

~~~python
import torch
import torchtext
from torchtext.datasets import text_classification
NGRAMS = 2
import os
if not os.path.isdir('./.data'):
    os.mkdir('./.data')
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None)
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
~~~

개인적으로 `train_dataset`의 모양이 궁금해서 다음 코드를 실행해봤습니다.

~~~python
print(train_dataset[0][1])
print(train_dataset[0][1].shape)
~~~

실행 결과:

~~~python
tensor([    572,     564,       2,    2326,   49106,     150,      88,       3,
           1143,      14,      32,      15,      32,      16,  443749,       4,
            572,     499,      17,      10,  741769,       7,  468770,       4,
             52,    7019,    1050,     442,       2,   14341,     673,  141447,
         326092,   55044,    7887,     411,    9870,  628642,      43,      44,
            144,     145,  299709,  443750,   51274,     703,   14312,      23,
        1111134,  741770,  411508,  468771,    3779,   86384,  135944,  371666,
           4052])
torch.Size([57])
~~~

----

### Define the model

튜토리얼에서 정의하는 네트워크는 [EmbeddingBag](https://pytorch.org/docs/stable/nn.html?highlight=embeddingbag#torch.nn.EmbeddingBag) layer와 선형 layer로 구성됩니다. 

파이토치 공식 문서를 보면 `nn.EmbeddingBag`은 embedding bag의 평균값을 계산합니다.  추가적으로 `nn.EmbeddingBag`은 평균을 즉석에서 누적하기 때문에 tensor의 시퀀스를 다룰 때 메모리 효율과 성능을 향상시킬 수 있습니다.

밑의 코드에서 처음에 offset의 역할에 대해 이해를 하지 못해서 관련 파이토치 공식 문서를 찾아봤습니다. 

* 만일 `input`이 (*B*, *N*)으로 2D이면, 각 길이가 고정으로 `N`인 `B`개의 주머니 (sequence)로 취급합니다. 이는 `mode`에 따라 합산된 `B`개의 값을 반환할 것이며, `offsets`는 이 경우에 무시됩니다.
* 만일 `input`이 (*N*)의 1D라면, 여러 주머니(sequences)의 concatenation으로 취급합니다. `offsets`는 1D tensor로, `input`안의 각 주머니가 시작하는 지점의 index를 포함합니다. 따라서, (*B*) 차원의 `offsets`에 대해서, `input`은 `B`개의 주머니를 갖는다고 할 수 있습니다. 빈 주머니(즉, 0짜리 길이)는 0으로 채워진 tensor를 반환합니다.

이를 쉽게 풀어서 설명하면 2D로 들어오면 자연스럽게 batch 단위로 들어오지만, 1D로 들어오면 offset을 기준으로 끊어서 본다는 의미입니다.

네트워크 구조는 다음과 같습니다.

![TorchText_1.PNG](https://github.com/hoon-923/hoon-923.github.io/blob/main/_images/%EB%94%A5%EB%9F%AC%EB%8B%9D/pytorch_tutorials/TorchText_1.PNG?raw=true)

~~~python
import torch.nn as nn
import torch.nn.functional as F
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        # vocab_size는 단어 수, embed_dim은 차원
        # vocab_size=2, embed_dim=5인 경우 [[0, 0, 0, 0, 0]] 크기의 Tensor가 나옴 
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True) # 네트워크에 집어넣으려면 embedding 과정이 필요. 일반적으로 [-1, 1] 사이 값이여야 함
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)
~~~

위의 코드에서 weight 초기화를 [-0.5, 0.5] 사이로 uniformly하게 distribute를 하는 부분이 있는데 원래 잘 안하는 편이지만 튜토리얼 코드여서 있는 것 같습니다.

----

### Initiate an instance

AG_NEWS dataset은 4개 label이 존재합니다.

1. World
2. Sports
3. Business
4. Sci/Tec

~~~python
VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)
~~~

`VOCAB_SIZE`에는 단어와 ngram이 모두 포함됩니다.

---

### Functions used to generate batch

텍스트 원소의 길이가 다를 수 있으므로, 데이터 배치와 오프셋을 생성하기 위한 사용자 함수 `generate_batch()`를 사용합니다. 이 함수는 `torch.utils.data.DataLoader` 의 `collate_fn` 인자로 넘겨줍니다.

~~~python
# 길이가 가변적인 batch를 처리하기 위한 사용자 정의 함수. DataLoader의 collate_fn 인자에 넣어줄 것

def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum은 dim 차원의 요소들의 누적 합계를 반환합니다.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label
~~~

가변적인 batch 사이즈에 대해 `DataLoader`의 `batch_size` 인자를 사용하면 다음과 같은 에러가 발생합니다.

~~~python
RuntimeError: stack expects each tensor to be equal size, but got [1] at entry 0 and [2] at entry 1
~~~

이를 방지하기 위해서 `generate_batch`와 같이 가변적인 batch 사이즈를 처리할 수 있는 사용자 함수를 정의해야 합니다.

----

### Define functions to train the model and evaluate results.

~~~python
from torch.utils.data import DataLoader

def train_func(sub_train_):

    # Train the model
    # 모델을 학습합니다
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch) # collate_fn: 배치로 묶일 경우 indices로 batch를 묶을 때 필요한 함수 정의
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad() # gradient buffer를 0으로
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item() # loss.item()은 loss의 스칼라 값
        loss.backward() # optimizer에 0이 아닌 값이 들어감(dw/dL)
        optimizer.step() # network의 가중치가 갱신됨. w = w - lr * dw/dL
        train_acc += (output.argmax(1) == cls).sum().item()

    # 학습률을 조절합니다
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad(): # test라 gradient 계산이 불필요
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)
~~~

위에서 `text`, `offsets`, `cls`의 형태가 궁금해서 다음과 같은 코드를 추가적으로 실행해봤습니다.

~~~python
data = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=generate_batch)
for i, (text, offsets, cls) in enumerate(data):
  print(f"text: {text}, len: {len(text)}")
  print(f"offsets: {offsets}, length:{len(offsets)}")
  print(f"cls: {cls}, len: {len(cls)}")
  if i == 0:
    break
~~~

실행 결과:

~~~python
text: tensor([    572,     564,       2,  ..., 1194110,  300136,   10278]), len: 1432
offsets: tensor([   0,   57,  140,  219,  298,  383,  478,  571,  668,  843,  904,  991,
        1102, 1163, 1262, 1369]), length:16
cls: tensor([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]), len: 16
~~~

16개의 bags가 `text` 안에 담겨 있고 `offsets` 는 각 bags의 시작점을 알려주고 있습니다. 

----

### Split the dataset and run the model

기존 AG_NEWS dataset에는 validation set이 따로 없기 때문에 데이터를 분할하기 위해 파이토치에서 제공하는`torch.utils.data.dataset.random_split` 을 이용합니다.

또한 learning rate는 초기값을 4.0으로 설정하고 `scheduler`를 이용해서 학습 과정에서 조절했습니다.

~~~python
import time
from torch.utils.data.dataset import random_split
N_EPOCHS = 5
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')
~~~

실행 결과:

~~~python
Epoch: 1  | time in 0 minutes, 8 seconds
	Loss: 0.0262(train)	|	Acc: 84.7%(train)
	Loss: 0.0001(valid)	|	Acc: 90.3%(valid)
Epoch: 2  | time in 0 minutes, 8 seconds
	Loss: 0.0119(train)	|	Acc: 93.7%(train)
	Loss: 0.0001(valid)	|	Acc: 91.5%(valid)
Epoch: 3  | time in 0 minutes, 8 seconds
	Loss: 0.0069(train)	|	Acc: 96.4%(train)
	Loss: 0.0002(valid)	|	Acc: 90.6%(valid)
Epoch: 4  | time in 0 minutes, 8 seconds
	Loss: 0.0039(train)	|	Acc: 98.1%(train)
	Loss: 0.0002(valid)	|	Acc: 91.6%(valid)
Epoch: 5  | time in 0 minutes, 8 seconds
	Loss: 0.0023(train)	|	Acc: 99.0%(train)
	Loss: 0.0003(valid)	|	Acc: 91.3%(valid)
~~~

----

### Evaluate the model with test dataset

~~~python
print('Checking the results of test dataset...')
test_loss, test_acc = test(test_dataset)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')
~~~

실행 결과:

~~~python
Checking the results of test dataset...
	Loss: 0.0003(test)	|	Acc: 90.4%(test)
~~~

----

### Test on a random news

이제 실제 뉴스 데이터를 이용해 네트워크가 과연 맞게 예측을 하는지 알아봅니다. 저는 튜토리얼 예제의 기사가 아닌 외부 기사(Business 라고 예측 하길 원하는)의 일부를 input으로 넣었습니다.

~~~python
import re
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer

ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}

def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item() + 1

ex_text_str = "Foot Locker shares tanked 12.1% in premarket trading after quarterly revenue\
               came in below Street forecasts and comparable-store sales unexpectedly declined.\
               The athletic apparel and footwear retailer also reported quarterly profit of $1.55 per share, beating consensus by 20 cents a share."

vocab = train_dataset.get_vocab()
model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])
~~~

실행 결과:

~~~python
This is a Business news
~~~

----

#### Reference

* pytorch tutorial: [Text Classification With TORCHTEXT](https://tutorials.pytorch.kr/beginner/text_sentiment_ngrams_tutorial.htmll)
* pytorch documents: [nn.EmbeddingBag](https://pytorch.org/docs/stable/nn.html?highlight=embeddingbag#torch.nn.EmbeddingBag)

