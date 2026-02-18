# 01
### 머신러닝과 딥러닝
머신러닝 : 데이터로부터 패턴 파악, 학습
- 알고리즘이 학습과정에서의 피드백 받는 방식
	- supervised
		- data + labels
		- overfitting 가능성
		- regression / classification
	- semisupervised / selfsupervised
		- pseudo-labeled data
	- unsupervised
		- unlabeled data
	- reinforcement
		- agent의 action -> state + reward ( reward system )
		- trial and error
딥러닝 : 하나 이상의 hidden layer를 갖고 있는 neural network
- neural network : 행렬의 모임 ( feature들의 weight 만큼의 정보가 담겨있는 유의미한 행렬임 )
	- 이 neural network를 학습한다는 것 : 각 node를 연결하는 weight의 값을 얼마로 할당해줄까? 에 대한 것들을 찾아가는 과정임
		- loss -> backpropagation으로 계속 업데이트해나가며 찾아감
- 단일 neural network -> 이건 아무리 딥하게 쌓아도 그냥 linear한 함수가 되는거임 -> non linear fuction이 꼬옥 필요한 이유. 비선형성이 추가되어야함 ( xor 문제 생각해보기! )
- 일반적인 neural network ( 비선형성이 추가됨 )
	- fully connected networkt : 앞 모든 뉴런들 - 다음 뉴런 모두 연결
- 그럼 prediction에서는?
	- 우리가 하자고 했던 task에 맞추어 결과를 뽑아낼 수 있어야 함 - 확률 분포를 뽑아내야 함.
		- softmax layer
	- prediction을 통해 loss를 구하고, backpropagation 수행

# 02
### 자연어 처리 모델
tokenization
- vocab size
	- too big : 너무 많은 계산량
	- too small : 너무 많은 단어가 unk token이 되는것
word embedding
- bag of words : 고냥 제일 단순한 단어 매칭임 (one hot vector같은 고)
- word2vec : 특정 단어A의 임베딩을 위해 주변 단어들을 이용해서 A를 예측하도록 하는 pseudo task를 정의하여 학습시킨 임베딩 방식
	- cbow : 주변 단어들을 통해 특정 단어를 예측하는 방식으로 vector값 업데이트
	- skip gram : 특정 단어를 통해 주변 단어를 예측하는 방식으로 vector값 업데이트
	- 위와 같은 방식들의 최종(마지막) hidden layer가 각 단어의 word vector묶음마냥 표현이 되는 것 -> word embedding vector
language model ( lstm 까지..! )
language model의 학습 단어 시퀀스에서 다음 단어의 확률 분포를 뽑는 과정
- fixed language model : windows가 정해진(문맥의 단어 개수가 정해진 상황) 상황이 가정된 모델 -> 전체 문맥을 봐야하는데 이러면 어떡함?
- rnn : sequential data에 너무나도 자연스러운 형태임 이전 내용 계속 누적. 하나의 weight를 계속 공유하기 때문에 이러한 것이 가능함. 그러나 computation측면에서 많은 단점.
	- 매 step(시퀀스에서 단어를 하나하나 가져올 때마다)마다 loss 계산, gradient 계산 하는 과정
		- 이 때 이전 결과를 다음 입력으로 넣을수도,
		- 정답 자체를 다음 입력으로 넣을수도 있음.
- lstm rnn : rnn에 gate, cell추가됨
	- forget gate : 얼마나 잊을것인가? -> 이전 정보를 얼마 만큼만 보존하고 반영할 것인지를 -> 이전 정보의 중요도 결정
	- input gate : 얼마나 입력할것인가? -> 새로운 내용을 얼마만큼 반영할 것인지를 -> 새 값의 가중치 결정
	- output gate : cell state에서 얼마나 출력할것인가? ( hidden sate 결정 )
	- cell state : 장기 기억 저장 벡터 값
	- `C_t = (forget gate × C_{t-1}) + (input gate × C̃_t)` : forget gate, input gate는 각각 순차적인 계산이 아니라 두 gate의 결과를 병렬로 계산 후 더하는 것

# 03
### transformer와 attention
기존 seq2seq (encoder-decoder) 구조 모델
- encoder rnn : input 문장 요약 / 이해
- decoder rnn : encoder layer를 거친 마지막 hidden state를 다시 decoder rnn 입력으로 넣어, 순차으로 단어 예측
- rnn이 가지는 기존 단점이 여전히 존재. encoder의 last hidden state만을 가지고 decoder의 입력으로 들어가기 때문에, encoder에서의 최근 정보만 강하게 기억, 초반 부분의 내용은 많이 잊게 됨
	- 그럼 초기~최근 으로 갈수록 더 많이 기억하게 되는 구조를 사용하지 말아보자
	- 중요한 부분에만 집중하도록 해보는 건? ->attention
		- encoder의 last hidden state를 쓰는 것이 아니라, context vector라는 것을 만들어서 사용해보기
		- context vector : encoder의 hidden state -> attention weight 계산 -> weighted sum - normalization (softmax)
		- attention weight
			- encoder의 hidden state와 decoder의 hidden state의 연관성을 보자
			- 여기서의 연관성 => dot product를 계산 후 socre 뽑아내기

rnn <-> transformer
- rnn의 기존 문제점 : 한 번에 많은 sequence를 처리하기 힘들다, 병렬 처리가 불가능한 구조, 장기 의존성 문제
- transformer : rnn과는 완전히 새롭고 다른 구조 -> attention mechanism만 이용해서 만들어진 구조, 병렬 처리 가능, 애초에 모든 단어들이 fully connected하게 연결되어 처리됨

self attention
- 스스로 각 문장 안에서 각 단어끼리의 관계성(attention weight) 계산
- 계속 단어간의 표현(representation)의 상호작용을 일으키게 됨
- query key value
	- 단어의 representation을 projection 하여 query key value를 만드는 것에서 부터 시작함
		- 기존 단어들에서부터 그대로 사영하여 query 하나를 만들어두고
		- 기존 단어들에서부터 그대로 사영하여 key 하나를 만들어두고
		- 기존 단어들에서부터 그대로 사영하여 value 하나를 만들어둠
		- 결국 이 query key value는 하나의 입력에서부터 동일하게 나온 것들임 -> "self" attention
	- query와 key의 dot product 계산 -> 확률 형태로 normalization : 문장 속 각 단어들간의 관계 (중요성)을 뽑아냄
	- 앞에서 뽑아낸 중요성을 value vector와 곱함(weighted sum) ->  단어의 새로운 representation을 만들기 => "self attention"

positional embedding
rnn은 시퀀스를 처리할 때 단어 하나하나를 순차적으로 처리 -> 시간의 순서가 반영됨
transformer는 병렬 처리되기 때문에 순차적(시간적)인 개념이 반영되지 않음. -> positional embedding 사용

multi head self attention
self attention을 여러번 나눠서 계산하는 것임
- 다양한 관점에서의 weight를 계산하여 그 결과를 합쳐나간다고 이해하면 됨 마지막에 concat해서 합침
- 실제 실험에서는 8개의 multi head를 사용함
- 이것 또한 동일하게 병렬 처리가 가능
- 각 head 별로 단어별 중요도가 다르게 나옴.
- 그 "다양한" 관점의 기준은 누가 정하는가? : 누가 정하는 것이 아님. 학습 과정에서 자동으로 정해지는 것. 
	- 각 헤드가 독립적인 가중치 행렬을 가지는데, 각 해드가 다른 패턴을 포착하도록 자동ㅡ로 학습됨. 우리는 이해하지 못하겠지만 attention layer의 신경망이 어떠한 "패턴"을 찾을것임.
	- 근데 8개의 헤드가 각각 다른 패턴을 찾도록 하는것

encoder
multi head self attention + feed forward neural network의 반복 : input문장에 대한 representation 업데이트
decoder
encoder에서 생성된 representation -> conditional 하게 문장 생성
encoder에서는 key와 value만 넘어오며,  query는 decoder에서 계속 업데이트해나가는 방식임.
- inference : 순차적으로 다음 단어 하나씩 예측하며 처리
	1. 지금까지 생성한 단어들"까지"만 보고 (masked self-attention)
	2. 다음 단어로 뭘 만들지를(query) 원문(encoder에서 생성된 representation) 참고하여
	3. 새 단어를 생성함(feed forward)
	- greedy search : 매번 가장 높은 확률의 다음 단어를 선택하지만, 이는 최적의 선택이 아닐 수 있음. 
	- beam search : 매번 가장 높은 확률의 다음 단어를 선택하는 것이 아님. 여러 후보를 유함


# 04
### 사전학습
word embedding의 pretraining


# 05
llm agent : 어떠한 특정환경과 상호작용할 수 있는 llm. 스스로 다음 절차를 생각해낼 수 있는 자율성과 반응성이라는 핵심 특징 존재.
short-term memory : 현재 대화 세션 내에서만 유지되는 즉각적인 문맥 정보로 대화의 최근 몇 턴을 기억하며 주로 컨텍스트 윈도우에 저장
long-term memory : 긴 대화 내용들을 저장해두고 
toolformer :  
MCP : 
Reasoning : 
planning : 
ReAct : reason and act planning을 하면서 계속 답을 내는 형태
multi agent collaboration : supervised agent의 역할을 중심으로 ... 
# 06
AI 하드웨어와 GPU
