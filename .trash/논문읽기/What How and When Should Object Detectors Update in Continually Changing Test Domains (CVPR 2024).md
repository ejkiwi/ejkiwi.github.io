20250305
 What How and When Should Object Detectors Update in Continually Changing Test Domains? (CVPR 2024)
 `#객체탐지 #TTA #DomainShift`  
`
### 이것무엇이지?
- 테스트 시 데이터 분포 변화 (Distribution Shift) -> 훈련 데이터와 테스트 데이터의 특성이 다를 때 발생하는 문제
- TTA -> test time adaptation 모델이 테스트 중 새롭게 들어오는 데이터에 맞게 스스로 적용하는 방법.
- 업데이트 시점을 판단 ? -> 가중치 업데이트를 뜻함. 논문에서는 모델이 충분히 적응한 상태라면 불필요한 업데이트를 건너뛰는 전략 사용,.
- 벤치마크 -> '벤치마크 데이터셋'이라고 이해해해보자. 표준 데이터셋과 테스트 환경 의미. ( 여기에서는 객체 탐지 모델의 대표적 벤치마크 데이터셋으로 COCO 데이터셋이 등장. 뭐... MNIST 머 그런거임... )
- mAP -> 평균 정밀도.. 모델의 평가 지표라 한다. 일단은 이렇ㄱㅔ 이해해보기로......~_~ 보통 50퍼면 절반정도 탐지햇다는 뜻임.
- data representation 학습 ->
- BN 레이어 기반 방식 -> 딥러닝 모델의 학습 과정을 개선하기 위한 정규화 기법. 각 미니배치 내의 데이터에 대해 평균과 분산을 계산해서 레이어에 입력되는 값들을 정규화함. ( 미니치? 데이터 한 번에 우다다 못 씀. 그래서 여러개의 묶음으로 나눠 사용. 50000개의 데이터가 있는데 미니 배치 크기가 128이면 총 50000/128 만큼의 에포크가 돌아가게 됨. )
- Lightweight Adaptor ->
- Feature Alignment ->
- Adaptive Update Strategy ->