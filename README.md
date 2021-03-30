# Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
 논문 <https://arxiv.org/abs/1707.01926>
 ## 시공간 Dependency를 잡기위해
  ### 공간 - Diffusion Convolution
   공간적 ->  bidirectional random walks on the graph
   
   GCN의 개념 들어감
    - 가중치 매트릭스를 이용하여(곱하여) 그래프를 CONVOLUTION 함
    - 
   
  ### 시간 - RNN
   시간 -> the encoder-decoder architecture with scheduled sampling
 
 ## 도로 네트워크의 트래픽 예측
 - 과거 교통 속도와 기본 도로망이 주어진 센서 네트워크의 미래 교통 속도를 예측하는 것이다

 # 어려움
 - 복잡한 시공간의존성
 - 장기적 예측의 어려움,  Recurring incidents -> 러시아워, 사고 가 비일관성을 일으킨다 


# 임시 readme
## 헷갈리는 용어들 
 - HORIZION - 5분단위로 레이어를 나눈다? 라는 뜻으로 해석함
강남구 데이터로 훈련된 dcrnn 모델

dcrnn 조금 수정했습니다

start.py로 시작하고 fine_name에 파일 이름 넣어주심 됩니다

make_predict로 예측합니다

사용된 데이터는 강남구 거리 속도 데이터입니다

topis에서 가져와 전처리 후 사용하였습니다

서울 데이터 모두 전처리 하였으니

정리하여 전처리 코드, 전처리된 파일들 올릴 예정입니다

# 데이터

모델 트레이닝 위해서는
- distances_{}.csv
- graph_sensor_ids_{}.txt
- graph_sensor_locations_{}.csv 
- {}.h5
필요합니다 {}에는 같은 파일 이름 넣어주세요
형식은 데이터 들어가보시면 됩니다


# 새로운 모델 훈련
python start.py --file_name="사용하는 파일"

 위 명령어 입력시

 -데이터 준비
 
 -그래프 구성
 
 -모델 트레이닝
 
 -베이스라인 평가
 
한번에 실행되도록 수정했습니다

# 예측하기
python make_predict.py --config_filename=data/model/트레이닝시킨폴더/config_100.yaml

데이터 준비할때 남겨둔 20%의 데이터로 12시간 후 까지 예측합니다
코드 다시 수정하여 더 나중 시간까지 예측하도록(데이터 test셋 제외) 수정할 예정입니다


# REF
https://github.com/liyaguang/DCRNN
여기서 코드 가져와서 수정하였습니다
추 후 모델 개선시킬 예정입니다
