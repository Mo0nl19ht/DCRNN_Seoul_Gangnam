# Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting

# 임시 readme

강남구 데이터로 훈련된 dcrnn 모델

dcrnn 조금 수정했습니다

start.py로 시작하고 fine_name에 파일 이름 넣어주심 됩니다

make_predict로 예측합니다

사용된 데이터는 강남구 거리 속도 데이터입니다

topis에서 가져와 전처리 후 사용하였습니다

# 새로운 모델 훈련
python start.py --file_name="사용하는 파일"

## 위 명령어 입력시

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
