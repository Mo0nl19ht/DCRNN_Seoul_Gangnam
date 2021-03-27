from scripts.generate_training_data import main_refer_gen_data
from scripts.gen_adj_mx import main_refer_gen_adj_mx
from dcrnn_train import main_refer_train
from scripts.eval_baseline_methods import main_refer_eval
import argparse
import os

# 파일이름은 Seoul_all 로 함
parser = argparse.ArgumentParser()
parser.add_argument(
    "--file_name", type=str, default="Seoul_all", help="file_name"
)
file = parser.parse_args()

path = f'./data/{file.file_name}'

try:
    os.mkdir(path)
except:
    True


#데이터 준비하기 - train, test, valivalidation 나누기
main_refer_gen_data(file.file_name)
print("\n\n\n\n@@@@@Data generated@@@@@\n\n\n\n")
#그래프 구성
main_refer_gen_adj_mx(file.file_name)
print("\n\n\n\n@@@@@Graph generated@@@@@\n\n\n\n")
# 트레이닝
main_refer_train(file.file_name)
print("\n\n\n\n@@@@@Model generated@@@@@\n\n\n\n")
# baseline 비교
main_refer_eval(file.file_name)
print("\n\n\n\n@@@@@Baseline generated@@@@@\n\n\n\n")