import argparse
import numpy as np
import os
import sys
import tensorflow as tf
import yaml
import pandas as pd
from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor


def run_dcrnn(args):
    with open(args.config_filename) as f:
        config = yaml.load(f)
    tf_config = tf.ConfigProto()
    if args.use_cpu_only:
        tf_config = tf.ConfigProto(device_count={'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    graph_pkl_filename = config['data']['graph_pkl_filename']
    _, _, adj_mx = load_graph_data(graph_pkl_filename)
    with tf.Session(config=tf_config) as sess:
        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
        supervisor.load(sess, config['train']['model_filename'])
        outputs = supervisor.evaluate(sess)
        for i in range(12):
            origin = pd.DataFrame(outputs['groundtruth'][i].reshape(len(outputs['groundtruth'][0]), len(outputs['groundtruth'][0][0])))
            predict = pd.DataFrame(outputs['predictions'][i].reshape(len(outputs['predictions'][0]), len(outputs['predictions'][0][0])))
            origin.to_csv(f'prediction/after_{i + 1}h_origin.csv')
            predict.to_csv(f'prediction/after_{i+1}h_predict.csv')
        #np.savez_compressed(args.output_filename, **outputs)
        #print('Predictions saved as {}.'.format(args.output_filename))
        print(f'Predictions saved as {"_predict.csv"}.')

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/Gangnam_trained/config_100.yaml', type=str,
                        help='Config file for pretrained model.')
    #parser.add_argument('--output_filename', default='data/dcrnn_Gangnam_predictions.npz')
    args = parser.parse_args()
    run_dcrnn(args)
