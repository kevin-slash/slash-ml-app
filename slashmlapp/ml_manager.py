""" This is test machine learning class
"""


import sys, os
import argparse
import time
import numpy as np

from khmerml.machine_learning import MachineLearning
from khmerml.preprocessing.preprocessing_data import Preprocessing
from khmerml.utils.file_util import FileUtil
from khmerml.utils.bg_colors import Bgcolors

import logging
class MLManager(object):
  """ 
    Machine learning application built on top of slashml
  """

  @staticmethod
  def perform_algo(ml, algo, dataset):
    result_acc = []
    result_acc_train = []
    result_exec_time = []
    for i in range(10):
      exec_st = time.time()
      # split dataset -> train set, test set
      training_set, test_set = ml.split_dataset(dataset, 1)
      # train
      model = algo.train(training_set)
      # make a prediction
      pred_test = algo.predict(model, test_set)
      pred_train = algo.predict(model, training_set)

      # Prediction accuracy
      acc = ml.accuracy(pred_test, test_set)
      acc_train = ml.accuracy(pred_train, training_set)
      exec_time = time.time() - exec_st
      print(acc, acc_train, exec_time)
      result_acc.append(acc)
      result_acc_train.append(acc_train)
      result_exec_time.append(exec_time)

    mean_acc = np.mean(np.array(result_acc))
    mean_acc_train = np.mean(np.array(result_acc_train))
    mean_exec_time = np.mean(np.array(result_exec_time))

    return {
      'acc': round(mean_acc,2),
      'acc_train': round(mean_acc_train,2),
      'exec_time': round(mean_exec_time,2)
    }

  @staticmethod
  def get_results(path_textfile, algo_list, eval_setting, start_time):
    """
      This function performs features extraction from client's data source\
      Train model based on extracted features
      Get Accuracy of each algorithm (e.g: Naive Bayes, Neural Network) based on\
      evaluation criteria e.g: LOO, 5 folds or 10 folds
    """

    config = {
      'text_dir': 'data/dataset/text',
      'dataset': 'data/matrix',
      'bag_of_words': 'data/bag_of_words',
      'train_model': 'data/model/train.model',
      'archive_dir': 'data/dataset/temp'
    }

    #logfile = '/Users/lion/Documents/py-workspare/slash-ml/logfile.log'
    #logging.basicConfig(filename=logfile, level=logging.DEBUG)

    #logging.info('Start ML')
    # Perform features extraction
    is_successful_fextract = MLManager.extract_features(path_textfile, config)
    #is_successful_fextract = True

    if is_successful_fextract:
      whole_st = time.time()

      prepro = Preprocessing(**config)
      # preposessing
      dataset_matrix = prepro.loading_data(config['text_dir'], 'doc_freq', 'all', 1)
      #load dataset from file (feature data)
      filename = "doc_freq_1.csv"
      dataset_path = FileUtil.dataset_path(config, filename)
      dataset_sample = FileUtil.load_csv(dataset_path)

      prepro_time = time.time() - whole_st

      ml = MachineLearning(**config)
      # choose your algorithm
      nb_algo = ml.NiaveBayes()
      nn_algo = ml.NeuralNetwork(hidden_layer_sizes=(250, 100), learning_rate=0.012, momentum=0.5, random_state=0, max_iter=200, activation='tanh')
      dt_algo = ml.DecisionTree(criterion='gini', prune='depth', max_depth=30, min_criterion=0.05)

      nb_result = MLManager.perform_algo(ml, nb_algo, dataset_sample)
      nn_result = MLManager.perform_algo(ml, nn_algo, dataset_sample)
      dt_result = MLManager.perform_algo(ml, dt_algo, dataset_sample)

      print(nb_result, nn_result, dt_result)

      total_execution_time = time.time() - whole_st

      result = {
        'com_time': round(total_execution_time,2),
        'text_extract_time': round(prepro_time,2),
        'figure_on_testing_data': {
          'NB': nb_result['acc'],
          'NN': nn_result['acc'],
          'DT': dt_result['acc'],
        },
        'figure_on_training_data': {
          'NB': nb_result['acc_train'],
          'NN': nn_result['acc_train'],
          'DT': dt_result['acc_train'],
        },
        'on_testing_data': {
          'NB': {'accuracy': nb_result['acc'], 'time': nb_result['exec_time']},
          'NN': {'accuracy': nn_result['acc'], 'time': nn_result['exec_time']},
          'DT': {'accuracy': dt_result['acc'], 'time': dt_result['exec_time']},
        },
        'on_training_data': {
          'NB': {'accuracy': nb_result['acc_train'], 'time': nb_result['exec_time']},
          'NN': {'accuracy': nn_result['acc_train'], 'time': nn_result['exec_time']},
          'DT': {'accuracy': dt_result['acc_train'], 'time': dt_result['exec_time']},
        }
      }

    return result

  @staticmethod
  def extract_features(text_file, config):
    """ this function can be used to extract features \
    in a format supported by machine learning \
    algorithms from datasets consisting of formats such as text.
    """

    #config = MLManager.config
    #path_to_zipfile = FileUtil.path_to_file(config, config['text_dir'], text_file)
    #path_to_tempdir = FileUtil.path_to_file(config, config['archive_dir'], text_file)
    path_to_zipfile = FileUtil.path_to_file(config['text_dir'], text_file)
    path_to_tempdir = FileUtil.path_to_file(config['archive_dir'], text_file)

    if os.path.exists(path_to_zipfile) is False:
        # Move data.zip file to temp directory
        FileUtil.move_file(path_to_tempdir, path_to_zipfile)
    try:
        # Extract zip file
        FileUtil.extract_zipfile(path_to_zipfile, FileUtil.join_path(config['text_dir']))
        # Move data.zip file to temp directory
        FileUtil.move_file(path_to_zipfile, path_to_tempdir)
    except OSError as error:
      raise Exception(error)
    else:
      return True

if __name__ == "__main__":

    _path_textfile = 'chatbot.zip'
    #_path_textfile = 'data.zip'
    _list_algo = ['DT']

    #print(MLManager.extract_features(_text_file))
    #print(MLManager.train())
    #_result_data_accuracy, _result_ontrainingdata_dict = MLManager.get_results(_path_textfile, _list_algo, '')
    start_time = time.time() 
    results = MLManager.get_results(_path_textfile, _list_algo, '', start_time)

    #print('accuracy %s, on testin data %s' %(_result_data_accuracy, _result_ontrainingdata_dict))
    print('accuracy %s' %(results))
