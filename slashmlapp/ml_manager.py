""" This is test machine learning class
"""

import os
from time import clock


from slashml.utils.file_util import FileUtil
from slashmlapp.machinelearning import MachineLearning
from slashml.preprocessing.preprocessing_data  import Preprocessing


class MLManager(object):
    """
     Machine learning application built on top of slashml
    """

    CONFIG = {
        #'root': '/var/www/slashml2/slash-ml',
        'root': '/Users/lion/Documents/py-workspare/slash-ml',
        'model_dataset': 'data/dataset',
        'train_model': 'data/naive_bayes_model.pickle',
        'train_dataset': 'data/train_dataset.pickle',
        'test_dataset': 'data/test_dataset.pickle',
        'text_dir': 'data/dataset/text',
        'archive_dir': 'data/dataset/temp',
        'dataset': 'data/dataset/matrix',
        'dataset_filename': 'data.csv',
        'mode': 'unicode'
    }

    ON_TESTING_DATA = 'on_testing_data'
    ON_TRAINING_DATA = 'on_training_data'
    ACCURACY = 'accuracy'
    TIME = 'time'
    FIGURE_ON_TESTING_DATA = 'figure_on_testing_data'
    FIGURE_ON_TRAINING_DATA = 'figure_on_training_data'

    @staticmethod
    def get_results(path_textfile, algo_list, eval_setting):
        """
            This function performs features extraction from client's data source\
            Train model based on extracted features
            Get Accuracy of each algorithm (e.g: Naive Bayes, Neural Network) based on\
             evaluation criteria e.g: LOO, 5 folds or 10 folds
        """

        config = MLManager.CONFIG

        # Keep track of result
        nbr_simulations = 1
        result_ontestingdata_accuracy = {}
        result_ontestingdata_dict = {}

        # result of on training data
        result_ontrainingdata_accuracy = {}
        result_ontrainingdata_dict = {}

        formatted_final_result = {}

        # Perform features extraction
        is_successful_fextract = MLManager.extract_features(path_textfile)
        #is_successful_fextract = True

        if is_successful_fextract:

            # Peform prediction and operation on testing data
            result_ontestingdata_accuracy, result_ontestingdata_dict = \
            MLManager.execute_ontestingdata(algo_list, config, nbr_simulations)

            # Peform training and prediction on training data
            result_ontrainingdata_accuracy, result_ontrainingdata_dict = \
            MLManager.execute_ontrainingdata(algo_list, config, nbr_simulations)

            # Structure data in dictionary
            formatted_final_result[MLManager.ON_TESTING_DATA] = result_ontestingdata_dict
            formatted_final_result[MLManager.FIGURE_ON_TESTING_DATA] = result_ontestingdata_accuracy

            # Structure results of training data
            formatted_final_result[MLManager.ON_TRAINING_DATA] = result_ontrainingdata_dict
            formatted_final_result[MLManager.FIGURE_ON_TRAINING_DATA] = result_ontrainingdata_accuracy

        #return result_ontrainingdata_accuracy, result_ontrainingdata_dict
        return formatted_final_result

    @staticmethod
    def execute_ontestingdata(algo_list, config, nbr_simulations):
        """ Peform training and prediction using on training data method
        """

        # Keep track of result
        result_data_accuracy = {}
        result_ontestingdata_dict = {}
        # Start training and predicion
        for _, algo in enumerate(algo_list):
            ''' if algo == 'NB':
                ml_algo = MachineLearning(**config).make_naivebayes()
            elif algo == 'NN':
                ml_algo = MachineLearning(**config).make_naivebayes()
            elif algo == 'DL':
                ml_algo = MachineLearning(**config).make_naivebayes()
            else:
                ml_algo = object() # instance empty object '''

            test_counter = 0
            accuracy_list = []
            computing_time_list = []

            while test_counter < nbr_simulations:

                if algo == 'NB':
                    ml_algo = MachineLearning(**config).make_naivebayes()
                elif algo == 'NN':
                    ml_algo = MachineLearning(**config).make_naivebayes()
                elif algo == 'DL':
                    ml_algo = MachineLearning(**config).make_naivebayes()
                else:
                    ml_algo = object() # instance empty object

                # Start time
                #start = clock()

                # Train and Get Accuracy
                accuracy, elapsed = MLManager.train_model(ml_algo, MLManager.execute_ontestingdata, config)

                # Calcuate computing time
                #end = clock()
                #elapsed = end - start

                # Increment counter
                test_counter = test_counter + 1

                # Keep tracking the accuracy per operation
                accuracy_list.append(accuracy)
                computing_time_list.append(elapsed)

            if algo not in result_data_accuracy:
                #result_data_accuracy[algo] = []
                result_data_accuracy[algo] = round(sum(accuracy_list)/len(accuracy_list), 2)

            # Add accuracy list of each algo to dico
            #result_data_accuracy[algo] = accuracy_list

            # Format accuracy and computing time in dict format
            _accuracy_time_dict = MLManager.format_result_todict(accuracy_list, computing_time_list)
            # Add result_dict to corresponding algo ex: NB, NN
            result_ontestingdata_dict[algo] = _accuracy_time_dict

        return result_data_accuracy, result_ontestingdata_dict

    @staticmethod
    def execute_ontrainingdata(algo_list, config, nbr_simulations):
        """ Peform training and prediction using on training data method
        """

        # Keep track of result
        result_data_accuracy = {}
        result_ontestingdata_dict = {}
        # Start training and predicion
        for _, algo in enumerate(algo_list):
            ''' if algo == 'NB':
                ml_algo = MachineLearning(**config).make_naivebayes()
            elif algo == 'NN':
                ml_algo = MachineLearning(**config).make_naivebayes()
            elif algo == 'DL':
                ml_algo = MachineLearning(**config).make_naivebayes()
            else:
                ml_algo = object() # instance empty object '''

            test_counter = 0
            accuracy_list = []
            computing_time_list = []

            while test_counter < nbr_simulations:

                if algo == 'NB':
                    ml_algo = MachineLearning(**config).make_naivebayes()
                elif algo == 'NN':
                    ml_algo = MachineLearning(**config).make_naivebayes()
                elif algo == 'DL':
                    ml_algo = MachineLearning(**config).make_naivebayes()
                else:
                    ml_algo = object() # instance empty object

                # Start time
                #start = clock()

                # Train and Get Accuracy
                accuracy, elapsed = MLManager.train_model(ml_algo, MLManager.execute_ontrainingdata, config)

                # Calcuate computing time
                #end = clock()
                #elapsed = end - start

                # Increment counter
                test_counter = test_counter + 1
                # Keep tracking the accuracy per operation
                accuracy_list.append(accuracy)
                computing_time_list.append(elapsed)

            if algo not in result_data_accuracy:
                #result_data_accuracy[algo] = []
                result_data_accuracy[algo] = round(sum(accuracy_list)/len(accuracy_list), 2)

            # Add accuracy list of each algo to dico
            #result_data_accuracy[algo] = accuracy_list

            # Format accuracy and computing time in dict format
            _accuracy_time_dict = MLManager.format_result_todict(accuracy_list, computing_time_list)
            # Add result_dict to corresponding algo ex: NB, NN
            result_ontestingdata_dict[algo] = _accuracy_time_dict

        return result_data_accuracy, result_ontestingdata_dict


    @staticmethod
    def format_result_todict(accuracy_list, computingtime_list):
        """ Format result in a dictionary of two elements
            - Accuracy
            - Time
        """

        result = {}
        avg_accuracy = round(sum(accuracy_list)/len(accuracy_list), 2)
        avg_computingtime = round(sum(computingtime_list)/len(computingtime_list), 2)

        result[MLManager.ACCURACY] = avg_accuracy
        result[MLManager.TIME] = avg_computingtime

        return result


    @staticmethod
    def extract_features(text_file):
        """ this function can be used to extract features \
        in a format supported by machine learning \
        algorithms from datasets consisting of formats such as text.
        """

        config = MLManager.CONFIG
        path_to_zipfile = FileUtil.path_to_file(config, config['text_dir'], text_file)
        path_to_tempdir = FileUtil.path_to_file(config, config['archive_dir'], text_file)

        if os.path.exists(path_to_zipfile) is False:
            # Move data.zip file to temp directory
            FileUtil.move_file(path_to_tempdir, path_to_zipfile)

        try:

            # Extract zip file
            FileUtil.extract_zipfile(path_to_zipfile, FileUtil.join_path(config, config['text_dir']))

            # Move data.zip file to temp directory
            FileUtil.move_file(path_to_zipfile, path_to_tempdir)
        except OSError as error:
            raise Exception(error)

        try:
            prepro = Preprocessing(**config)
            dataset_matrix = prepro.loading_data(config['text_dir'], 'doc_freq', 15)

            # Save features in text file
            path_to_filetext = FileUtil.path_to_file(config, config['dataset'], config['dataset_filename'])
            prepro.write_mat(path_to_filetext, dataset_matrix)

        except OSError as error:
            #print(error)
            raise Exception(error)
        else:
            return True

    @staticmethod
    def train():
        """ Test purpose
        """

        config = MLManager.CONFIG
        naive_bayes = MachineLearning(**config).make_naivebayes()

        #filename = "data.csv"
        filename = config['dataset_filename']
        #filename = "feature15_13.csv"
        path_to_cvs_dataset = FileUtil.path_to_file(config, config['dataset'], filename)
        #path_to_cvs_dataset = FileUtil.path_to_file(config, config['model_dataset'], filename)
        dataset_matrix = FileUtil.load_csv(path_to_cvs_dataset)

        #traning_model = FileUtil.load_model(CONFIG)
        # Splite dataset into two subsets: traning_set and test_set
        # training_set:
            # it is used to train our model
        # test_set:
            # it is used to test our trained model
        training_set, test_set = naive_bayes.split_dataset(dataset_matrix, 6)
        #_ = FileUtil.save_pickle_dataset(config, config['train_dataset'], training_set)
        #_ = FileUtil.save_pickle_dataset(config, config['test_dataset'], test_set)

        _train_model = naive_bayes.train(training_set)
        _ = naive_bayes.predict(test_set)

        #print("Accuracy ", naive_bayes.naive_bayes.accuracy(test_set))
        #accuracy = 'Accuracy: {0}'.format(naive_bayes.naive_bayes.accuracy(test_set))
        accuracy = naive_bayes.naive_bayes.accuracy(test_set)

        return accuracy

    @staticmethod
    def train_model(ml_algo, operation_mode, config):
        """ Test purpose
        """

        # Trace computing time of this train and prediction process
        # Start time
        start = clock()

        #filename = "data.csv"
        filename = config['dataset_filename']
        #filename = "feature15_13.csv"
        path_to_cvs_dataset = FileUtil.path_to_file(config, config['dataset'], filename)
        #path_to_cvs_dataset = FileUtil.path_to_file(config, config['model_dataset'], filename)
        dataset_matrix = FileUtil.load_csv(path_to_cvs_dataset)

        #traning_model = FileUtil.load_model(CONFIG)
        # Splite dataset into two subsets: traning_set and test_set
        # training_set:
            # it is used to train our model
        # test_set:
            # it is used to test our trained model
        if operation_mode == MLManager.execute_ontestingdata:
            training_set, test_set = ml_algo.split_dataset(dataset_matrix, 6)
        else:
            training_set, test_set = ml_algo.extract_testingdata_dataset(dataset_matrix, 6)

        ''' if bool(ml_algo.train_model):
            ml_algo.load_model()
        else:
            _train_model = ml_algo.train(training_set) '''

        # train and predict
        _train_model = ml_algo.train(training_set)
        _ = ml_algo.predict(test_set)

        #print("Accuracy ", naive_bayes.naive_bayes.accuracy(test_set))
        #accuracy = 'Accuracy: {0}'.format(naive_bayes.naive_bayes.accuracy(test_set))
        accuracy = ml_algo.accuracy(test_set)

        # Calcuate computing time
        end = clock()
        elapsed = end - start

        return accuracy, elapsed

    @staticmethod
    def train_get_accuracy(ml_algo, config):
        """ Test purpose
        """

        #filename = "data.csv"
        filename = config['dataset_filename']
        #filename = "feature15_13.csv"
        path_to_cvs_dataset = FileUtil.path_to_file(config, config['dataset'], filename)
        #path_to_cvs_dataset = FileUtil.path_to_file(config, config['model_dataset'], filename)
        dataset_matrix = FileUtil.load_csv(path_to_cvs_dataset)

        #traning_model = FileUtil.load_model(CONFIG)
        # Splite dataset into two subsets: traning_set and test_set
        # training_set:
            # it is used to train our model
        # test_set:
            # it is used to test our trained model
        training_set, test_set = ml_algo.split_dataset(dataset_matrix, 6)
        #_ = FileUtil.save_pickle_dataset(config, config['train_dataset'], training_set)
        #_ = FileUtil.save_pickle_dataset(config, config['test_dataset'], test_set)

        _train_model = ml_algo.train(training_set)
        predicts = ml_algo.predict(test_set)

        #print("Training model ", _train_model)
        #print("Predicts ", predicts)

        accuracy = ml_algo.accuracy(test_set)

        return accuracy

if __name__ == "__main__":

    #test_machinelearning = TestMachineLearning()

    _path_textfile = 'data.zip'
    #_text_file = 'data.csv'
    #_text_file = 'feature22.csv'
    _list_algo = ['NB', 'NN']

    #print(MLManager.extract_features(_text_file))
    #print(MLManager.train())
    #_result_data_accuracy, _result_ontrainingdata_dict = MLManager.get_results(_path_textfile, _list_algo, '')
    results = MLManager.get_results(_path_textfile, _list_algo, '')

    #print('accuracy %s, on testin data %s' %(_result_data_accuracy, _result_ontrainingdata_dict))
    print('accuracy %s' %(results))
