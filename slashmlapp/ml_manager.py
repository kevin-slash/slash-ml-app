""" This is test machine learning class
"""

from slashml.utils.file_util import FileUtil
from slashmlapp.machinelearning import MachineLearning
from slashml.preprocessing.preprocessing_data  import Preprocessing


class MLManager(object):
    """
     Machine learning application built on top of slashml
    """

    CONFIG = {
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

    @staticmethod
    def get_results(path_textfile, list_algo, eval_setting):
        """
            This function performs features extraction from client's data source\
            Train model based on extracted features
            Get Accuracy of each algorithm (e.g: Naive Bayes, Neural Network) based on\
             evaluation criteria e.g: LOO, 5 folds or 10 folds
        """

        config = MLManager.CONFIG

        is_successful_fextract = MLManager.extract_features(path_textfile)

        # Keep track of result
        results = {}

        if is_successful_fextract:

            # Start training and predicion
            for _, algo in enumerate(list_algo):
                if algo == 'NB':
                    ml_algo = MachineLearning(**config).make_naivebayes()
                elif algo == 'NN':
                    ml_algo = MachineLearning(**config).make_naivebayes()
                elif algo == 'DL':
                    ml_algo = MachineLearning(**config).make_naivebayes()
                else:
                    ml_algo = object() # instance empty object

                test_counter = 0
                accuracy_list = []
                while test_counter < 3:

                    # Train and Get Accuracy
                    accuracy = MLManager.train_model(ml_algo, config)

                    # Increment counter
                    test_counter = test_counter + 1

                    # Keep tracking the accuracy per operation
                    accuracy_list.append(accuracy)

                if algo not in results:
                    results[algo] = []

                # Add accuracy list of each algo to dico
                results[algo] = accuracy_list

                mode = max(set(accuracy_list), key=accuracy_list.count)
                #print("Accuracy list: ", accuracy_list)
                #print("Average Accuracy", round(sum(accuracy_list)/test_counter, 2))
                #print("Mode of accuracy is : ", mode)

            #print('Result :', results)
        return results


    @staticmethod
    def extract_features(text_file):
        """ this function can be used to extract features \
        in a format supported by machine learning \
        algorithms from datasets consisting of formats such as text.
        """

        config = MLManager.CONFIG
        #text_file = 'data.zip'
        path_to_zipfile = FileUtil.path_to_file(config, config['text_dir'], text_file)
        path_to_tempdir = FileUtil.path_to_file(config, config['archive_dir'], text_file)

        try:
            FileUtil.extract_zipfile(path_to_zipfile, FileUtil.join_path(config, config['text_dir']))
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
        predicts = naive_bayes.predict(test_set)

        #print("Training model ", _train_model)
        #print("Predicts ", predicts)

        #print("Accuracy ", naive_bayes.naive_bayes.accuracy(test_set))
        #accuracy = 'Accuracy: {0}'.format(naive_bayes.naive_bayes.accuracy(test_set))
        accuracy = naive_bayes.naive_bayes.accuracy(test_set)

        return accuracy

    @staticmethod
    def train_model(ml_algo, config):
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

        if bool(ml_algo.train_model):
            ml_algo.load_model()
        else:
            _train_model = ml_algo.train(training_set)

        predicts = ml_algo.predict(test_set)

        #print("Training model ", _train_model)
        #print("Predicts ", predicts)

        #print("Accuracy ", naive_bayes.naive_bayes.accuracy(test_set))
        #accuracy = 'Accuracy: {0}'.format(naive_bayes.naive_bayes.accuracy(test_set))
        accuracy = ml_algo.accuracy(test_set)

        return accuracy

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

        #print("Accuracy ", naive_bayes.naive_bayes.accuracy(test_set))
        #accuracy = 'Accuracy: {0}'.format(naive_bayes.naive_bayes.accuracy(test_set))
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
    MLManager.get_results(_path_textfile, _list_algo, '')
