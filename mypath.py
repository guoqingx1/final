class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'crack':
            return '/content/gdrive/MyDrive/dataset/'
            # return '/content/gdrive/MyDrive/dataset_test/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
