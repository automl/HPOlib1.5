import abc


class AbstractIterativeBenchmark(object, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_data(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_model(self, config=None):
        raise NotImplementedError()
