'''
@author: Stefan Staeglich
'''

from hpolib.container.client.abstract_benchmark import AbstractBenchmarkClient


class CartpoleFull(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "CartpoleFull"
        self._setup(imgName="CartpoleBase", **kwargs)


class CartpoleReduced(AbstractBenchmarkClient):
    def __init__(self, **kwargs):
        self.bName = "CartpoleReduced"
        self._setup(imgName="CartpoleBase", **kwargs)
