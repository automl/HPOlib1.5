Quickstart Guide
================

All benchmarks in HPOlib2 follow the same interface and run using **Python 3.5**.
It is designed to be easy to use and works out of the box with optimizers
from the scipy.optimize module for continuous problems.

--------------------------
Download & Install HPOlib2
--------------------------

.. code-block:: bash

  git clone git@github.com:automl/HPOlib2.git
  cd HPOlib2
  for i in `cat requirements.txt`; do pip install $i; done
  python setup.py install

(optionally) run tests

.. code-block:: bash

  nosetests tests

A more detailed explanation can be found in the :ref:`installation` section.

-------
Example
-------

.. code-block:: python

  from hpolib.benchmarks.ml import svm_benchmark

  # Download datasets
  b = svm_benchmark.SvmOnMnist()

  # Evaluate one configuration
  res = b.objective_function(configuration=[5, -5])
  # res = {'cost': 251.88, 'function_value': 0.012}

  # Evaluate one configuration on a subset
  res = b.objective_function(configuration=[5, -5], dataset_fraction=0.5)

  # Show hyperparameter space
  print(b.configuration_space)

  # Get meta information
  print(b.get_meta_information())

All our benchmarks support an interface similar to this. More examples can be
found in the example folder in ``HPOlib2/examples``
