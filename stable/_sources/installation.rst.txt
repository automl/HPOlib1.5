******************
Installing HPOlib2
******************

.. role:: bash(code)
    :language: bash

.. role:: python(code)
    :language: python

Installing from the master branch
*********************************

As HPOlib2 is under development, there is not yet a stable release. We do our
best to only merge to the master branch if the code is stable, but cannot
give any guarantees. Thus

The dependencies of ``HPOlib2`` are listed in the file ``requirements.txt``
and contain all dependencies which are necessary to run most of the
benchmarks. To install these, execute:

.. code:: bash

    curl https://raw.githubusercontent.com/automl/HPOlib2/master/requirements.txt | xargs -n 1 -L 1 pip install

Further dependencies are listed in ``optional-requirements.txt`` and can be
installed with:

.. code:: bash

    curl https://raw.githubusercontent.com/automl/HPOlib2/master/optional-requirements.txt | xargs -n 1 -L 1 pip install

Afterwards, you can install HPOlib2:

.. code:: bash

    pip install git+https://github.com/automl/HPOlib2

Installing from the development branch
**************************************

.. code:: bash

    curl https://raw.githubusercontent.com/automl/HPOlib2/development/requirements.txt | xargs -n 1 -L 1 pip install
    curl https://raw.githubusercontent.com/automl/HPOlib2/development/optional-requirements.txt | xargs -n 1 -L 1 pip install
    pip install git+https://github.com/automl/HPOlib2@development

Additional information
**********************

* We recommend using `virtual environments <https://virtualenv.pypa.io/en/stable/>`_
  or `AnaConda <https://www.continuum.io/downloads>`_ to set up one or more
  isolated python environments.
* We develop all our code with Ubuntu. We don't run any tests to check
  whether our code runs on different operating systems and therefore cannot
  guarantee any compability.
