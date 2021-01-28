# What's this?
This folder contains code for running HPOlib2 benchmarks as containers.

The mechanism used is known as RPC and is implemented by Pyro4. There is
a client class simulating the original API for running the benchmark
inside the container.

The client class controls the container. The RPC server is implemented by
a server class. The server class follows the original API and runs inside
the container. Both classes are using JSON and Pyro4 to communicate.

# How can I use it?
There are some modified examples in the subfolder `examples`. It requires
at least ConfigSpace 0.4.7 and Pyro4. On Ubuntu you can install it with
* `sudo apt install pyro4 python3-pyro4 -y`
* `pip3 install configspace`

You can run the example with
`python3 example.py`

At the moment it uses Singularity. So you have to install Singularity first:
https://singularity.lbl.gov/install-linux

## Config options
There are some new configuration options that can be adjusted:
* `socket_dir`: The folder where the unix socket will be stored. It's used for the
communication between server and client. So the folder has to be accessible
inside and outside the container.
* `image_dir`: The folder where the singularity images will be stored. Singularity
can be restricted to run images only from certain paths. In this case this config
option should be adjusted accordingly.
* `image_source`: The base source path of the singularity images. It must be
adjusted only if you want to test your own images.
* `use_global_data`: The benchmark container can provide needed data files. In
this case the container files will be used by default. If you want to use the data
dir defined in `data_dir`, you have to set `use_global_data=False`.
* `pyro_connect_max_wait`: The benchmark server needs a while for getting ready, so
benchmark client will wait for it. In case of problems the client could wait forever.
For avoiding this, there is a time limit.

# How can I build my own container?
For providing a new benchmark as container, you have to write a client class and
a Singularity recipe.

## The client class
The client class should have the same name as the original benchmark class.
It should be stored inside the client subfolder. The client subfolder has the same
hierarchy as the folder `benchmarks`.

For the client class you have to write at least a constructor. Inside the constructor
you have to set the property `self.bName`. Also you have to call the
`self.setup(**kwargs)` method. You can find such a minimal example in
`client/ml/svm_benchmark.py`.

If you want to use a container for more than one benchmark, you maybe have to set the
named argument imgName of the method `self._setup()`. You can find such an example
in `client/synthetic_functions/levy.py`.

If your benchmark needs GPU support, you have to set the named argument gpu of the
method `self._setup()` to `True`. Look in `client/ml/convnet_net.py` for getting an
example.

If your benchmark constructor has some mandatory arguments, this should also be true
for your client class. You have to add the values to the kwargs dictionary. Look in
`client/ml/fcnet_classification.py` for getting an example.

You have to check your return data types. If they aren't serializable with Pyro4,
you should think about using other data types. If this isn't possible, you have to
adjust the encoder class `BenchmarkEncoder` defined in `server/abstract_benchmark.py`.
The deserialization should be done inside your client class. For this you have to
write some additional methods. Look in `client/ml/autosklearn_benchmark.py` for
getting an example.

If you provide some variants of your benchmark via sub classes and your client
classes are getting more complicated, you should write a client base class.

## The singularity recipe
The recipes are stored in the subfolder `singularity/`. For getting an basic idea,
you should read the official documentation:
* https://www.sylabs.io/guides/2.5/user-guide/container_recipes.html

First you should look for an base image. It's easier if this fits to your development
environment. So if you are using Ubuntu 18.04, you should think about using an Ubuntu
18.04 image as the base image. If your benchmark needs CUDA support, you should use
the official Docker images from NVIDIA. For getting an example look in
`singularity/ml/bnn_benchmark/Singularity.BNNOnYearPrediction`.

In the post section, you have to install (e.g. with apt and pip3) the libraries and
Python packages that are needed for running your benchmark. It's helpful if you have
used an clean development system. In any case you should check which libraries and
Python packages are installed. Additionally you have to install Pyro4.

If your benchmark needs some data files, you can provide the files inside the
container folder `/var/lib/hpolib/`. If the download is normally be done by the
benchmark constructor, you can call the script `util/download_data.py`. There
folder /var/lib/hpolib/ should be writable for everyone, so you have to adjust the
permissions inside the recipe. For getting an example look in
`singularity/ml/svm_benchmark/Singularity.SvmOnVehicle`.

In the runscript section you have to provide a call to `server/abstract_benchmark.py`.
The call has to use the python parameter `-s` for avoiding using python modules that may
be installed inside the `$HOME` folder. Also you have to provide the import base
`importBase` so that the benchmark can be imported by the benchmark server via
`importBase.benchmarkName`.

For getting the full container path of `server/abstract_benchmark.py` you should check
which Python version will be provided inside the container.
