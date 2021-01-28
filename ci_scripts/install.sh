#!/usr/bin/env sh
# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Use the miniconda installer for faster download / install of conda
# itself
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"

ls -l
echo

if [[ ! -f miniconda.sh ]]
   then
   wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
       -O miniconda.sh
   fi

chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
cd ..
# export PATH="$HOME/miniconda/bin:$PATH"
export PATH=$HOME/miniconda/bin:$PATH
conda update --yes conda
popd

# Configure the conda environment and put it in the path using the
# provided versions
conda create -n testenv --yes python=$PYTHON_VERSION pip wheel nose
source activate testenv

echo "Using GCC at "`which gcc`
echo "GCC Version " `gcc --version`
export CC=`which gcc`

echo "SWIG Version "`swig -version`

pip install coverage pep8 codecov
cat requirements.txt | grep -v '#' | xargs -n 1 -L 1 pip install
cat optional-requirements.txt | grep -v '#' | xargs -n 1 -L 1 pip install
python setup.py install
