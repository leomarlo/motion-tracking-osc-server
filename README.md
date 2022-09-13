# MediaPipe

This document describes the process of setting up a virtual environment (inside a repo) to run the MediaPipe for python.

## Make a (repo)folder

- Create a repo in bitbucket

- clone to a local folder

- edit *.gitignore* to include everything in the hidden folder *.venv/*

```
# Exclude files from virtual environment
.venv/
```

Then you can launch the terminal in that folder and set up an instance.

## Setting up an instance

Using the [getting started](https://google.github.io/mediapipe/getting_started/python) instructions. 

Calling the virtual environment `venv` is a convention...

```python
# create a python instance
# createing it in a hidden folder is a convention for better version control.
# the folder ./.venv (I assume) will be included in .gitignore
python3 -m venv ./.venv
source .venv/bin/activate


# (leave an environment)
deactivate
```

you can check out where the local python and pip binaries live:

```python
which python
which pip


# in order to check the versions:
python --version
pip --version
```

Just for the fun of it, we may want to install a jupyter notebook:

```python
pip install jupyter


# numpy is already included in jupyter, otherwise:
pip install numpy


# launch Visual Studio Code using the command included in PATH
code .
```

Once in *Visual Studio Code*, create a new *Jupyter* notebook:

- New file (Jupyter)

- Set kernel to the python local installation: **.venv/ (Python 3.7.0)**

- Save with an appropriate name (inside the git folder)

- Test that everything is fine by running:

```python
test = 1

# hit the run icon on the left
# alternatively: Shift + Return
```

If everything seems fine, we can start working in the notebook :)



### OSC in Python

In order to send/receive OSC data (according to this online documentation: [python-osc · PyPI](https://pypi.org/project/python-osc/)).

```python
pip install python-osc
```



### Using the python interpreter:

```python
# Enter the python interpreter
python3

# (leave the interpreter)
>>> exit()
```

Troubleshooting

I encountered an error when attempting to install **mediapipe** that seems to have to do with my python version...

```python
# trying to instal mediapipe
pip install mediapipe

# error message
ERROR: Could not find a version that satisfies the requirement mediapipe (from versions: none)
ERROR: No matching distribution found for mediapipe
```

in order to find out my python version:

```python
python --version
```

*(my version shows to be Python 3.9.12)*

Trying to fix the issue using `virtualenv` instead of `venv`, which seems to allow me to set the python version to the accepted 3.7.0.

```python
# install virtualenv
virtualenv -p python3.7 mp_env

#create instance with specific python version
virtualenv -p python3.7 mp_env
source .mp_env/bin/activate
```

This environment is set to 3.7.0 version of python, yet the problem persists.

```python
# Another try to create an environment, this time with venv
python3.7 -m venv ./.venv37
source .venv37/bin/activate
```

I'll try installing MediaPipe fro MacOs first, according to these instructions:

[MediaPipe#installing-on-macos](https://google.github.io/mediapipe/getting_started/install.html#installing-on-macos)

### How to find out my local python architecture (32/64 bits)

Go into python console by typin `python` in the terminal (inside /MediaPipe folder)

```python
#go into python console
python

# exectute commands to get python's architecture
import platform
platform.architecture()[0]


# leave the python console
exit()
```

____

# To-Do

- Adapt to the Effects format


# Contributors

Adrian,
Leo