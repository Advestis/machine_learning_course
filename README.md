# Machine Learning Course : Practical work

## Content 

* First steps in Python (*first_steps.ipynb* and *first_steps_exercise.ipynb*)
* Python for data science (*py_data_science.ipynb* and *py_data_science_exercise(_2).ipynb*)
* Hand-made linear regression (*linreg/*)
* Logistic regression with sklearn (*logistic_regression.ipynb*)
* Neural Network with NumPy and Tensorflow (*neural_network*)

## Installation

**Note** : the installation instructions here are valid for Unix systems (Linux and Mac). I do not use Windows so I do
not know the equivalent commands on Windows. Look them up on the internet.

First of all, install git (look up the internet) then clone this repository on your computer (in your document for example, but that's up to you). To do that,
on Unix open a terminal and do

```bash
git clone https://github.com/Advestis/machine_learning_course
```

This will copy the content of this repository in a new directory called *machine_learning_course*. 

You will need 
 * to install PyCharm (look up the internet to find out how)
 * to have Python 3.8 (same, look up the internet)
 * to have pip installed for Python 3.8 (same, look up the internet)
 * to make a working virtual environment : on Unix, open a terminal in the directory *
cd machine_learning_course* and launch 
   * `python3.8 -m pip install virtualenv`
   * `python3.8 -m virtualenv venv`
   This will create a virtual environment called *venv* in your current directory.
 
Once the virtualenv is created, activate it. On Unix :

```bash
source venv/bin/activate
```

Then install the required python packages :

```bash
pip install -r requirements.txt
```

## Jupyter Lab

Jupyter Lab is a web-based user interface for Python. It is perfect for writing small scripts that execute
specific tasks or to test new ideas. It is not the tool to use to develop large software however, for it does
not include an efficient debug tool, does not include PEP8 warning and automatic code completion or correction,
and its files (notebooks, with the *.ipynb* extension) can not be executed directly in a command line.

We will use it to introduce Python for data science, and experiment with sklearn and TensorFlow.

To open a Jupyter Lab notebook, open a terminal in the directory containing the notebook, source your venv and write 

```bash
jupyter lab
```

Then click on your notebook

## PyCharm

PyCharm is the best IDE for developping in Python. It includes a lot of tools like debugging, automatic code completion,
PEP8 corrections suggestions, has a builtin virtualenv management, supports git, and supports the installations of
various plugins.

We will use it to create our own linear regression algorithm from scratch.

## TensorBoard

```bash
tensorboard --logdir neural_network/logs/fit/
```


## Resources

* All useful Python tutorials : https://www.w3schools.com/python/default.asp
* Neural network in numpy : https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
* slides from theoretical course useful for the exercises : https://www.overleaf.com/read/nfprtjqgvzyc