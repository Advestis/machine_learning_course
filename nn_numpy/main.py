import numpy as np
import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from functions import make_plot, train, full_forward_propagation, get_accuracy_value


# 5 Layer network. First layer takes two entries : x (pairs of coordinates) and y (a vector of 0 and 1). The output
# layer has one output, that is activated by a sigmoid, and can thus take the value 0 or 1, corresponding to one or the
# other classification possibilities
NN_ARCHITECTURE = [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]

# number of samples in the data set
N_SAMPLES = 1000
# ratio between training and test sets
TEST_SIZE = 0.1

# Create artificial data
x, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)
# Split test and train sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)

# Plot the artificial data
make_plot(x, y, "Dataset", file_name="dataset.pdf")

# Training
params_values = train(
    x=np.transpose(x_train),
    y=np.transpose(y_train.reshape((y_train.shape[0], 1))),
    nn_architecture=NN_ARCHITECTURE,
    epochs=10000,
    learning_rate=0.01,
    verbose=True,
)

# Prediction
y_test_hat, _ = full_forward_propagation(np.transpose(x_test), params_values, NN_ARCHITECTURE)
# Accuracy achieved on the test set
acc_test = get_accuracy_value(y_test_hat, np.transpose(y_test.reshape((y_test.shape[0], 1))))
print("Test set accuracy: {:.2f}".format(acc_test))

# boundary of the graph
GRID_X_START = -1.5
GRID_X_END = 2.5
GRID_Y_START = -1.0
GRID_Y_END = 2
# output directory (the folder must be created on the drive)
OUTPUT_DIR = "./"

grid = np.mgrid[GRID_X_START:GRID_X_END:100j, GRID_X_START:GRID_Y_END:100j]
grid_2d = grid.reshape(2, -1).T
XX, YY = grid


def callback_numpy_plot(index, params):
    plot_title = "NumPy Model - It: {:05}".format(index)
    file_name = "numpy_model_{:05}.png".format(index // 50)
    file_path = os.path.join(OUTPUT_DIR, file_name)
    prediction_probs, _ = full_forward_propagation(np.transpose(grid_2d), params, NN_ARCHITECTURE)
    prediction_probs = prediction_probs.reshape(prediction_probs.shape[1], 1)
    make_plot(x_test, y_test, plot_title, file_name=file_path, XX=XX, YY=YY, preds=prediction_probs)


# Training
params_values = train(
    x=np.transpose(x_train),
    y=np.transpose(y_train.reshape((y_train.shape[0], 1))),
    nn_architecture=NN_ARCHITECTURE,
    epochs=10000,
    learning_rate=0.01,
    verbose=False,
    callback=callback_numpy_plot,
)

prediction_probs_numpy, _ = full_forward_propagation(np.transpose(grid_2d), params_values, NN_ARCHITECTURE)
prediction_probs_numpy = prediction_probs_numpy.reshape(prediction_probs_numpy.shape[1], 1)
make_plot(x_test, y_test, "NumPy Model", file_name=None, XX=XX, YY=YY, preds=prediction_probs_numpy)
