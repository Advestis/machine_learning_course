import numpy as np
import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from functions_numpy import make_plot, train, full_forward_propagation, get_accuracy_value
from functions_tf import build_mlp
from tensorflow.keras.callbacks import TensorBoard

import argparse

parser = argparse.ArgumentParser(
    description="main",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--engine",
    "-e",
    type=str,
    help="numpy or tf",
    default="numpy"
)


def callback_numpy_plot(index, params, which="numpy"):
    if which == "numpy":
        plot_title = "NumPy Model - It: {:05}".format(index)
        file_name = "numpy_model_{:05}.png".format(index // 50)
    else:
        plot_title = "Tensorflow Model - It: {:05}".format(index)
        file_name = "tensorflow_{:05}.png".format(index // 50)

    file_path = os.path.join(OUTPUT_DIR, file_name)
    prediction_probs, _ = full_forward_propagation(np.transpose(grid_2d), params, NN_ARCHITECTURE)
    prediction_probs = prediction_probs.reshape(prediction_probs.shape[1], 1)
    make_plot(x_test, y_test, plot_title, file_name=file_path, XX=XX, YY=YY, preds=prediction_probs)


NN_ARCHITECTURE = [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]

""" EXERCISE : 
Draw the neural network corresponding to NN_ARCHITECTURE. 
What does it do ? Regression or Classification ? Hint : look at the activation function of the last layer.
Why are there 2 dimensions in the input of the first layer ?
 """

# number of samples in the data set
N_SAMPLES = 1000
# ratio between training and test sets
TEST_SIZE = 0.1

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

# Create artificial data
x, y = make_moons(n_samples=N_SAMPLES, noise=0.2, random_state=100)
# Split test and train sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)

# Plot the artificial data
make_plot(x, y, "Dataset", file_name="dataset.pdf")


def use_numpy():
    # Training with NumPy
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

    # Training
    params_values = train(
        x=np.transpose(x_train),
        y=np.transpose(y_train.reshape((y_train.shape[0], 1))),
        nn_architecture=NN_ARCHITECTURE,
        epochs=10000,
        learning_rate=0.01,
        verbose=False,
        callback=callback_numpy_plot,
        which="numpy"
    )

    prediction_probs_numpy, _ = full_forward_propagation(np.transpose(grid_2d), params_values, NN_ARCHITECTURE)
    prediction_probs_numpy = prediction_probs_numpy.reshape(prediction_probs_numpy.shape[1], 1)
    make_plot(x_test, y_test, "NumPy Model", file_name=None, XX=XX, YY=YY, preds=prediction_probs_numpy)


def use_tensorflow():
    # Training with Tensorflow
    log_dir = "logs/fit/"
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    model = build_mlp(NN_ARCHITECTURE, learning_rate=0.01)
    model.fit(
        x_train,
        y_train,
        epochs=30,
        batch_size=128,
        validation_split=0.2,  # Does not exist in the NumPy version
        verbose=2,
        callbacks=tensorboard_callback
    )
    print(model.evaluate(x_test, y_test))


if __name__ == '__main__':
    args = parser.parse_args()
    engine = args.engine
    if engine == "np" or engine == "numpy":
        use_numpy()
    elif engine == "tf" or engine == "tensorflow":
        use_tensorflow()
    else:
        raise ValueError(f"Invalid engine {engine}")
