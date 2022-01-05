import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def init_layers(nn_architecture, seed=99):
    # random seed initiation
    np.random.seed(seed)
    # parameters storage initiation
    weights_and_biases = {}

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1

        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        # initiating the values of the W matrix
        # and vector b for subsequent layers
        weights_and_biases[f"W{layer_idx}"] = np.random.randn(layer_output_size, layer_input_size) * 0.1
        weights_and_biases[f"b{layer_idx}"] = np.random.randn(layer_output_size, 1) * 0.1

    return weights_and_biases


def sigmoid(a):
    return 1 / (1 + np.exp(-a))


def relu(a):
    return np.maximum(0, a)


def sigmoid_derivated(dxtild, a):
    sig = sigmoid(a)
    return dxtild * sig * (1 - sig)  # dLoss/dh * dh/dz


def relu_derivated(dxtild, a):
    dx = np.array(dxtild, copy=True)
    dx[a <= 0] = 0
    return dx


def single_layer_forward_propagation(xtild_prev, w_curr, b_curr, activation="relu"):
    # calculation of the input value for the activation function
    a_curr = np.dot(w_curr, xtild_prev) + b_curr

    # selection of activation function
    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
    else:
        raise ValueError("Non-supported activation function")

    # return of calculated activation A and the intermediate A matrix
    xtild_curr = activation_func(a_curr)
    return xtild_curr, a_curr


def full_forward_propagation(x, weights_and_biases, nn_architecture):
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0
    xtild_curr = x

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        xtild_prev = xtild_curr

        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extraction of W for the current layer
        w_curr = weights_and_biases[f"W{layer_idx}"]
        # extraction of b for the current layer
        b_curr = weights_and_biases[f"b{layer_idx}"]
        # calculation of activation for the current layer
        xtild_curr, a_curr = single_layer_forward_propagation(xtild_prev, w_curr, b_curr, activ_function_curr)

        # saving calculated values in the memory
        memory[f"xtild{idx}"] = xtild_prev
        memory[f"A{layer_idx}"] = a_curr

    # return of prediction vector and a dictionary containing intermediate values
    return xtild_curr, memory


def bce_loss(y_hat, y):
    """ Binary crossentropy """
    # number of examples
    n = y_hat.shape[1]
    # calculation of the loss according to the formula
    loss = -1 / n * (np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T))
    return np.squeeze(loss)


def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def get_accuracy_value(y_hat, y):
    y_hat_ = convert_prob_into_class(y_hat)
    # noinspection PyUnresolvedReferences
    return (y_hat_ == y).all(axis=0).mean()


def single_layer_backward_propagation(dxtild_curr, w_curr, a_curr, xtild_prev, activation="relu"):
    # number of examples
    n = xtild_prev.shape[1]

    # selection of activation function
    if activation == "relu":
        activation_func_derivated = relu_derivated
    elif activation == "sigmoid":
        activation_func_derivated = sigmoid_derivated
    else:
        raise ValueError("Non-supported activation function")

    # calculation of the activation function derivative
    da_curr = activation_func_derivated(dxtild_curr, a_curr)

    # derivative of the matrix W
    dw_curr = np.dot(da_curr, xtild_prev.T) / n
    # derivative of the vector b
    db_curr = np.sum(da_curr, axis=1, keepdims=True) / n
    # derivative of the matrix h_prev
    dxtild_prev = np.dot(w_curr.T, da_curr)

    return dxtild_prev, dw_curr, db_curr


def full_backward_propagation(y_hat, y, memory, weights_and_biases, nn_architecture):
    grads_values = {}

    # a hack ensuring the same shape of the prediction vector and labels vector
    y = y.reshape(y_hat.shape)

    # Initiation of gradient descent algorithm (derivate of the BCE against y_hat (xtild of last layer))
    dxtild_prev = -(np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]

        dxtild_curr = dxtild_prev

        xtild_prev = memory[f"xtild{layer_idx_prev}"]
        a_curr = memory[f"A{layer_idx_curr}"]

        w_curr = weights_and_biases[f"W{layer_idx_curr}"]

        dxtild_prev, dw_curr, db_curr = single_layer_backward_propagation(
            dxtild_curr, w_curr, a_curr, xtild_prev, activ_function_curr
        )

        grads_values[f"dW{layer_idx_curr}"] = dw_curr
        grads_values[f"db{layer_idx_curr}"] = db_curr

    return grads_values


def update(weights_and_biases, grads_values, nn_architecture, learning_rate):

    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        weights_and_biases[f"W{layer_idx}"] -= learning_rate * grads_values[f"dW{layer_idx}"]
        weights_and_biases[f"b{layer_idx}"] -= learning_rate * grads_values[f"db{layer_idx}"]

    return weights_and_biases


def train(x, y, nn_architecture, epochs, learning_rate, verbose=False, callback=None, which=None):
    # initiation of neural net parameters
    weights_and_biases = init_layers(nn_architecture, 2)
    # initiation of lists storing the history
    # of metrics calculated during the learning process
    loss_history = []
    accuracy_history = []

    # performing calculations for subsequent iterations
    for i in range(epochs):
        # step forward
        y_hat, cache = full_forward_propagation(x, weights_and_biases, nn_architecture)

        # calculating metrics and saving them in history
        loss = bce_loss(y_hat, y)
        loss_history.append(loss)
        accuracy = get_accuracy_value(y_hat, y)
        accuracy_history.append(accuracy)

        # step backward - calculating gradient
        grads_values = full_backward_propagation(y_hat, y, cache, weights_and_biases, nn_architecture)
        # updating model state
        weights_and_biases = update(weights_and_biases, grads_values, nn_architecture, learning_rate)

        if i % 50 == 0:
            if verbose:
                print("Iteration: {:05} - loss: {:.5f} - accuracy: {:.5f}".format(i, loss, accuracy))
            if callback is not None:  # For graph
                callback(i, weights_and_biases, which)

    return weights_and_biases


# noinspection PyUnresolvedReferences
def make_plot(x, y, plot_name, file_name=None, xx=None, yy=None, preds=None):
    plt.figure(figsize=(16, 12))
    axes = plt.gca()
    axes.set(xlabel="$X_1$", ylabel="$X_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if xx is not None and yy is not None and preds is not None:
        plt.contourf(xx, yy, preds.reshape(xx.shape), 25, alpha=1, cmap=cm.Spectral)
        plt.contour(xx, yy, preds.reshape(xx.shape), levels=[0.5], cmap="Greys", vmin=0, vmax=0.6)
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors="black")
    if file_name:
        plt.savefig(file_name)
        plt.close()
