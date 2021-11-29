import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def init_layers(nn_architecture, seed=99):
    # random seed initiation
    np.random.seed(seed)
    # parameters storage initiation
    params_values = {}

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1

        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        # initiating the values of the W matrix
        # and vector b for subsequent layers
        params_values[f"W{layer_idx}"] = np.random.randn(layer_output_size, layer_input_size) * 0.1
        params_values[f"b{layer_idx}"] = np.random.randn(layer_output_size, 1) * 0.1

    return params_values


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def sigmoid_backward(dh, z):
    sig = sigmoid(z)
    return dh * sig * (1 - sig)  # dLoss/dh * dh/dz


def relu_backward(dh, z):
    dx = np.array(dh, copy=True)
    dx[z <= 0] = 0
    return dx


def single_layer_forward_propagation(a_prev, w_curr, b_curr, activation="relu"):
    # calculation of the input value for the activation function
    z_curr = np.dot(w_curr, a_prev) + b_curr

    # selection of activation function
    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception("Non-supported activation function")

    # return of calculated activation A and the intermediate Z matrix
    return activation_func(z_curr), z_curr


def full_forward_propagation(x, params_values, nn_architecture):
    # creating a temporary memory to store the information needed for a backward step
    memory = {}
    # X vector is the activation for layer 0
    a_curr = x

    # iteration over network layers
    for idx, layer in enumerate(nn_architecture):
        # we number network layers from 1
        layer_idx = idx + 1
        # transfer the activation from the previous iteration
        a_prev = a_curr

        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]
        # extraction of W for the current layer
        w_curr = params_values[f"W{layer_idx}"]
        # extraction of b for the current layer
        b_curr = params_values[f"b{layer_idx}"]
        # calculation of activation for the current layer
        a_curr, z_curr = single_layer_forward_propagation(a_prev, w_curr, b_curr, activ_function_curr)

        # saving calculated values in the memory
        memory[f"h{idx}"] = a_prev
        memory[f"Z{layer_idx}"] = z_curr

    # return of prediction vector and a dictionary containing intermediate values
    return a_curr, memory


def bce_loss(y_hat, y):
    """ Binary crossentropy """
    # number of examples
    n = y_hat.shape[1]
    # calculation of the cost according to the formula (NOT the same than in the slides)
    cost = -1 / n * (np.dot(y, np.log(y_hat).T) + np.dot(1 - y, np.log(1 - y_hat).T))
    return np.squeeze(cost)


def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    # noinspection PyUnresolvedReferences
    return (Y_hat_ == Y).all(axis=0).mean()


def single_layer_backward_propagation(dh_curr, w_curr, z_curr, h_prev, activation="relu"):
    # number of examples
    n = h_prev.shape[1]

    # selection of activation function
    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception("Non-supported activation function")

    # calculation of the activation function derivative
    dz_curr = backward_activation_func(dh_curr, z_curr)

    # derivative of the matrix W
    dw_curr = np.dot(dz_curr, h_prev.T) / n
    # derivative of the vector b
    db_curr = np.sum(dz_curr, axis=1, keepdims=True) / n
    # derivative of the matrix h_prev
    dh_prev = np.dot(w_curr.T, dz_curr)

    return dh_prev, dw_curr, db_curr


def full_backward_propagation(y_hat, y, memory, params_values, nn_architecture):
    grads_values = {}

    # a hack ensuring the same shape of the prediction vector and labels vector
    y = y.reshape(y_hat.shape)

    # Initiation of gradient descent algorithm (derivate of the BCE against y_hat (h of last layer))
    # NOT the same than the slides, because not the same loss
    dh_prev = -(np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        # we number network layers from 1
        layer_idx_curr = layer_idx_prev + 1
        # extraction of the activation function for the current layer
        activ_function_curr = layer["activation"]

        dh_curr = dh_prev

        h_m1 = memory[f"h{layer_idx_prev}"]  # Derivate of h(layer_idx_prev-1)*W(layer_idx_prev) by W(layer_idx_prev)
        z_curr = memory[f"Z{layer_idx_curr}"]

        w_curr = params_values[f"W{layer_idx_curr}"]

        dh_prev, dw_curr, db_curr = single_layer_backward_propagation(
            dh_curr, w_curr, z_curr, h_m1, activ_function_curr
        )

        grads_values[f"dW{layer_idx_curr}"] = dw_curr
        grads_values[f"db{layer_idx_curr}"] = db_curr

    return grads_values


def update(params_values, grads_values, nn_architecture, learning_rate):

    # iteration over network layers
    for layer_idx, layer in enumerate(nn_architecture, 1):
        params_values[f"W{layer_idx}"] -= learning_rate * grads_values[f"dW{layer_idx}"]
        params_values[f"b{layer_idx}"] -= learning_rate * grads_values[f"db{layer_idx}"]

    return params_values


def train(x, y, nn_architecture, epochs, learning_rate, verbose=False, callback=None, which=None):
    # initiation of neural net parameters
    params_values = init_layers(nn_architecture, 2)
    # initiation of lists storing the history
    # of metrics calculated during the learning process
    cost_history = []
    accuracy_history = []

    # performing calculations for subsequent iterations
    for i in range(epochs):
        # step forward
        y_hat, cache = full_forward_propagation(x, params_values, nn_architecture)

        # calculating metrics and saving them in history
        cost = bce_loss(y_hat, y)
        cost_history.append(cost)
        accuracy = get_accuracy_value(y_hat, y)
        accuracy_history.append(accuracy)

        # step backward - calculating gradient
        grads_values = full_backward_propagation(y_hat, y, cache, params_values, nn_architecture)
        # updating model state
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)

        if i % 50 == 0:
            if verbose:
                print("Iteration: {:05} - cost: {:.5f} - accuracy: {:.5f}".format(i, cost, accuracy))
            if callback is not None:  # For graph
                callback(i, params_values, which)

    return params_values


# noinspection PyUnresolvedReferences
def make_plot(x, y, plot_name, file_name=None, XX=None, YY=None, preds=None):
    plt.figure(figsize=(16, 12))
    axes = plt.gca()
    axes.set(xlabel="$X_1$", ylabel="$X_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if XX is not None and YY is not None and preds is not None:
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=1, cmap=cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[0.5], cmap="Greys", vmin=0, vmax=0.6)
    plt.scatter(x[:, 0], x[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors="black")
    if file_name:
        plt.savefig(file_name)
        plt.close()
