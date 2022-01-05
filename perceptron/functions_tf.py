from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def build_mlp(nn_architecture, learning_rate):
    model = Sequential()
    for layer in nn_architecture:
        model.add(Dense(layer["output_dim"], input_dim=layer["input_dim"], activation=layer["activation"]))
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )
    return model
