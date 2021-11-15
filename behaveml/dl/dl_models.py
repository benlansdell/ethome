from keras.models import Sequential
import keras.layers as layers
import keras

#from .fscores import F1Score

def build_baseline_model(input_dim, layer_channels=(512, 256), dropout_rate=0., 
                        learning_rate=1e-3, conv_size=5, num_classes = 4,
                        class_weight = None):

    print("Building baseline 1D CNN model with parameters:")
    print(f"dropout_rate: {dropout_rate}, learning_rate: {learning_rate}, layer_channels: {layer_channels}, conv_size: {conv_size}")

    def add_dense_bn_activate(model, out_dim, activation='relu', drop=0.):
        model.add(layers.Dense(out_dim))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        if drop > 0:
            model.add(layers.Dropout(rate=drop))
        return model

    def add_conv_bn_activate(model, out_dim, activation='relu', conv_size=3, drop=0.):
        model.add(layers.Conv1D(out_dim, conv_size))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation('relu'))
        model.add(layers.MaxPooling1D(2, 2))
        if drop > 0:
            model.add(layers.Dropout(rate=drop))
        return model

    model = Sequential()
    model.add(layers.Input(input_dim))
    model.add(layers.BatchNormalization())
    for ch in layer_channels:
        model = add_conv_bn_activate(model, ch, conv_size=conv_size,
                                     drop=dropout_rate)
    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, activation='softmax'))

    #metrics = [F1Score(num_classes=num_classes, average = 'macro', labels = [0,1,2]), 'accuracy']
    metrics = []
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', 
                optimizer=optimizer,
                metrics=metrics)

    return model 