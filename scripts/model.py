from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import kerastuner as kt

def build_model(hp):
    model = Sequential()
    model.add(Dense(units=hp.Int('units_layer_1', min_value=64, max_value=512, step=64), input_shape=(40,), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_rate_1', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('units_layer_2', min_value=64, max_value=512, step=64), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_rate_2', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(units=hp.Int('units_layer_3', min_value=32, max_value=256, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout_rate_3', min_value=0.0, max_value=0.5, step=0.1)))
    model.add(Dense(28, activation='softmax'))
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
