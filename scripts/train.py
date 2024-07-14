from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import build_model
from preprocess import load_data
from keras_tuner import HyperParameters

data_dir = 'datasets'
x, y = load_data(data_dir)

hp = HyperParameters()
hp.Int('units_1', min_value=32, max_value=512, step=32)
hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)
hp.Int('units_2', min_value=32, max_value=512, step=32)
hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)
hp.Int('units_3', min_value=32, max_value=512, step=32)
hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

model = build_model(hp)

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(filepath='best_model.h5', save_best_only=True)
]

history = model.fit(x, y, epochs=20, validation_split=0.2, callbacks=callbacks)

model.save('final_model.h5')
