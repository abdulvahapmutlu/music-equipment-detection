import kerastuner as kt
from model import build_model
from preprocess import load_data

data_dir = 'datasets'
x, y = load_data(data_dir)

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    directory='tuner_dir',
    project_name='music_equipment_tuning'
)

tuner.search(x, y, epochs=10, validation_split=0.2)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"The optimal number of units in the first dense layer is {best_hps.get('units_1')}")
