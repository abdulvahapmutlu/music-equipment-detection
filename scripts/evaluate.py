import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from preprocess import load_data

data_dir = 'datasets'
x, y = load_data(data_dir)

model = load_model('final_model.h5')

y_pred = np.argmax(model.predict(x), axis=-1)

conf_matrix = confusion_matrix(y, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('results/confusion_matrix.png')

print(classification_report(y, y_pred, target_names=os.listdir(data_dir)))
