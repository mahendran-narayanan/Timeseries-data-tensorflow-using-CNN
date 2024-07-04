import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

data = {
    'time': pd.date_range(start='2024-07-01', periods=20, freq='D'),
    'value': [0.5, 0.6, 0.8, 0.9, 1.0, 1.2, 1.4, 1.5, 1.7, 1.8, 2.0, 2.1, 2.3, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6]
}
data = pd.DataFrame(data)

def window(data, look_back=1):
    X, y = [], []
    for i in range(len(data) - look_back):
        a = data[i:(i + look_back)]
        X.append(a)
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

look_back = 3
X, y = window(data['value'].values, look_back)

X = X.reshape((X.shape[0], X.shape[1], 1))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(look_back, 1)))
model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(50, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(X, y, epochs=250, verbose=1)

with open('trainres', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

loss = model.evaluate(X, y, verbose=0)
print(f'Model Loss: {loss}')