# Timeseries-data-tensorflow-using-CNN
Train a simple Tensorflow CNN model for timeseries data. 

## Results

Results added in a pickle file 'trainres'.

Load pickle file with the following code to view the train logs.

```
with open('trainres', "rb") as file_pi:
    history = pickle.load(file_pi)
    print(history)
```