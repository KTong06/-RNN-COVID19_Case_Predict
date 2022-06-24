![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Spyder](https://img.shields.io/badge/Spyder-838485?style=for-the-badge&logo=spyder%20ide&logoColor=maroon)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
<a><img alt='love' src="http://ForTheBadge.com/images/badges/built-with-love.svg"></a>

# [RNN] What's for TOMORROW? -COVID19_Case_Predict-
As slow as it takes I am finally pick up a dying trend (or maybe not) : COVID-19 Case Prediction Model! Predictions are important especially when it comes to anticipating a crisis e.g. food supply shortage, disease breakout, earthquake/volcanic eruptions etc. [Dataset](https://github.com/KTong06/-RNN-COVID19_Case_Predict/tree/main/dataset) is readily available.

# Model Performance
Model is able to achieve **Mean Absolute Percentage Error (MAPE)** of value **0.14**:

|             Actual vs. Predicted             |           Loss and MAPE Plot           |
|----------------------------------------------|----------------------------------------|
| ![eval_test_plot](static/eval_test_plot.png) | ![tensorboard](static/tensorboard.png) |

Evaluation of predicted cases: (MSE-Mean Squared Error, MAE-Mean Absolute Error)

![eval_test](static/eval_test.png)

# Model Architecture
RNN model is constructed using 2 LSTM layers with `activation function='tanh'`, having 64 and 32 nodes respectively:

![model](static/model.png)

## Other hyperparameters
Settings for **optimizer**,**loss function**,**metrics** are summarised below:
```
model.compile(optimizer='adam',loss='mse',metrics=['mse','mae','mape'])

tb=TensorBoard(log_dir=LOG_PATH)

hist=model.fit(x_train,y_train,batch_size=64,epochs=100,callbacks=tb,verbose=1)
```
# Data Summary
Dataset train and test files prepared separately. 




