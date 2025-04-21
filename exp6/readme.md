
## Objective
WAP to train and evaluate a Recurrent Neural Network using PyTorch Library to predict the next value in a sample time series dataset.

---

## Model Description

The model is a simple RNN-based time series regressor implemented in **PyTorch**. It consists of:
- An **RNN layer** with 128 hidden units to capture temporal dependencies.
- A **Dropout layer** (20%) to reduce overfitting.
- A **Fully Connected Linear layer** to map RNN outputs to final predictions.

The model is trained to minimize **Mean Squared Error (MSE)** between predicted and actual milk production values.

---

## Code Description

### Data Preprocessing:
- The milk dataset (`milk.csv`) is read and cleaned.
- Only the `Milk_Production` column is used.
- Data is normalized using `MinMaxScaler` (0 to 1).
- Time series sequences of length 12 are generated for supervised learning (i.e., previous 12 months → next month's prediction).

### Model Training:
- Input data is reshaped to fit RNN expectations: `(batch_size, sequence_length, input_size)`.
- Model is trained using the **Adam optimizer** for 100 epochs.
- At each epoch:
  - MSE loss is computed.
  - A custom accuracy metric is calculated using **Mean Absolute Error (MAE)** to better reflect real-world forecasting error.

### Visualization:
- Three plots are generated:
  1. **Actual vs Predicted Milk Production**
  2. **Loss per Epoch**
  3. **Accuracy per Epoch**

---

## Performance Evaluation

| Metric             | Value          |
|--------------------|----------------|
| Final MSE Loss     | `817.4564`     |
| Final Accuracy     | `96.93%`       |

**Training Metrics Snapshot:**
- Loss reduced from `0.0379` at Epoch 10 to `0.0072` at Epoch 100.
- Accuracy improved from `91.35%` to `96.93%`.
- Final Custom Accuracy: `96.93%`

These results demonstrate the model's ability to effectively learn and forecast the time series data.

---

## My Comment

This project gave me hands-on experience with:
- LSTM or GRU may improve performance
- I understood how RNNs process sequential data and how to implement them using PyTorch.




