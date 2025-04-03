# Handwritten Digit Recognition

The goal is to create a deep learning model to classify handwritten digits (0–9) using images. Using the famous MNIST Dataset, available directly in TensorFlow/Keras.

## Table of Contents
- [Previous Configurations](#previous-configurations)
- [Data Preprocessing](#data-preprocessing)
- [Model](#model)
- [Contributing](#contributing)
- [License](#license)

## Previous Configurations

In order to run this script, you will need:

- The necessary libraries installed
- A `.env` file with the required environment variables

### Installing the Libraries

The script works with the following versions:
- python `3.11.5`
- numpy `1.25.2`
- tensorflow `2.13.0`
- scikit-learn `1.3.0`

To install the necessary libraries, run:
```sh
pip install -r requirements.txt
```

## Data Preprocessing

The train and test sets are obtained from the `mnist.load_data()` which already loads the necessary data.

Because the pixel values in MNIST images range from 0 to 255 (grayscale), dividing by 255.0 scales the values to the range [0,1], which helps neural networks converge faster and improves numerical stability.

The MNIST dataset consists of 28×28 grayscale images.

The `.reshape(-1, 28, 28, 1)` reformats the data to match the expected input shape of convolutional neural networks (CNNs), where:

- -1 allows automatic calculation of the batch size.

- 28, 28 represents the height and width of each image.

- 1 represents the number of color channels (grayscale).

## Model

For the model, we use a Convolutional Neural Network (CNN).

This CNN processes MNIST images, extracts features using convolution and pooling, flattens the data, and classifies digits using fully connected layers. It is optimized using Adam, and trained for 10 epochs with validation on the test set.

### Model Evaluation

The model's predictions are probability scores (one for each digit 0-9).
`np.argmax(y_pred, axis=1)` selects the index with the highest probability for each image, converting softmax outputs into class labels (digits 0-9).
The condition checks if `y_pred` contains probability distributions before applying `argmax`.

`classification_report(y_test, y_pred)` (from `sklearn.metrics`) computes and prints:

Precision, recall, and F1-score for each digit class.
Overall accuracy, macro/micro averages.
Helps understand how well the model performs across different digits, identifying misclassifications.

The computed results are the following:

| Digit | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.99      | 1.00   | 0.99     | 980     |
| 1     | 0.99      | 0.99   | 0.99     | 1135    |
| 2     | 0.99      | 0.98   | 0.98     | 1032    |
| 3     | 0.99      | 0.98   | 0.99     | 1010    |
| 4     | 0.99      | 0.99   | 0.99     | 982     |
| 5     | 0.99      | 0.98   | 0.99     | 892     |
| 6     | 0.98      | 0.99   | 0.98     | 958     |
| 7     | 0.97      | 0.99   | 0.98     | 1028    |
| 8     | 0.99      | 0.98   | 0.98     | 974     |
| 9     | 0.98      | 0.97   | 0.98     | 1009    |
| **Accuracy** |       |        | **0.99** | 10000   |
| **Macro Avg** | 0.99  | 0.99   | 0.99     | 10000   |
| **Weighted Avg** | 0.99  | 0.99   | 0.99     | 10000   |

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
