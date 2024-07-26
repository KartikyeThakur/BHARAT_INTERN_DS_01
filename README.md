# BHARAT_INTERN_DS_01
The SMS Spam Classifier is a Python-based application that classifies text messages as either spam or ham using a logistic regression model

# SMS Spam Classifier


The SMS Spam Classifier is a Python-based application designed to classify text messages as either spam or ham using a logistic regression model. This application leverages advanced text classification techniques to provide an intuitive and interactive experience for users, allowing them to manage and analyze SMS messages effectively..

## Features

- **Message Classification:** Classifies input messages as spam or ham using a trained logistic regression model.
- **Message Display:** Displays spam and ham messages in a list with their indices from the original dataset.
- **Export Functionality:** Exports classified spam and ham messages to separate CSV files.
- **User-Friendly GUI:** Provides an intuitive interface for message entry and classification with design enhancements.

## Technologies Used

- **Python 3.7+**
- **Tkinter:** For GUI creation.
- **Pandas:** For data manipulation and export.
- **Scikit-Learn:** For machine learning model training and evaluation.
- **Joblib:** For saving and loading the trained model and vectorizer.

## Dataset

The dataset used for training and testing the model is `spam.csv`. It is taken from Kaggle and contains text messages labeled as spam or ham. The dataset is used to train the logistic regression model for classifying SMS messages.

## File Descriptions

### `sms_spam_classifier.py`

This is the main script that contains all functionalities for:
- Loading and preprocessing the dataset.
- Training and saving the logistic regression model.
- Creating and managing the Tkinter GUI.
- Predicting and displaying message classifications.
- Exporting spam and ham messages to CSV files.

### `spam.csv`

The dataset used for training the model. It contains text messages labeled as spam or ham. The dataset should be placed in the `Downloads` directory or update the path in the script.

## Installation

1. **Clone the Repository:**
    ```sh
    git clone https://github.com/KartikyeThakur/BHARAT_INTERN_DS_01.git
    cd sms_spam_classifier
    ```

2. **Install Required Packages:**
    Ensure you have Python 3.7+ installed. Then, install the required packages using `pip`:
    ```sh
    pip install pandas scikit-learn joblib
    ```

3. **Run the Application:**
    Execute the script using Python:
    ```sh
    python sms_spam_classifier.py
    ```

## Usage

1. **Enter a Message:**
    Type a message into the input field and click the 'Predict' button to classify the message as spam or ham.

2. **View Messages:**
    Click on 'Show Spam' to display all spam messages or 'Show Ham' to display all ham messages. Messages will be shown with their indices from the dataset.

3. **Export Messages:**
    Click on 'Export Spam' to save all spam messages to a CSV file or 'Export Ham' to save all ham messages. A file dialog will appear to select the save location.

## Model Training

The model is trained using logistic regression with TF-IDF vectorization on the message text. If the model and vectorizer files are not found, they will be created and saved automatically.

### Model and Vectorizer Paths
- `sms_spam_logistic_regression.pkl`: The trained logistic regression model.
- `tfidf_vectorizer.pkl`: The TF-IDF vectorizer used for feature extraction.

