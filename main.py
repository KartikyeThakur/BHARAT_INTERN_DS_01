import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os
# Load and preprocess the dataset
url = r'C:\Users\KARTIKYE\Downloads\spam.csv'
raw_mail_data = pd.read_csv(url, encoding='latin-1')
raw_mail_data = raw_mail_data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'])
raw_mail_data.columns = ['Category', 'Message']
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# Label spam mail as 0; ham mail as 1
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 1

spam_messages = mail_data[mail_data['Category'] == 0]
ham_messages = mail_data[mail_data['Category'] == 1]

# Check if the model and vectorizer are already saved
model_path = 'sms_spam_logistic_regression.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    # Split the data into training and testing sets
    X = mail_data['Message']
    Y = mail_data['Category']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    # Transform the text data to feature vectors
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)

    # Convert Y_train and Y_test values to integers
    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')

    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_features, Y_train)

    # Save the model and vectorizer
    joblib.dump(model, model_path)
    joblib.dump(feature_extraction, vectorizer_path)
else:
    # Load the saved model and vectorizer
    model = joblib.load(model_path)
    feature_extraction = joblib.load(vectorizer_path)

# Function to predict if the message is spam or ham
def predict_message():
    message = entry_message.get()
    input_data_features = feature_extraction.transform([message])
    prediction = model.predict(input_data_features)
    if prediction[0] == 1:
        result.set('Ham')
    else:
        result.set('Spam')

# Function to show spam messages
def show_spam():
    message_list.delete(0, tk.END)
    for index, row in spam_messages.iterrows():
        message_list.insert(tk.END, f"{index}: {row['Message']}")

# Function to show ham messages
def show_ham():
    message_list.delete(0, tk.END)
    for index, row in ham_messages.iterrows():
        message_list.insert(tk.END, f"{index}: {row['Message']}")

# Function to export spam messages to a CSV file
def export_spam():
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        spam_messages.to_csv(file_path, index=True)
        messagebox.showinfo("Export", "Spam messages exported successfully!")

# Function to export ham messages to a CSV file
def export_ham():
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if file_path:
        ham_messages.to_csv(file_path, index=True)
        messagebox.showinfo("Export", "Ham messages exported successfully!")

# Create the main window
root = tk.Tk()
root.title('SMS Spam Classifier')

# Create a frame for message entry and prediction
frame_entry = tk.Frame(root, bg='#f0f0f0', padx=10, pady=10)
frame_entry.pack(padx=20, pady=10, fill=tk.X)

label_message = tk.Label(frame_entry, text='Enter Message:', bg='#f0f0f0', font=('Arial', 12))
label_message.pack(side=tk.LEFT, padx=5)

entry_message = tk.Entry(frame_entry, width=50, font=('Arial', 12))
entry_message.pack(side=tk.LEFT, padx=5)

button_predict = tk.Button(frame_entry, text='Predict', command=predict_message, bg='#4CAF50', fg='white', font=('Arial', 12))
button_predict.pack(side=tk.LEFT, padx=5)

result = tk.StringVar()
label_result = tk.Label(frame_entry, textvariable=result, bg='#f0f0f0', font=('Arial', 12))
label_result.pack(side=tk.LEFT, padx=5)

# Create a frame for message list and buttons
frame_messages = tk.Frame(root, bg='#f0f0f0', padx=10, pady=10)
frame_messages.pack(padx=20, pady=10, fill=tk.X)

message_list = tk.Listbox(frame_messages, width=100, height=20, font=('Arial', 12))
message_list.pack(side=tk.LEFT, padx=5)

scrollbar = tk.Scrollbar(frame_messages, orient="vertical")
scrollbar.config(command=message_list.yview)
scrollbar.pack(side=tk.LEFT, fill="y")

message_list.config(yscrollcommand=scrollbar.set)

button_spam = tk.Button(root, text='Show Spam', command=show_spam, bg='#f44336', fg='white', font=('Arial', 12))
button_spam.pack(side=tk.LEFT, padx=5, pady=5)

button_ham = tk.Button(root, text='Show Ham', command=show_ham, bg='#2196F3', fg='white', font=('Arial', 12))
button_ham.pack(side=tk.LEFT, padx=5, pady=5)

button_export_spam = tk.Button(root, text='Export Spam', command=export_spam, bg='#FFC107', fg='black', font=('Arial', 12))
button_export_spam.pack(side=tk.LEFT, padx=5, pady=5)

button_export_ham = tk.Button(root, text='Export Ham', command=export_ham, bg='#FFC107', fg='black', font=('Arial', 12))
button_export_ham.pack(side=tk.LEFT, padx=5, pady=5)

# Run the application
root.mainloop()
