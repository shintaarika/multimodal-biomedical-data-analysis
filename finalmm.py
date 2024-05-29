import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #D8BFD8;
}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Define the ANN model
class ANNModel(nn.Module):
    def __init__(self):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(30, 30)
        self.fc2 = nn.Linear(30, 15)
        self.fc3 = nn.Linear(15, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def train_model(X_train, y_train, X_test, y_test, num_epochs=50):
    model = ANNModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        with torch.no_grad():
            train_predictions = (outputs > 0.5).float()
            train_accuracy = accuracy_score(y_train_tensor, train_predictions)
            train_accuracies.append(train_accuracy)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)
            val_losses.append(val_loss.item())
            
            val_predictions = (val_outputs > 0.5).float()
            val_accuracy = accuracy_score(y_test_tensor, val_predictions)
            val_accuracies.append(val_accuracy)
        if (epoch + 1) % 10 == 0:
            st.write(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        
    end_time = time.time()
    total_training_time = end_time - start_time
    
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
        y_pred = (y_pred_tensor > 0.5).float()

    test_accuracy = accuracy_score(y_test_tensor, y_pred)
    cm = confusion_matrix(y_test_tensor, y_pred)
    cr = classification_report(y_test_tensor, y_pred)

    return (train_accuracies, val_accuracies, train_losses, val_losses, 
            total_training_time, test_accuracy, cm, cr)

# Streamlit UI
st.title("Breast Cancer Classification with ANN")
st.write("""
_Upload a CSV file containing breast cancer data, and click the "Train Model" button to train an Artificial Neural Network (ANN) on the data. The model uses the Adam optimizer._
""")
st.sidebar.write("Kelompok 2 : ")
st.sidebar.write("1. Ayuning Sekar Agriensyah Putri (5023211028) ")
st.sidebar.write("2. Farhan Majid Ibrahim (5023211049) ")
st.sidebar.write("3. Narika Shinta (5023211057)")
st.sidebar.write("4. Ratna Indriani (5023211064)")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Data Preview:")
    st.write(data.head())

    if st.button("Train Model"):
        data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

        X = data.drop('diagnosis', axis=1)
        y = data['diagnosis']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        (train_accuracies, val_accuracies, train_losses, val_losses, 
         total_training_time, test_accuracy, cm, cr) = train_model(X_train_scaled, y_train, X_test_scaled, y_test)

        st.write(f"Total Training Time: {total_training_time:.2f} seconds")
        st.write(f"Test Accuracy: ({test_accuracy:.4f}*100)%")

        st.write("Classification Report:")
        st.text(cr)

        st.write("Confusion Matrix:")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        st.write("Accuracy and Loss over Epochs:")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(train_accuracies, label='Train Accuracy')
        ax1.plot(val_accuracies, label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        ax2.plot(train_losses, label='Train Loss')
        ax2.plot(val_losses, label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        st.pyplot(fig)
