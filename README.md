# 🧠 Customer Churn Prediction using Deep Learning

This project applies **Deep Learning** techniques to predict whether a bank customer will churn (leave) or stay, using the popular **Churn Modelling dataset**.  
I built and trained an **Artificial Neural Network (ANN)** from scratch with multiple epochs, improving accuracy step by step through backpropagation and optimization.

---

## 📌 Project Overview

Customer churn is a critical problem in the banking and financial sector.  
By applying **deep learning** with an ANN, this project aims to identify customers likely to churn, giving businesses the chance to take proactive measures.

- **Dataset**: Churn_Modelling.csv (10,000 customer records from a bank)
- **Goal**: Predict the `Exited` column (1 = churned, 0 = stayed)
- **Approach**: Build and train an Artificial Neural Network with multiple hidden layers

---

## 📂 Dataset Details

- **Features**:
  - CreditScore
  - Geography (encoded)
  - Gender (encoded)
  - Age
  - Tenure
  - Balance
  - NumOfProducts
  - HasCrCard
  - IsActiveMember
  - EstimatedSalary

- **Dropped Columns**:
  - RowNumber, CustomerId, Surname (not useful for prediction)

- **Target**:
  - `Exited` → Binary classification (0 = Not Churned, 1 = Churned)

---

## ⚙️ Methodology

1. **Data Preprocessing**  
   - Encoded categorical features (Geography, Gender)  
   - Dropped irrelevant columns  
   - Normalized numerical values  

2. **Deep Learning Model (ANN)**  
   - Input Layer: Encoded & scaled features  
   - Hidden Layers: Dense layers with ReLU activation  
   - Output Layer: Sigmoid activation for binary classification  

3. **Training**  
   - Optimizer: Adam  
   - Loss Function: Binary Crossentropy  
   - Batch size: 32  
   - Epochs: 50+ (tuned for best accuracy)  

4. **Evaluation**  
   - Training accuracy improved with each epoch  
   - Achieved strong accuracy on validation/test data  

---

## 📊 Results

- **Model Accuracy**: ~85% (varies by run and hyperparameters)  
- **Loss Curve**: Loss decreased steadily across epochs  
- **Key Insight**: Deep learning ANN successfully captures complex patterns in customer behavior  

---

## 🚀 Tech Stack

- **Language**: Python  
- **Libraries**:  
  - NumPy  
  - Pandas  
  - Matplotlib  
  - TensorFlow / Keras  
  - Scikit-learn  

---

## 🖥️ Example Training Output

Epoch 1/50

loss: 0.64 - accuracy: 0.65
Epoch 50/50

loss: 0.34 - accuracy: 0.85
