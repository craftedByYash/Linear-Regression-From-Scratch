# 🚀 Linear Regression From Scratch in Python

> A complete implementation of Linear Regression from scratch using only Python's built-in libraries — **No NumPy, No Pandas, No Scikit-learn, No External Machine Learning Libraries.**

---

## 📖 Project Overview

This project demonstrates how a complete Machine Learning pipeline can be built from scratch using only Python's standard library.

Instead of relying on machine learning frameworks, every preprocessing step, optimization algorithm, mathematical operation, and prediction function has been implemented manually to understand the internal working of Linear Regression.

The project is designed for students, beginners, and developers who want to learn how Machine Learning algorithms work behind the scenes.

---

# ✨ Features

✅ Automatic CSV Reader

✅ Automatic Feature Type Detection

✅ Missing Value Handling

✅ Duplicate Record Removal

✅ Numerical Feature Processing

✅ Categorical Feature Encoding (One-Hot Encoding)

✅ Polynomial Feature Expansion

✅ Standard Feature Scaling

✅ Target Variable Scaling

✅ Linear Regression From Scratch

✅ Gradient Descent Optimization

✅ Adam Optimizer

✅ Normal Equation Implementation

✅ Cost Function Calculation

✅ Automatic Feature Engineering

✅ Model Serialization

✅ JSON Model Saving

✅ Model Loading

✅ Interactive User Prediction

---

# 📂 Project Structure

```
Linear-Regression-From-Scratch/

│
├── Train_Model.py
├── Load_Model.py
├── house_price_100000.csv
├── linear_regression_model.json
├── README.md
```

---

# 🛠 Technologies Used

- Python 3.x
- JSON
- Math
- Random
- Regular Expressions
- OS Module
- Standard Library Only

No third-party libraries are used.

---

# 🚫 External Libraries Used

This project intentionally **does NOT use**

- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- PyTorch
- SciPy
- OpenCV
- Matplotlib
- Seaborn

Everything is implemented manually.

---

# ⚙️ Machine Learning Pipeline

The training pipeline performs the following steps automatically.

```
CSV Dataset
      │
      ▼
Read Dataset
      │
      ▼
Detect Numeric & Categorical Columns
      │
      ▼
Handle Missing Values
      │
      ▼
Remove Duplicate Records
      │
      ▼
One-Hot Encode Categorical Features
      │
      ▼
Polynomial Feature Expansion
      │
      ▼
Standard Scaling
      │
      ▼
Train Linear Regression
      │
      ▼
Gradient Descent + Adam Optimizer
      │
      ▼
Model Evaluation
      │
      ▼
Save Model as JSON
```

---

# 📊 Supported Features

The training script automatically supports:

- Numeric Features
- Categorical Features
- Polynomial Features
- Interaction Features

No manual preprocessing is required.

---

# 🧮 Mathematical Concepts

This implementation includes manual implementations of:

- Mean Squared Error (MSE)
- Gradient Descent
- Adam Optimizer
- Standard Scaler
- Polynomial Regression Features
- One-Hot Encoding
- Matrix Operations
- Normal Equation
- Cost Function

---

# 📦 Model Saving

After training, the model is saved as:

```
linear_regression_model.json
```

The saved model contains:

- Learned Weights
- Bias
- Standard Scaler Parameters
- Target Scaler Parameters
- Polynomial Degree
- One-Hot Encoding Map
- Feature Metadata
- Target Column Information

---

# 🚀 How to Run

## 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/Linear-Regression-From-Scratch.git
```

---

## 2️⃣ Open Project

```bash
cd Linear-Regression-From-Scratch
```

---

## 3️⃣ Train Model

```bash
python Train_Model.py
```

or

```bash
python Train_Model.py house_price_100000.csv
```

---

## 4️⃣ Load Saved Model

```bash
python Load_Model.py
```

The program will ask for feature values and return the predicted result.

---

# 📈 Example Workflow

```
Dataset
   ↓
Training
   ↓
Feature Engineering
   ↓
Scaling
   ↓
Gradient Descent
   ↓
Model Saved
   ↓
Load Model
   ↓
Predict New Data
```

---

# 🎯 Learning Objectives

This project was built to understand:

- Machine Learning from Scratch
- Feature Engineering
- Data Preprocessing
- Optimization Algorithms
- Linear Regression Mathematics
- Model Serialization
- Prediction Pipeline
- End-to-End ML Workflow

---

# 💡 Why This Project?

Most Linear Regression examples use Scikit-learn in only a few lines of code.

This project shows **how those libraries work internally** by implementing every important component manually.

It is intended for educational purposes and to build a strong understanding of Machine Learning fundamentals.

---

# 📚 Future Improvements

- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Logistic Regression From Scratch
- Decision Tree From Scratch
- Random Forest From Scratch
- Image Classification Models
- Model Visualization
- Cross Validation
- Additional Evaluation Metrics

---

# 👨‍💻 Author

**Yashship**

Machine Learning Enthusiast | Python Developer

Currently building Machine Learning algorithms from scratch to gain a deep understanding of their mathematical foundations and practical implementation.

---

# ⭐ Support

If you found this project useful:

⭐ Star this repository

🍴 Fork it

🛠 Contribute improvements

Sharing feedback and suggestions is always appreciated.

---

# 📜 License

This project is licensed under the MIT License.

Feel free to use, modify, and learn from this implementation.
