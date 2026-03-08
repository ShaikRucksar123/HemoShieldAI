# 🩸 HemoShield AI

### Hybrid Deep Learning Based Blood Cancer Detection System

HemoShield AI is a hybrid deep learning web application that predicts the type of blood cancer using:

* 🔬 Microscopic Blood Smear Images (CNN - MobileNetV2)
* 📊 Clinical Parameters (Dense Neural Network)

The system combines image-based feature extraction and structured clinical data analysis to perform multi-class blood cancer classification.

---

## 🚀 Features

* ✅ CNN-based image classification (MobileNetV2)
* ✅ Dense Neural Network for clinical data
* ✅ Hybrid multimodal learning (Image + Clinical)
* ✅ Flask-based web interface
* ✅ Real-time prediction with confidence score
* ✅ Input validation (medical range protection)
* ✅ Dashboard with Accuracy & F1 visualization
* ✅ Drag & Drop image upload
* ✅ Secure backend validation

---

## 🧠 Model Architecture

The system uses a hybrid architecture:

Image Input (160x160x3)
→ MobileNetV2 (CNN Feature Extractor)
→ Dense Layer

Clinical Data Input (6 Features)
→ Dense Neural Network

Concatenation
→ Fully Connected Layer
→ Softmax Output Layer (Multi-class Classification)

---

## 🏥 Supported Classes

* ALL – Acute Lymphoblastic Leukemia
* AML – Acute Myeloid Leukemia
* CLL – Chronic Lymphocytic Leukemia
* CML – Chronic Myeloid Leukemia
* FL – Follicular Lymphoma
* Healthy

---

## 📂 Project Structure

```
HemoShield-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   └── saved/
│       ├── combined_model_fast.h5
│       ├── scaler.pkl
│       └── class_names.pkl
│
├── static/
│   ├── css/
│   ├── js/
│   └── uploads/
│
├── templates/
│   ├── index.html
│   └── dashboard.html
│
└── scripts/
    └── train_model.py
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/HemoShield-AI.git
cd HemoShield-AI
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000
```

---

## 📊 Model Performance

* Accuracy: ~94.6%
* F1 Score: ~95.8%
* Training Time: < 20 minutes
* Dataset Size: Hybrid (Image + Clinical)

---

## 🔒 Clinical Input Validation

The system validates medical parameters:

| Parameter  | Allowed Range |
| ---------- | ------------- |
| WBC        | 1 – 200       |
| RBC        | 1 – 10        |
| Platelets  | 10 – 1500     |
| Hemoglobin | 3 – 25        |
| Blasts     | 0 – 100       |
| Age        | 0 – 120       |

Both frontend and backend validation are implemented.

---

## 📈 Dashboard

The dashboard displays:

* Model Accuracy
* F1 Score
* Training Data Size
* Accuracy Curve
* F1 Score Curve

---

## 🧪 Technologies Used

* Python
* Flask
* TensorFlow / Keras
* MobileNetV2 (Transfer Learning)
* NumPy
* Scikit-learn
* OpenCV
* HTML, CSS, JavaScript
* Chart.js

---

## 🎓 Academic Value

This project demonstrates:

* Deep Learning (CNN + Dense)
* Transfer Learning
* Multimodal Data Fusion
* Medical AI System Design
* Model Deployment with Flask
* Secure AI Input Validation

---

## ⚠️ Disclaimer

This system is for educational and research purposes only.
It is not a substitute for professional medical diagnosis.

---

## 👨‍💻 Author

Developed as a Deep Learning Academic Project.

---

If you found this project useful, consider giving it a ⭐ on GitHub.
