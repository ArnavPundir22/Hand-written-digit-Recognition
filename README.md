<p align="center">
  <img src="asset/banner.jpg" alt="Handwritten Digit Recognition Banner" width="100%">
</p>

# ✍️ Handwritten Digit Recognition using Deep Neural Networks

A complete **Handwritten Digit Recognition System** built using **TensorFlow/Keras**, capable of training a **Deep Neural Network (DNN)** on handwritten digit images and predicting digits from new images with high confidence.

This project demonstrates an **end-to-end machine learning pipeline** — from data preprocessing and augmentation to training, evaluation, and real-world prediction.

---

## 🚀 Features

- 🔢 Recognizes handwritten digits **(0–9)**
- 🧠 Deep Neural Network (DNN) architecture
- 📈 Data augmentation for better generalization
- 📊 Accuracy & loss visualization
- 🧾 Confusion matrix & classification report
- 💾 Automatic model checkpointing
- 🖼️ Predict digits from custom images with confidence %

---

## 🧠 Model Architecture

```
Input (28×28 grayscale)
↓
Flatten
↓
Dense (512) + Dropout (0.5)
↓
Dense (256) + Dropout (0.4)
↓
Dense (128) + Dropout (0.3)
↓
Dense (10) + Softmax
```

---

## 📂 Project Structure

```
Hand-written-digit-Recognition/
│
├── dataset_emnist/
│   ├── train/
│   └── test/
│
├── outputs/
│   ├── model/
│   │   ├── epochs/
│   │   ├── best_model.h5
│   │   └── final_model.h5
│   ├── plots/
│   ├── evaluation/
│   └── predictions/
│
├── dig.py            # Model training & evaluation
├── pridict.py        # Digit prediction script
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/ArnavPundir22/Hand-written-digit-Recognition.git
cd Hand-written-digit-Recognition
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Training the Model

```bash
python dig.py
```

This will:
- Train the DNN model
- Save all epoch checkpoints
- Save the best-performing model
- Generate accuracy & loss plots
- Create confusion matrix & classification report
- Save sample prediction images

---

## 🔍 Predicting a Digit

```bash
python pridict.py
```

### Example Output
```
Predicted Digit: 5
Confidence: 98.34%
```

---

## 📊 Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision, Recall, F1-Score
- Sample Prediction Visualization

All evaluation outputs are saved inside:
```
outputs/evaluation/
```

---

## 🧪 Dataset

- EMNIST-style directory structure
- Separate training and testing folders
- Grayscale handwritten digit images
- Data augmentation enabled

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## 🌟 Future Enhancements

- CNN-based architecture for higher accuracy
- Web interface using Flask or Streamlit
- Real-time digit drawing canvas
- API or mobile deployment

---

⭐ **If you find this project useful, consider starring the repository!**
