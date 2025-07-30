
# 🩺 Anemia Sense: Leveraging Machine Learning for Precise Anemia Recognition

Anemia Sense is a machine learning-based web application designed to predict anemia using key health indicators. Built using Python and scikit-learn, it empowers healthcare professionals and patients by offering a fast, intelligent diagnostic assistant.

---

## 📸 Screenshots

### 🏠 Homepage
<img width="1366" height="1478" alt="homepage" src="https://github.com/user-attachments/assets/603d0961-191a-4086-a1c5-2aef10912355" />


### 🔍 Prediction Page
<img width="1366" height="1182" alt="predict" src="https://github.com/user-attachments/assets/aa294df7-b668-4163-8d35-4867d41b7dcf" />


### 📊 Results Display
<img width="1366" height="1408" alt="predict2" src="https://github.com/user-attachments/assets/18ff64ae-ae6d-4488-9925-9941be6c4b4f" />


---

## 🚀 Features

- Clean and responsive frontend for user interaction.
- Accepts input for key health parameters (e.g., Hemoglobin, Age, etc.).
- Trained on real anemia-related datasets.
- Supports multiple ML models (Logistic Regression, SVM, Random Forest, Gradient Boosting).
- Optimized using GridSearchCV and RandomizedSearchCV.
- Predicts whether a user is anemic or not with high accuracy.

---

## 🧠 Tech Stack

- **Frontend:** HTML, CSS
- **Backend:** Python (Flask)
- **ML Models:** scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Flask server (can be hosted on platforms like Render, Heroku, or GCP)

---

## 📂 Project Structure
```
├── static/                  # CSS, JS, images
├── templates/               # HTML templates
├── models/                  # Saved ML models
├── notebooks/               # Jupyter notebooks for EDA & model training
├── app.py                   # Flask server
├── predict.py               # ML prediction logic
├── anemia_data.csv          # Dataset used
└── requirements.txt         # Python dependencies
```

---

## ⚙️ How to Run

```
git clone https://github.com/yourusername/anemia-sense.git
cd anemia-sense
pip install -r requirements.txt
python app.py


Then, open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.
```
---

## 📈 Model Performance

* Final Model: **Gradient Boosting Classifier**
* Accuracy: **95.2%**
* Balanced performance across Precision, Recall, and F1-Score

---



## 🙏 Acknowledgements

* SmartInternz Virtual Internship Program
* Mentor: **Shiva Charan**



