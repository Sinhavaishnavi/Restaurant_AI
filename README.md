
# 🍽️ Restaurant Rating Prediction

This project uses a machine learning model to predict the **aggregate rating** of restaurants based on various features such as location, cuisines, cost, etc.

## 🧠 Objective

To build a regression model that predicts the `Aggregate rating` of a restaurant using other available features in the dataset.

---
<img width="1893" height="870" alt="Screenshot 2025-08-03 000640" src="https://github.com/user-attachments/assets/b09c792f-16a2-4b0e-bffb-3d12df429f76" />
## 📁 Dataset

- File: `Dataset .csv`
- Contains various columns such as:
  - Restaurant Name
  - Location
  - Cuisines
  - Price range
  - Votes
  - Online order, etc.

---

## 🔧 Installation & Setup

### 1. Clone or Download this Repository
```bash
git clone https://github.com/Sinhavasihnavi/restaurant-rating-prediction.git
cd restaurant-
rating-prediction
```

### 2. Create Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Libraries
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

Run the main script:
```bash
streamlit run app3.py
```

This will:
- Load the dataset
- Preprocess it
- Train a Random Forest Regressor
- Predict ratings
- Display evaluation metrics and feature importance

---

## 📊 Model Performance

- **Predicted Rating Example**: 4.00 / 5.0  
- **Mean Squared Error (MSE)**: 0.2035  
- **R-squared (R²) Score**: 0.9106

---

## 🔍 Top Features Impacting Rating

- Votes  
- Average Cost for Two  
- Cuisines  
- Location  
- Online Order Availability

---

## 📎 Notes

- You can replace the model with any other regressor like `LinearRegression`, `decision tree` for experimentation.
- This project is suitable for beginners in ML focusing on real-world tabular datasets.

---

## 📧 Contact

For feedback or queries, feel free to contact:

**Vaishnavi Sinha**  
📧 vaishnavisinha476@gmail.com.com  
📍 India
