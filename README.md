# 🌾 Crop Recommendation System

A machine learning project that recommends the best crop to grow based on soil and climate conditions.

## 📌 Problem Statement

Farmers in India often rely on guesswork or tradition when deciding which crop to grow, leading to poor yield and financial loss. This project uses Machine Learning to recommend the most suitable crop based on measurable soil and weather parameters.

## 💡 Solution

A classification model trained on real agricultural data that takes soil nutrients and climate inputs and predicts the best crop to grow.

## 🛠️ Tech Stack

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 📂 Project Structure

```
crop-recommendation/
│
├── crop_recommendation.py   # Main ML code
├── crop_data.csv            # Dataset (download from Kaggle)
├── crop_distribution.png    # Auto-generated chart
├── correlation_heatmap.png  # Auto-generated chart
├── feature_importance.png   # Auto-generated chart
└── README.md
```

## 📊 Dataset

- **Source:** [Kaggle - Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Rows:** 2,200
- **Features:** Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, Rainfall
- **Label:** Crop name (22 types including rice, wheat, mango, cotton, etc.)

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/crop-recommendation.git
cd crop-recommendation
```

### 2. Install dependencies
```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

### 3. Download the dataset
- Go to: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
- Download `Crop_recommendation.csv`
- Rename it to `crop_data.csv`
- Place it in the same folder as `crop_recommendation.py`

### 4. Run the code
```bash
python crop_recommendation.py
```

## 🎯 How to Make Your Own Prediction

In `crop_recommendation.py`, scroll to **STEP 8** and change the values:

```python
sample_input = {
    'N': 90,            # Nitrogen level in soil
    'P': 42,            # Phosphorus level
    'K': 43,            # Potassium level
    'temperature': 20,  # Temperature in Celsius
    'humidity': 82,     # Humidity in %
    'ph': 6.5,          # Soil pH value
    'rainfall': 202     # Rainfall in mm
}
```

Run the file and it will tell you the recommended crop!

## 📈 Model Performance

- **Algorithm:** Random Forest Classifier
- **Accuracy:** ~99% on test data
- **Train/Test Split:** 80% / 20%

## 🌍 Real-World Impact

This tool can help small and marginal farmers — especially in states like Madhya Pradesh — make data-driven decisions about crop selection, potentially improving yield and reducing financial risk.

## 👤 Author

- **Name:** Chitra Prabhakar 
- **Course:** Fundamentals of AI and ML
- **Institution:** VIT BHOPAL UNIVERSITY 
