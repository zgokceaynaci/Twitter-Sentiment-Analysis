# Twitter Sentiment Analysis

## 📌 Project Overview
This project performs sentiment analysis on Twitter data using various machine learning algorithms. The main goal is to classify tweets into three sentiment categories:
- **Positive**
- **Neutral**
- **Negative**

The dataset is sourced from multiple Kaggle repositories and contains over 1.5 million tweets after preprocessing.

## 🛠 Technologies Used
- **Python**
- **scikit-learn** (Machine Learning Models)
- **TensorFlow/Keras** (Deep Learning Models)
- **Pandas & NumPy** (Data Processing)
- **Matplotlib & Seaborn** (Data Visualization)
- **TF-IDF Vectorization**

## 📂 Project Structure
📁 twitter-sentiment-analysis │── 📂 data # (Dataset is not included in this repo) │── 📂 notebooks # Jupyter Notebooks for experiments │── 📂 src # Source code for preprocessing and training │── requirements.txt # Python dependencies │── .gitignore # Ignore unnecessary files │── README.md # Project documentation


## 🔧 Installation & Usage
### 1️⃣ Clone the Repository
    ```bash
    git clone https://github.com/your-username/twitter-sentiment-analysis.git
    cd twitter-sentiment-analysis
### 2️⃣ Install Dependencies
    ```bash
    pip install -r requirements.txt
### 3️⃣ Run the Model
    ```bash
    python src/preprocessing.py
    python src/train_model.py
## 🏆 Results & Performance
| Model                | Accuracy  | Precision | Recall  | F1-Score |
|----------------------|----------|-----------|---------|----------|
| Logistic Regression | 79.42%   | 78.00%    | 78.00%  | 78.00%   |
| Random Forest       | 89.71%   | 90.00%    | 90.00%  | 90.00%   |
| SVM                 | 79.37%   | 79.00%    | 79.00%  | 79.00%   |
| ANN                 | 90.00%   | 89.50%    | 89.00%  | 89.25%   |
| k-NN                | 87.43%   | 88.00%    | 87.00%  | 87.50%   |
| Decision Tree       | 80.22%   | 80.00%    | 80.00%  | 80.00%   |


## 📢 Future Improvements
-Implement deep learning models such as BERT or RoBERTa for better sentiment understanding.

-Use more advanced NLP techniques like word embeddings and transformers.

-Collect real-time Twitter data for dynamic analysis.
    

