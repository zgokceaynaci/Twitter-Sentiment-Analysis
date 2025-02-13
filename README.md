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
- **Google Colab** (Model Training & Experiments)

## 📂 Project Structure

```
📁 twitter-sentiment-analysis
│── 📂 datasets          # Raw and processed dataset files
│── 📂 notebooks         # Jupyter Notebooks for experiments
│── 📂 src               # Source code for preprocessing and training
│── requirements.txt     # Python dependencies
│── .gitignore          # Ignore unnecessary files
│── README.md           # Project documentation
│── main.py             # Main script for processing datasets
```

## 🔧 Installation & Usage

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/zgokceaynaci/Twitter-Sentiment-Analysis.git
cd Twitter-Sentiment-Analysis
```

    git clone https://github.com/your-username/twitter-sentiment-analysis.git
    cd twitter-sentiment-analysis
### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Data Preprocessing Script

```bash
python main.py
```

This script will:
- Load and clean three datasets
- Remove duplicates and unnecessary columns
- Standardize sentiment labels
- Save the cleaned dataset as `final_cleaned_dataset.csv`

### 3️⃣ Run the Model
    python src/preprocessing.py
    python src/train_model.py
## 🏆 Results & Performance

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 79.42%   | 78.00%    | 78.00% | 78.00%   |
| Random Forest       | 89.71%   | 90.00%    | 90.00% | 90.00%   |
| SVM                 | 79.37%   | 79.00%    | 79.00% | 79.00%   |
| ANN                 | 90.00%   | 89.50%    | 89.00% | 89.25%   |
| k-NN                | 87.43%   | 88.00%    | 87.00% | 87.50%   |
| Decision Tree       | 80.22%   | 80.00%    | 80.00% | 80.00%   |

The best-performing model was Artificial Neural Networks (ANN) with an accuracy of 90.00%.

### 📌 **Dataset Sources**

The dataset consists of three sources:

1. **Sentiment140 Dataset** ([Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140))
2. **Twitter Sentiment Analysis Dataset** ([Kaggle](https://www.kaggle.com/datasets/abdallahwagih/twitter-sentiment-analysis))
3. **Twitter Validation Set** ([Kaggle](https://www.kaggle.com/datasets/muhammadimran112233/eda-twitter-sentiment-analysis-using-nn))

### 📈 **Data Preprocessing Steps**
- Removed unnecessary columns (IDs, timestamps, usernames)
- Cleaned text (lowercasing, punctuation removal, extra spaces)
- Standardized sentiment labels to numerical values (0 = Negative, 1 = Neutral, 2 = Positive)
- Removed duplicates and handled missing values
- Combined all datasets into a final cleaned dataset (`final_cleaned_dataset.csv`)
The best-performing model was Artificial Neural Networks (ANN) with an accuracy of 90.00%.

## 🚀 Future Improvements
- Implement **BERT** or **RoBERTa** for advanced NLP modeling
- Utilize **word embeddings** instead of TF-IDF
- Collect real-time Twitter data and update model dynamically
-Use more advanced NLP techniques like word embeddings and transformers.

## 📢 References

- **scikit-learn**: [Machine Learning in Python](https://scikit-learn.org/stable/)
- **TensorFlow**: [Building Deep Learning Models](https://www.tensorflow.org/)
- **Kaggle Datasets** (linked above)

## 🤝 Contributions
Feel free to fork this repository and submit pull requests. Contributions are always welcome!

## 📜 License
This project is licensed under the MIT License.

