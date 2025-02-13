# Import necessary libraries
import pandas as pd
import re

# Function to clean text
def clean_text(text):
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip().lower()

# Function to convert target values (Dataset 1)
def convert_target_to_label(target):
    if target == 0:
        return 0  # Negative
    elif target == 2:
        return 1  # Neutral
    elif target == 4:
        return 2  # Positive
    else:
        return None

# Function to convert sentiment values (Dataset 2 and 3)
def convert_sentiment_to_label(sentiment):
    sentiment = sentiment.lower()
    if sentiment == "negative":
        return 0  # Negative
    elif sentiment == "neutral":
        return 1  # Neutral
    elif sentiment == "positive":
        return 2  # Positive
    else:
        return None

# ============================
# Load and Process Dataset 1
# ============================
print("Processing Dataset 1...")
df1 = pd.read_csv(
    "training.1600000.processed.noemoticon.csv",
    encoding="latin-1",
    header=None,
    on_bad_lines="skip"
)
df1.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
df1 = df1[['text', 'target']]  # Keep only relevant columns
df1['label'] = df1['target'].apply(convert_target_to_label)
df1.dropna(subset=['label'], inplace=True)  # Remove rows with invalid labels
df1['text'] = df1['text'].astype(str)  # Ensure all text values are strings
df1['cleaned_text'] = df1['text'].apply(clean_text)  # Clean text
df1 = df1[['cleaned_text', 'label']]  # Retain only cleaned text and label
print("Dataset 1 Label Distribution:")
print(df1['label'].value_counts())

# ============================
# Load and Process Dataset 2
# ============================
print("Processing Dataset 2...")
df2 = pd.read_csv("twitter_training.csv", header=None)
df2.columns = ['id', 'category', 'sentiment', 'text']
df2 = df2[['text', 'sentiment']]  # Keep only relevant columns
df2['label'] = df2['sentiment'].apply(convert_sentiment_to_label)
df2.dropna(subset=['label'], inplace=True)  # Remove rows with invalid labels
df2['text'] = df2['text'].astype(str)  # Ensure all text values are strings
df2['cleaned_text'] = df2['text'].apply(clean_text)  # Clean text
df2 = df2[['cleaned_text', 'label']]  # Retain only cleaned text and label
print("Dataset 2 Label Distribution:")
print(df2['label'].value_counts())

# ============================
# Load and Process Dataset 3
# ============================
print("Processing Dataset 3...")
df3 = pd.read_csv("twitter_validation.csv", header=None)
df3.columns = ['id', 'category', 'sentiment', 'text']
df3 = df3[['text', 'sentiment']]  # Keep only relevant columns
df3['label'] = df3['sentiment'].apply(convert_sentiment_to_label)
df3.dropna(subset=['label'], inplace=True)  # Remove rows with invalid labels
df3['text'] = df3['text'].astype(str)  # Ensure all text values are strings
df3['cleaned_text'] = df3['text'].apply(clean_text)  # Clean text
df3 = df3[['cleaned_text', 'label']]  # Retain only cleaned text and label
print("Dataset 3 Label Distribution:")
print(df3['label'].value_counts())

# ============================
# Combine All Datasets
# ============================
print("Combining all datasets...")
combined_df = pd.concat([df1, df2, df3], ignore_index=True)
combined_df.dropna(inplace=True)  # Remove rows with missing values
combined_df.drop_duplicates(subset='cleaned_text', inplace=True)  # Remove duplicates
print("Final Dataset Label Distribution:")
print(combined_df['label'].value_counts())

# Save the Final Dataset
combined_df.to_csv("final_cleaned_dataset.csv", index=False)
print("Final cleaned dataset saved as 'final_cleaned_dataset.csv'")
