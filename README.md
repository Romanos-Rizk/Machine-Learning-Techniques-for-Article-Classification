# Text Classification System for The New York Times Articles

## Project Overview
This project involves developing a text classification system using a dataset of over 2.1 million articles from The New York Times, spanning from January 1, 2000, to the present day. The goal is to categorize news articles based on their section names using various machine learning models. The BERT model, optimized without a validation set, emerged as the most effective classifier with an accuracy of 88%.

## Data Sources
- **Dataset**: "The New York Times Articles Metadata" 
  - Source: Kaggle
  - Description: Contains metadata for articles including abstract, snippet, headline, keywords, and section names.

## Tools + Purpose
- **Python**: For data processing, machine learning model development, and evaluation.
  - **sklearn**
  - **nltk**
  - **matplotlib**
  - **seaborn**

## Data Cleaning and Preparation
- **Transforming Column Names**: Standardized column names for consistency.
- **Handling Missing Values**: Removed or replaced missing values in key columns.
- **Converting Date Columns**: Changed date columns to date/time format for meaningful analysis.
- **Normalizing Categorical Columns**: Standardized categorical data.
- **Addressing Illogical Values**: Removed rows with invalid values.
- **Normalizing Description Columns**: Cleaned and standardized description text.

## Exploratory Data Analysis
- **Questions Asked**:
  - What are the most frequent words in article headlines?
  - How are articles distributed across different sections?
  - How have article distributions changed over time?
- **Visuals**:
  - Word frequency in headlines.
  - Distribution of articles across sections.
  - Heatmaps of article counts by year and section.

## Data Analysis
- **Interesting Code**:
  - Text vectorization using TF-IDF.
  - Feature selection using Recursive Feature Elimination.
- **Models Evaluated**:
  - Logistic Regression
  - Random Forest Classifier
  - KNN
  - Gradient Boosting Classifier
  - Sequential Neural Network
  - BERT

## Models

### Random Forest Classifier
- **Initial Accuracy**: 71.88%
- **Improved Accuracy**: 72.16% with 3,000 estimators and max depth of 100.
- **Optimized Accuracy**: 72.77% with parameters: `n_estimators=300`, `max_depth=None`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features='sqrt'`, `bootstrap=False`.

### Logistic Regression
- **Initial Accuracy**: 66.01%
- **Improved Accuracy**: 71.49% with initial hyperparameter adjustments.
- **Optimized Accuracy**: 74.73% with parameters: `C=10`, `class_weight='balanced'`, `penalty='L2'`, `solver='LBFGS'`.

### K-Nearest Neighbors (KNN)
- **Initial Accuracy**: 62.77%
- **Improved Accuracy**: 63.83% with 9 neighbors.
- **Optimized Accuracy**: 66% with parameters: `algorithm='auto'`, `metric='euclidean'`, `n_neighbors=10`, `weights='distance'`.
- **Note**: KNN performed poorly on large-scale datasets and high-dimensional feature spaces.

### XGBoost
- **Initial Accuracy**: 71.49%
- **Improved Accuracy**: 72.61% with parameters: `use_label_encoder=False`, `eval_metric='mlogloss'`, `max_depth=5`.
- **Optimized Accuracy**: 72.55% with parameters: `eval_metric='mlogloss'`, `learning_rate=0.2`, `max_depth=4`, `num_class=24`, `objective='multi:softmax'`.

### Neural Network
- **Initial Accuracy**: 71.86%
- **Architecture**: Embedding Layer, LSTM Layer, Dense Layer, Dropout Layer, Softmax Output.
- **Training**: 10 epochs, batch size of 32.
- **Notes**: Potential for improvement by refining model structure or training methodology.

### BERT
- **Initial Accuracy**: 79% (with validation set).
- **Optimized Accuracy**: 88% (without validation set).
- **Configuration**: 'bert-base-uncased', modified for specific dataset labels, 3 epochs, monitored loss for learning rate adjustments.

## Results and Findings

### Best Model and Comparative Analysis with Related Work
Among the models tested, the BERT model, optimized without a validation set, achieved an impressive accuracy of 88% on the test set. This model leveraged the deep learning capabilities of the 'bert-base-uncased' configuration, demonstrating superior performance due to its ability to extract and learn from complex textual relationships effectively. This success highlights BERT's robustness in text classification tasks, surpassing traditional machine learning techniques that typically achieve accuracies between 70% and 85%.

### Results Table
| Model                  | Base Accuracy | Literature Review | Grid Search |
|------------------------|---------------|-------------------|-------------|
| Random Forest Classifier | 71.88%        | 72.16%            | 72.77%      |
| Logistic Regression      | 66.01%        | 71.49%            | 74.73%      |
| KNN                      | 62.77%        | 63.83%            | 66%         |
| XGBoost                  | 71.49%        | 72.61%            | 72.55%      |
| Neural Network           | 71.86%        | -                 | -           |
| BERT                     | 79% (validation) | 88% (test only) | -           |

### Answering the Research Question
Our research involved key preprocessing steps including handling null values, text standardization, and TF-IDF vectorization. Multiple models were tested and refined, with BERT emerging as the most effective for handling the diverse and intricate text data in the dataset. This approach optimized machine learning and NLP techniques, demonstrating significant improvements in classification accuracy and relevance.

## Error Analysis
- **Confusion Matrix**: The model performed well in distinguishing categories with distinct content (e.g., "Sports") but struggled with overlapping themes (e.g., "U.S." vs. "World").
- **Classification Report**: Performance varied across categories, with issues in "Blogs" and "Technology." Categories with less training data or broader themes showed lower performance.
- **Text Length Analysis**: Shorter texts were more prone to misclassification, indicating the model benefits from more context.
- **Word and Phrase Analysis**: Common phrases (e.g., "New York," "Los Angeles") led to misclassifications due to their occurrence across various categories.
- **Sentiment Analysis**: No significant correlation between text sentiment and classification accuracy was found.

## Data Handling, Business Recommendations, Strategic Insights, and Future Work

### Data Handling
Advanced machine learning algorithms have proven effective for categorizing articles, enhancing content delivery efficiency.

### Business Recommendations
- **User Experience**: Utilize categorization algorithms to tailor information streams based on user interests, potentially increasing engagement and subscriptions.
- **Editorial Strategy**: Guide editorial decisions by identifying popular subjects and content gaps to attract a larger readership.

### Strategic Insights
- **Real-Time Analytics**: Explore real-time analytics to dynamically recommend content as it's created, using user interaction data to refine predictions.
- **Unsupervised Learning**: Investigate unsupervised methods to discover emerging topics and patterns, offering new approaches for content generation and dissemination.

### Future Work
- **Real-Time Recommendations**: Develop models that adapt to real-time user interactions.
- **Advanced Techniques**: Explore unsupervised learning and other emerging techniques to further enhance classification accuracy and content relevance.

## References
- **Dataset Source**: [Kaggle](https://www.kaggle.com/)
- **Text Processing**: Various NLP and machine learning literature for preprocessing and model selection.
