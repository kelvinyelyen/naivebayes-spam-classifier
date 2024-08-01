# Naive Bayes-Spam-classification
### Implementation Details

1. **Data Loading and Preprocessing**:
   - The dataset, containing email messages labeled as 'ham' (not spam) or 'spam', is loaded from a TSV file.
   - Labels are converted into binary values (`0` for 'ham' and `1` for 'spam') to facilitate training.

2. **Feature Extraction**:
   - **CountVectorizer** is employed to transform the text messages into a matrix of token counts. This matrix serves as the feature set (`X`) for the machine learning model.
   - The vectorizer is first fitted on the training data (`X_train`) and then used to transform both training and test datasets.

3. **Model Training**:
   - A **Multinomial Naive Bayes** model is trained on the transformed training data (`X_train_transformed`, `y_train`). This model is particularly effective for text classification problems, where the features are represented by word counts.

4. **Prediction Function**:
   - The `predict_spam_or_ham` function takes a user input string, transforms it using the previously fitted vectorizer, and predicts whether the message is 'Spam' or 'Not Spam (ham)' using the trained Naive Bayes model.

5. **Streamlit Application**:
   - The app's UI allows users to input a message and classify it as 'Spam' or 'Not Spam (ham)'.
   - The `st.text_area()` component captures user input, and the `st.button()` triggers the classification. The result is displayed using `st.write()`.

### Installation and Setup

1. **Environment Setup**:
   - Ensure Python 3.7+ is installed.
   - Install required libraries:
     ```bash
     pip install streamlit pandas scikit-learn
     ```

2. **Running the Application**:
   - Execute the Streamlit app with the following command:
     ```bash
     streamlit run app.py
     ```
