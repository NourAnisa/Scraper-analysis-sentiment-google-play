from flask import Flask, render_template, request, send_file
from google_play_scraper import Sort, reviews
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import io
import base64
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        app_id = request.form['app_id']

        result, continuation_token = reviews(
            app_id,
            lang='id',
            country='id',
            sort=Sort.MOST_RELEVANT,
            count=1000,
            filter_score_with=None
        )

        data = pd.DataFrame(np.array(result), columns=['review'])
        data = data.join(pd.DataFrame(data.pop('review').tolist()))
        data = data[['content', 'score']]
        data = data.rename(columns={'content': 'komentar', 'score': 'value'})

        data = data.dropna()
        data['clean_text'] = data['komentar'].str.replace(
            '[^\w\s]', '', regex=True)
        data['clean_text'] = data['clean_text'].str.lower()

        nltk.download('stopwords')
        stop_words = set(stopwords.words('indonesian'))
        stop_words.update(['dan', 'yang', 'lalu', 'yg', 'gk', 'saya', 'lagi'])
        data['clean_text'] = data['clean_text'].apply(lambda x: ' '.join(
            [word for word in x.split() if word not in (stop_words)]))

        # Word Cloud
        all_text = ' '.join(data['clean_text'])
        wordcloud = WordCloud(
            width=1000, height=500, max_font_size=150, random_state=42).generate(all_text)
        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()

        nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
        data['sentiment'] = data['clean_text'].apply(
            lambda x: sia.polarity_scores(x)['compound'])
        data['sentiment_label'] = data['sentiment'].apply(
            lambda x: 'positif' if x > 0 else ('negatif' if x < 0 else 'netral'))
        sentiment_counts = data['sentiment_label'].value_counts()

        # Sentiment Distribution Plot
        plt.figure(figsize=(8, 6))
        sns.barplot(x=sentiment_counts.index,
                    y=sentiment_counts.values, palette="viridis")
        plt.title('Distribusi Sentimen')
        plt.xlabel('Sentimen')
        plt.ylabel('Jumlah')
        plt.tight_layout()
        img_graph = io.BytesIO()
        plt.savefig(img_graph, format='PNG')
        img_graph.seek(0)
        img_graph_base64 = base64.b64encode(img_graph.getvalue()).decode()

        data.to_csv("processed_comments.csv", index=False, encoding='utf-8')

        # TF-IDF Calculation
        tfidf_vectorizer = TfidfVectorizer(max_features=10)
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['clean_text'])
        feature_names = tfidf_vectorizer.get_feature_names_out()
        dense = tfidf_matrix.todense()
        denselist = dense.tolist()
        tfidf_df = pd.DataFrame(denselist, columns=feature_names)
        tfidf_top_words = tfidf_df.sum().sort_values(ascending=False).head(10)

        # TF-IDF Plot
        plt.figure(figsize=(8, 6))
        sns.barplot(x=tfidf_top_words.values,
                    y=tfidf_top_words.index, palette="viridis")
        plt.title('Top 10 Kata dengan TF-IDF Tertinggi')
        plt.xlabel('Nilai TF-IDF')
        plt.ylabel('Kata')
        plt.tight_layout()
        img_tfidf = io.BytesIO()
        plt.savefig(img_tfidf, format='PNG')
        img_tfidf.seek(0)
        img_tfidf_base64 = base64.b64encode(img_tfidf.getvalue()).decode()

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            data['clean_text'], data['sentiment_label'], test_size=0.3, random_state=42)
        tfidf_vectorizer = TfidfVectorizer()
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Naïve Bayes
        nb = MultinomialNB()
        nb.fit(X_train_tfidf, y_train)
        y_pred_nb = nb.predict(X_test_tfidf)
        nb_accuracy = accuracy_score(y_test, y_pred_nb)

        # SVM
        svm = SVC()
        svm.fit(X_train_tfidf, y_train)
        y_pred_svm = svm.predict(X_test_tfidf)
        svm_accuracy = accuracy_score(y_test, y_pred_svm)

        # Logistic Regression
        lr = LogisticRegression()
        lr.fit(X_train_tfidf, y_train)
        y_pred_lr = lr.predict(X_test_tfidf)
        lr_accuracy = accuracy_score(y_test, y_pred_lr)

        # Lexicon-based Accuracy
        y_pred_lexicon = data['clean_text'].apply(lambda x: 'positif' if sia.polarity_scores(
            x)['compound'] > 0 else ('negatif' if sia.polarity_scores(x)['compound'] < 0 else 'netral'))
        lexicon_accuracy = accuracy_score(
            data['sentiment_label'], y_pred_lexicon)

        return render_template(
            'result.html',
            img_base64=img_base64,
            img_graph_base64=img_graph_base64,
            img_tfidf_base64=img_tfidf_base64,
            sentiment_counts=sentiment_counts.to_dict(),
            data=data,
            nb_accuracy=nb_accuracy,
            svm_accuracy=svm_accuracy,
            lr_accuracy=lr_accuracy,
            lexicon_accuracy=lexicon_accuracy
        )


@app.route('/download')
def download_file():
    return send_file("processed_comments.csv", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
