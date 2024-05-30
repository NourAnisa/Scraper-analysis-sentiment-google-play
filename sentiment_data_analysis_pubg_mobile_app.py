from flask import Flask, render_template, request
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

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        # Mendapatkan data dari Google Play Store
        result, continuation_token = reviews(
            'com.tencent.ig',
            lang='id',
            country='id',
            sort=Sort.MOST_RELEVANT,
            count=1000,
            filter_score_with=None
        )

        # Dataframe dengan nama
        data = pd.DataFrame(np.array(result), columns=['review'])
        data = data.join(pd.DataFrame(data.pop('review').tolist()))
        data = data[['content', 'score']]
        data = data.rename(columns={'content': 'komentar', 'score': 'value'})

        # Membersihkan data
        data = data.dropna()
        data['clean_text'] = data['komentar'].str.replace(
            '[^\w\s]', '', regex=True)
        data['clean_text'] = data['clean_text'].str.lower()

        # Menghilangkan stop words
        nltk.download('stopwords')
        stop_words = set(stopwords.words('indonesian'))
        stop_words.update(['dan', 'yang', 'lalu', 'yg', 'gk', 'saya', 'lagi'])
        data['clean_text'] = data['clean_text'].apply(lambda x: ' '.join(
            [word for word in x.split() if word not in (stop_words)]))

        # Membuat word cloud
        all_text = ' '.join(data['clean_text'])
        wordcloud = WordCloud(
            width=1000, height=500, max_font_size=150, random_state=42).generate(all_text)

        # Menyimpan word cloud sebagai gambar dalam format base64
        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode()

        # Melakukan sentimen analisis
        nltk.download('vader_lexicon')
        sia = SentimentIntensityAnalyzer()
        data['sentiment'] = data['clean_text'].apply(
            lambda x: sia.polarity_scores(x)['compound'])
        data['sentiment_label'] = data['sentiment'].apply(
            lambda x: 'positif' if x > 0 else ('negatif' if x < 0 else 'netral'))

        # Menghitung jumlah sentimen
        sentiment_counts = data['sentiment_label'].value_counts()

        return render_template('result.html', img_base64=img_base64, sentiment_counts=sentiment_counts.to_dict(), data=data)


if __name__ == '__main__':
    app.run(debug=True)
