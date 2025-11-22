import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast

df = pd.read_csv("tmdb_5000_credits.csv")
df.dropna(inplace=True)

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

df['cast'] = df['cast'].apply(convert)

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

df['crew'] = df['crew'].apply(fetch_director)

df['tags'] = df['cast'] + df['crew']
df['tags'] = df['tags'].apply(lambda x: " ".join(x)).str.lower()

cv = CountVectorizer(max_features=2000, stop_words='english')
vectors = cv.fit_transform(df['tags']).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie = movie.lower()
    df['title_lower'] = df['title'].str.lower()

    if movie not in df['title_lower'].values:
        print("Movie not found! Check spelling or try another.")
        return

    movie_index = df[df['title_lower'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    for i in movies_list:
        print(df.iloc[i[0]].title)

name=input("enter your movie name:")
recommend(name)
