import pandas as pd
import numpy as np
import re
import pandas as pd
import numpy as np
import re
app = app.server

#creating a function that cleans the title
def clean_title(title):
    return re.sub("[^A-Za-z0-9 ]", "", title)

movies = pd.read_csv('dataset/movies_data/movies.zip')
ratings =  pd.read_csv('dataset/movie_data/ratings.zip')
movies['clean_title'] = movies['title'].apply(clean_title)
    


#movies['clean_title'] = movies['title'].apply(clean_title)

#vectorizing the title i.e transforming the title to numbers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def search(title):
    vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf = vectorizer.fit_transform(movies['clean_title'])
    #title = "Toy Story 1995"
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec,tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices][::-1]
    return results

''' import ipywidgets as widgets
from IPython.display import display

movie_input = widgets.Text(
    value="Toy Story",
    description="Movie Title:",
    disabled=False)

movie_list = widgets.Output()

def on_type(data):
    with movie_list:
        movie_list.clear_output()
        title = data['new']
        if len(title) > 5:
            display(search(title))
            
movie_input.observe(on_type, names='value')

display(movie_input, movie_list)'''



def find_similar_movies(movie_id):
    similar_users = ratings[(ratings['movieId']==movie_id) & (ratings['rating'] > 4)]['userId'].unique()
    similar_user_recs = ratings[(ratings['userId'].isin(similar_users) & (ratings['rating'] > 4))]['movieId']
    
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)
    similar_user_recs = similar_user_recs[similar_user_recs > .1]    
    
    all_users = ratings[(ratings['movieId'].isin(similar_user_recs.index)) & (ratings['rating'] > 4)]
    all_user_recs = all_users['movieId'].value_counts() / len(all_users['userId'].unique())
    
    rec_percentage = pd.concat([similar_user_recs,all_user_recs], axis=1)
    rec_percentage.columns = ['similar','all']
    
    rec_percentage['score'] = rec_percentage['similar'] / rec_percentage['all']
    rec_percentage = rec_percentage.sort_values('score',ascending=False)
    
    return rec_percentage.head(10).merge(movies, left_index=True, right_on='movieId')[['score','title','genres']]

# movie_name_input = widgets.Text(
   # value="Toy Story",
   # description="Movie Title:",
    #disabled=False)

#recommendation_list = widgets.Output()

#def on_type(data):
   # with recommendation_list:
    #    recommendation_list.clear_output()
     #   title=data['new']
     #   if len(title) > 5:
     #       results = search(title)
     #       movie_id = results.iloc[0]['movieId']
      #      display(find_similar_movies(movie_id))
            
#movie_name_input.observe(on_type, names='value')

#display(movie_name_input, recommendation_list)``



