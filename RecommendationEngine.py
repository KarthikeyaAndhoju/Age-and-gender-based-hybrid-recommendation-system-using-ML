
#RECOMMENDATION SYSTEM
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv(r"C:\Users\Koushik\Desktop\mini\netflix_titles.csv")
data.dropna(subset=['cast','title','description','listed_in'],inplace=True,axis=0)
data = data.reset_index(drop=True)
for i in range(len(data['age'])):
    data['age'][i]=str(data['age'][i])+'+'


#print(data['age'])
data['listed_in'] = [re.sub(r'[^\w\s]', '', t) for t in data['listed_in']]
data['cast'] = [re.sub(',',' ',re.sub(' ','',t)) for t in data['cast']]
data['description'] = [re.sub(r'[^\w\s]', '', t) for t in data['description']]
data['title'] = [re.sub(r'[^\w\s]', '', t) for t in data['title']]
data["combined"] = data['listed_in'] + '  ' + data['cast'] + ' ' + data['title'] + ' ' + data['description']

data.drop(['listed_in','cast','description'],axis=1,inplace=True)
data.head()



# Content Similarity
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(data["combined"])
cosine_similarities = linear_kernel(matrix,matrix)
movie_title = data['title']+data['age']
indices = pd.Series(data.index, index=data['title'])

def content_recommender(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return movie_title.iloc[movie_indices]

title = 'The Crown'
suggestions = content_recommender(title)

suggestions_df = pd.DataFrame(data=suggestions)
for i in suggestions_df[0]:
    if(int(i[-3:-1])>15):
        print(i[:-3])
