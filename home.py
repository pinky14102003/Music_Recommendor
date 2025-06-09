import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.neighbors import NearestNeighbors



st.set_page_config(layout='wide')
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)




#----------------MODEL 3--------------------#
s=pd.read_csv('dataset.csv')
s=s.drop_duplicates(subset=['track_name','artists']).reset_index(drop=True)
s=s.drop(columns=['Unnamed: 0'])
s.dropna(inplace=True)
s['artists']=s['artists'].apply(lambda x:x.split(";"))
s['artists']=s['artists'].apply(lambda x:[i.replace(' ','') for i in x])
s['artists']=s['artists'].apply(lambda x:' '.join(x))
s['artists']=s['artists'].apply(lambda x: x.lower())

s['track_name']=s['track_name'].apply(lambda x: x.lower())
s['track_name']=s['track_name'].apply(lambda x: x.replace('-',''))

s['album_name']=s['album_name'].apply(lambda x: x.lower())
s['track_genre']=s['track_genre'].apply(lambda x: x.lower())

s['ft']=s['artists']+' '+s['album_name']+' '+s['track_name']+' '+s['track_genre']

ps=PorterStemmer()
def stem(text):
  y=[]
  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

ps=PorterStemmer()
c=CountVectorizer(max_features=10000,stop_words='english')
de=c.fit_transform(s['ft']).toarray()
nn=NearestNeighbors(n_neighbors=20,metric='cosine')
nn.fit(de)
def basicrecommend(name,artist,album):
  p = s[(s['track_name'] == name) & (s['artists'] == artist)& (s['album_name'] == album)]
  p=p.sort_values(by='popularity',ascending=False).head(1)['ft']
  d=c.transform(p).toarray()
  distances, indices = nn.kneighbors(d)
  return s.iloc[[int(i) for i in indices[0]]]


#------------STREAMLIT APP----------------#

st.title('Song Recommendor System')
a=st.selectbox('Select Your Song Name',options=s['track_name'])
k=st.selectbox('Select Your Artist',options=s[s['track_name']==a]['artists'])
d=st.selectbox('Select Your Album/Playlist/Movie',options=s[(s['track_name']==a)&(s['artists']==k)]['album_name'])
b=st.button('Search')
if b:
  with st.spinner(text='Generating Your Recommendations'):
    e=basicrecommend(a,k,d)
  z=e['track_id'].reset_index(drop=True)
  e=e.reset_index(drop=True)
  st.header("You May also like")
  st.dataframe(e[['track_name','artists', 'album_name', 
       'popularity','danceability', 'energy','loudness','speechiness', 'acousticness',
       'instrumentalness', 'liveness', 'valence','track_genre']])
  for i in range(len(z)):
    spotify_url = f"https://open.spotify.com/track/{z.loc[i]}"
    st.markdown(f"[{e.loc[i,'track_name']}]({spotify_url})")


#-------------------STREAMLIT FEEDBACK--------------------#
st.sidebar.write("""
This Feedback section is under Development please skip
""")
form1=st.sidebar.form('1')
with form1:
    st.write("Your Feedback")
    r1=st.radio("Did the model reccomend you songs according \
                      to your current mood ?",['Yes','Partially','No'])
    r2=st.radio("Did the model reccomend you songs according \
                      to your current favourite artist ?",['Yes','Partially','No'])
    r3=st.radio("Did the model reccomend you songs according \
                      to your current favourite genre ?",['Yes','Partially','No'])
    r4=st.radio("Did the model help you sustain or improve your mood ?",['Yes','Partially','No'])
    
    f1 = st.text_area("Overall review", height=150)
    submitted = st.form_submit_button("Submit")
    if submitted:
       pass
       
