import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



st.set_page_config(layout='wide')

s=pd.read_csv('dataset.csv')
s=s.drop_duplicates(subset=['artists','track_name']).reset_index(drop=True)
s.dropna(inplace=True)
s.drop(columns=['Unnamed: 0'],inplace=True)
s['artists']=s['artists'].apply(lambda x:x.split(';'))
s['artists']=s['artists'].apply(lambda x:[i.replace(' ','')for i in x])
s['artists']=s['artists'].apply(lambda x:" ".join(x))
new_df=s[['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(new_df)
kmeans=KMeans(n_clusters=7,init='k-means++',random_state=42)
kmeans.fit(scaled_data)
predicted_data=kmeans.predict(scaled_data)
s['group']=pd.Series(predicted_data)
def recommend(song):
  target_group = s[s['track_name']==song].sort_values(by='popularity',ascending=False)['group'].head(1).iloc[0]
  return s[s['group']==target_group].head(5)[['track_name','album_name','artists']]




st.title('Song Recommendor System')
a=st.selectbox('Enter Your Song Name',options=s['track_name'])
b=st.button('Search')
if b:
    e=recommend(a)
    st.write("You May also like")
    c1,c2,c3=st.columns(3)
    c1.title('Song Name')
    c2.title('Artist')
    c3.title('Album')
    for i in range(e.shape[0]):
       d=st.container()
       with d:
          c1,c2,c3=st.columns(3)
          c1.subheader(e['track_name'].iloc[i])
          c2.subheader(e['artists'].iloc[i])
          c3.subheader(e['album_name'].iloc[i])

