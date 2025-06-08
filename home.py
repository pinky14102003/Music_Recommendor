import streamlit as st
import numpy as np
import pandas as pd
# from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



st.set_page_config(layout='wide')

# s=pd.read_csv('dataset.csv')
# s=s.drop_duplicates(subset=['artists','track_name']).reset_index(drop=True)
# s.dropna(inplace=True)
# s.drop(columns=['Unnamed: 0'],inplace=True)
# s['artists']=s['artists'].apply(lambda x:x.split(';'))
# s['artists']=s['artists'].apply(lambda x:[i.replace(' ','')for i in x])
# s['artists']=s['artists'].apply(lambda x:" ".join(x))
# new_df=s[['danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence']]
# scaler = StandardScaler()
# scaled_data = scaler.fit_transform(new_df)
# kmeans=KMeans(n_clusters=7,init='k-means++',random_state=42)
# kmeans.fit(scaled_data)
# predicted_data=kmeans.predict(scaled_data)
# s['group']=pd.Series(predicted_data)
# def recommend(song):
#   target_group = s[s['track_name']==song].sort_values(by='popularity',ascending=False)['group'].head(1).iloc[0]
#   return s[s['group']==target_group].head(5)[['track_name','album_name','artists']]



s=pd.read_csv('dataset.csv')
s=s.drop_duplicates(subset=['track_name','artists']).reset_index(drop=True)
s=s.drop(columns=['Unnamed: 0'])
s.dropna(inplace=True)
s['artists']=s['artists'].apply(lambda x:x.split(';'))
s['artists']=s['artists'].apply(lambda x:" ".join(x))
de=s[['danceability', 'energy','loudness','speechiness','acousticness','instrumentalness', 'liveness', 'valence']]
ss=StandardScaler()
de=ss.fit_transform(de)
from sklearn.neighbors import NearestNeighbors
nn=NearestNeighbors(n_neighbors=10,metric='euclidean')
nn.fit(de)
s[['danceability', 'energy','loudness','speechiness','acousticness','instrumentalness', 'liveness', 'valence']]=pd.DataFrame(de)
def basicrecommend(name):
  p=s[s['track_name']==name]
  p=p.sort_values(by='popularity',ascending=False).head(1)
  p=p[['danceability', 'energy','loudness','speechiness','acousticness','instrumentalness', 'liveness', 'valence']]
  print(p)
  distances, indices = nn.kneighbors(p)
  return s.iloc[[int(i) for i in indices[0]]][['track_id','track_name','album_name','artists']]











st.title('Song Recommendor System')
a=st.selectbox('Enter Your Song Name',options=s['track_name'])
b=st.button('Search')
if b:
    e=basicrecommend(a)
    e=e.reset_index(drop=True)
    st.header("You May also like")
    st.dataframe(e)
    # c1,c2,c3=st.columns(3)
    # c1.title('Song Name')
    # c2.title('Artist')
    # c3.title('Album')
    # for i in range(e.shape[0]):
    #    d=st.container()
    #    with d:
    #       c1,c2,c3=st.columns(3)
    #       c1.subheader(e.loc[i,'track_name'])
    #       c2.subheader(e.loc[i,'artists'])
    #       c3.subheader(e.loc[i,'album_name'])

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
       


import streamlit as st
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Set up Spotify credentials
SPOTIPY_CLIENT_ID = '3e42d2e4c66c4bcebb10ba7c216edf86'
SPOTIPY_CLIENT_SECRET = '872d5fe12bae4aa5abd92359f0c39f7f'
SPOTIPY_REDIRECT_URI = 'https://music-recommendor-pjxy.onrender.com/'  # Your Streamlit app URL

# Authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope="user-read-playback-state,user-modify-playback-state"
))

def play_track(track_id):
    sp.start_playback(uris=[f'spotify:track:{track_id}'])

track_id = st.text_input("Enter Spotify Track ID")
if st.button("Play Song") and track_id:
    play_track(track_id)
    st.success(f"Playing track {track_id}")
   
