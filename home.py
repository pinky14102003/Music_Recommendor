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
nn=NearestNeighbors(n_neighbors=10,metric='cosine')
nn.fit(de)
s[['danceability', 'energy','loudness','speechiness','acousticness','instrumentalness', 'liveness', 'valence']]=pd.DataFrame(de)
def basicrecommend(name):
  p=s[s['track_name']==name]
  p=p.sort_values(by='popularity',ascending=False).head(1)
  p=p[['danceability', 'energy','loudness','speechiness','acousticness','instrumentalness', 'liveness', 'valence']]
  print(p)
  distances, indices = nn.kneighbors(p)
  return s.iloc[[int(i) for i in indices[0]]][['track_id','track_name','album_name','artists','track_genre']]











st.title('Song Recommendor System')
a=st.selectbox('Enter Your Song Name',options=s['track_name'])
b=st.button('Search')
if b:
    e=basicrecommend(a)
    z=e['track_id'].reset_index(drop=True)
    e=e.reset_index(drop=True)
    st.header("You May also like")
    st.dataframe(e)
    for i in range(len(z)):
      spotify_url = f"https://open.spotify.com/track/{z.loc[i]}"
      st.markdown(f"[{e.loc[i,'track_name']}]({spotify_url})")
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
       
