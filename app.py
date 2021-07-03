import streamlit as st
import pandas as pd
import numpy as np
import re
import emoji
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
st.set_option('deprecation.showPyplotGlobalUse', False)

# Initial page config

st.set_page_config(
    page_title="WhatsApp Chat Processor", 
    page_icon="▶", 
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title('WhatsApp Chat Processor')
st.markdown("Daily messages in your chat box, **says something?**")
st.markdown("**♟ General Statistics ♟**")
st.markdown("* This app is meant as a playground to explore the whatsApp Chat. Try it out by **Uploading without-Media whatsapp chat export** here.") 

URLPATTERN = r'(https?://\S+)'

def extract_emojis(s):
    """Extract emojis from message string"""
    return ''.join(c for c in s if c in emoji.UNICODE_EMOJI['en'])

def generate_word_cloud(text):
        """Generate Word Cloud"""
        # print ("There are {} words in all the messages.".format(len(text)))
        stopwords = set(STOPWORDS)
        # Generate a word cloud image
        wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
        # Display the generated image:
        # the matplotlib way:
        plt.figure( figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()


uploaded_file = st.file_uploader("Choose a TXT file", accept_multiple_files=False)


if uploaded_file is not None:
    data = uploaded_file.getvalue().decode("utf-8")

    regex_iphone = re.findall('(\[\d+/\d+/\d+, \d+:\d+:\d+ [A-Z]*\]) (.*?): (.*)', data)
    if regex_iphone:
        regex_string = regex_iphone
    regex_android = re.findall('(\d+/\d+/\d+, \d+:\d+\d+ [a-zA-Z]*) - (.*?): (.*)', data)
    if regex_android:
        regex_string = regex_android
    
    # Convert list to dataframe and name teh columns
    raw_df = pd.DataFrame(regex_string,columns=['DateTime','Author','Message'])
    # Convert to dataframe date
    if regex_iphone:
        raw_df['DateTime'] = pd.to_datetime(raw_df['DateTime'],format="[%d/%m/%y, %H:%M:%S %p]")
    if regex_android:
        raw_df['DateTime'] = pd.to_datetime(raw_df['DateTime'],format="%d/%m/%y, %H:%M %p")

    # Splitting Date and Time 
    raw_df['Date'] = pd.to_datetime(raw_df['DateTime']).dt.date
    raw_df['Time'] = pd.to_datetime(raw_df['DateTime']).dt.time
    # Drop DateTime Column
    # raw_df.drop('DateTime',axis='columns', inplace=True)
    # Drop NAN Values 
    raw_df = raw_df.dropna()
    raw_df = raw_df.reset_index(drop=True)
    # Handling *Messages and calls are end-to-end encrypted* 
    group_name = raw_df.iloc[0:1]['Author'][0]
    df = raw_df.iloc[1: , :]
    # Handle EMojis
    df = df.assign(Emojis=df["Message"].apply(extract_emojis))
    # Handle Links
    df = df.assign(Urlcount=df["Message"].apply(lambda x: re.findall(URLPATTERN, x)).str.len())
    # Handle Image ommited
    df = df.assign(Media=df['Message'].apply(lambda x: re.findall("image omitted", x)).str.len())
    
    # Handling wordCount         
    media_messages_df = df[df['Message'].str.contains("image omitted")]
    messages_df = df.drop(media_messages_df.index)
    messages_df['Letter_Count'] = messages_df['Message'].apply(lambda s : len(s))
    messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(s.split(' ')))
    

    
    st.title(f"{group_name}")
    st.text("")

    # View Data uploaded
    with st.beta_expander(f"Chat Processor ▶"):
        st.text("")
        st.write(raw_df[["Author", "Message", "DateTime"]])


    with st.beta_expander(f"Words and Phrases frequently used ▶"):
        st.text("")
        text = " ".join(review for review in messages_df.Message)
        generate_word_cloud(text)

    st.text("")
    with st.beta_expander("Individual Statistic ▶"):
        st.text("")
        sorted_authors = messages_df.groupby('Author')['Message'].count()\
        .sort_values(ascending=False).index
        st.markdown("### **Select Member:**")
        select_author = []

        select_author.append(st.selectbox('', sorted_authors))
        
        #Filter df based on selection
        # select_author = messages_df[messages_df['Author'].isin(select_author)]

        col1, col2 = st.beta_columns(2)
            
        with col1:
            st.markdown(f"**Total Messages:** {messages_df.shape[0]}")
            st.markdown(f"**Messages:** {messages_df[messages_df['Author'] == select_author[0] ].shape[0]}")
            st.markdown(f"**Media Shared:** {messages_df[messages_df['Media'] == select_author[0] ].shape[0]}")

        with col2:
            st.markdown(f"**Highest Messanger:** {np.min(messages_df['Author'])}")
            st.markdown(f"**Emoji's Shared:** {sum(messages_df[messages_df['Author'] == select_author[0]].Emojis.str.len())}")
            st.markdown(f"**Link Shared:** {messages_df[messages_df['Urlcount'] == select_author[0] ].shape[0]}")
            
        
        dummy_df = messages_df[messages_df['Author'] == select_author[0] ] 
        text = " ".join(review for review in dummy_df.Message)

        st.text("")
        fig = generate_word_cloud(text)
        st.write(fig)


    # View Data uploaded
    
    with st.beta_expander(f"Histogram Plot ▶"):
        st.text("")
        hour = st.slider("selected_hour",0,0,23,1)
        hist_data = messages_df[messages_df['DateTime'].dt.hour == hour]
        if st.checkbox('view_data'):
            st.subheader(f"Raw Data at {hour}")
            st.write(hist_data)
            
        st.subheader(f"Data by Minute at {hour}")
        st.bar_chart(np.histogram(hist_data['DateTime'].dt.minute, bins=60, range=(0,60))[0])

    st.text("")
    with st.beta_expander("Chats [Under Development] ▶"):
        st.text("")
