import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import emoji
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.express as px
st.set_option('deprecation.showPyplotGlobalUse', False)

# Initial page config

st.set_page_config(
    page_title="WhatsApp Chat Processor", 
    page_icon="▶", 
    # layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'copyrights © 2021 rahul.kumeriya@outlook.com'; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 10px; 
                top: 12px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)

st.title('WhatsApp Chat Processor')
st.markdown("Messages in your chat box, **says something?**...Let's find out")
st.markdown("**♟ General Statistics ♟**")
st.markdown("* This app is meant as a playground to explore the whatsApp Chat. Try it out by **Uploading without-Media whatsapp chat export** here.") 

st.sidebar.title("WhatsApp Chat Analaysis is Data Science project:")
st.sidebar.markdown("This application is compatible with both iOS and Android exported chat.")
st.sidebar.markdown('''** Application Feature: **
- English/Marathi/Hindi Language vistualization
- Individual Messenger Statistics 
- Emoji's distrubution
- Message distribution w.r.t Date time
''')

link = '[GitHub](https://github.com/raahoolkumeriya/whatsapp-chat-streamlit)'
st.sidebar.markdown("Source code")
st.sidebar.markdown(link, unsafe_allow_html=True)

def extract_emojis(s):
    """Extract emojis from message string"""
    return ''.join(c for c in s if c in emoji.UNICODE_EMOJI['en'])

def add_multilingual_stopwords():
    multilingul_list = []
    for file in os.listdir('stopwords'):
        stopword = open('stopwords/'+ file, "r") 
        for word in stopword:
            word = re.sub('[\n]','',word)
            multilingul_list.append(word)
    return multilingul_list

       
def generate_word_cloud(text):
        """Generate Word Cloud"""
        #wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
        wordcloud = WordCloud(stopwords=stopwords,font_path='Laila-Regular.ttf',background_color="white").generate(text)
        # Display the generated image:
        # the matplotlib way:
        plt.figure( figsize=(15,3))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot()

# Preprocessing 
URLPATTERN = r'(https?://\S+)'
stopwords = set(STOPWORDS).union(set(add_multilingual_stopwords()))

uploaded_file = st.file_uploader("Choose a TXT file only",type=['txt'], accept_multiple_files=False)


def main():
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
        # Convert time to IST
        raw_df['DateTime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata')
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
        df = df.assign(
            Urlcount=df["Message"].apply(lambda x: re.findall(URLPATTERN, x)).str.len())
        # Handle Image ommited
        df = df.assign(Media=df['Message'].apply(lambda x: re.findall("image omitted", x)).str.len()) 
        # Handling wordCount
        media_messages_df = df[df['Message'].str.contains("image omitted")]
        messages_df = df.drop(media_messages_df.index)
        messages_df['Letter_Count'] = messages_df['Message'].apply(lambda s : len(s))
        messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(s.split(' ')))
        

        
        st.title(f"{group_name}")
        st.text("")

        st.subheader("Raw Dataset ▶")
        with st.beta_container():
            st.dataframe(raw_df[["Author", "Message", "DateTime"]])


        with st.beta_expander("Words and Phrases frequently used ▶"):
            col1, col2 = st.beta_columns(2)        
            with col1:
                st.markdown(f"**Total Messages:** {messages_df.Message.shape[0]}")
            with col2:
                st.markdown(f"**Total words:** {np.sum(messages_df.Word_Count)}")
            text = " ".join(review for review in messages_df.Message)
            generate_word_cloud(text)

        
        with st.beta_expander("Individual Statistic ▶"):
        
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
                st.markdown(f"**Highest Messanger:** {sorted_authors[0]}")
                st.markdown(f"**Emoji's Shared:** {sum(messages_df[messages_df['Author'] == select_author[0]].Emojis.str.len())}")
                st.markdown(f"""**Link Shared:** {messages_df[messages_df['Urlcount'] == select_author[0] ].shape[0]}""")
            dummy_df = messages_df[messages_df['Author'] == select_author[0]] 
            text = " ".join(review for review in dummy_df.Message)

            st.text("")
            fig = generate_word_cloud(text)
            st.write(fig)

        # View Data uploaded
        with st.beta_expander(f"Histogram Plot ▶"):
            st.text("")
            hour = st.slider("selected_hour", 0, 0, 23, 1)
            hist_data = messages_df[messages_df['DateTime'].dt.hour == hour]
            if st.checkbox('view_data'):
                st.subheader(f"Raw Data at {hour}")
                st.write(hist_data)
            st.subheader(f"Data by Minute at {hour}")
            st.bar_chart(
                np.histogram(hist_data['DateTime'].dt.minute, bins=60, range=(0,60))[0])

        st.text("")
        with st.beta_expander("Emojis distribution accross group▶"):
            total_emojis_list = list(set([a for b in messages_df.Emojis for a in b]))
            st.subheader(f" Total Emoj's Shared: {len(total_emojis_list)}")

            total_emojis_list = list([a for b in messages_df.Emojis for a in b])
            emoji_dict = dict(Counter(total_emojis_list))
            emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
            #for i in emoji_dict:
            #  print(i)
            emoji_df = pd.DataFrame(emoji_dict, columns=['emojis', 'count'])
            import plotly.express as px
            fig = px.pie(emoji_df, values='count', names='emojis')
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig)


        st.text("")
        with st.beta_expander("Message Distribution w.r.t DateTime ▶"):
            st.markdown("### **Select Member:**")
            selector = []
            selector.append(st.selectbox('', ['Date', "DateTime"]))
        
            z = messages_df[selector[0]].value_counts() 
            z1 = z.to_dict() #converts to dictionary
            df['Msg_count'] = df[selector[0]].map(z1)
            ### Timeseries plot 
            fig = px.line(x=df[selector[0]],y=df['Msg_count'])
            fig.update_layout(title='Analysis of number of messages using TimeSeries plot.',
                            xaxis_title='Month',
                            yaxis_title='No. of Messages')
            fig.update_xaxes(nticks=20)
            st.write(fig)


if __name__ == "__main__":
    main()