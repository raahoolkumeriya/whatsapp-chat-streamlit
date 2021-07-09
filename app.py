import re
import os
import emoji
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)

# Initial page config
st.set_page_config(
    page_title="WhatsApp Chat Processor",
    page_icon="▶",
    # layout="wide",
    initial_sidebar_state="expanded",
)

# Footer Configuration of web page
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

        
# Disbling Streamlit default style configuration
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
st.markdown('''* This app is meant as a playground to explore the whatsApp Chat.
    Try it out by **Uploading without-Media whatsapp chat export** here.''')

st.sidebar.title("WhatsApp Chat Analaysis is Data Science project:")
st.sidebar.markdown('''This application is compatible with both iOS and\
    Android exported chat.''')
st.sidebar.markdown('''** Application Feature: **
- Multilungual text support
- Individual Messenger Statistics
- Emoji's distrubution
- Message distribution w.r.t DateTime
''')

st.sidebar.markdown("** About **")
github = '[GitHub](https://github.com/raahoolkumeriya/whatsapp-chat-streamlit)'
st.sidebar.markdown(github, unsafe_allow_html=True)
twitter = '[Twitter](https://twitter.com/KumeriyaRahul)'
st.sidebar.markdown(twitter, unsafe_allow_html=True)


def extract_emojis(s):
    """
    Extract emojis from message string
    """
    return ''.join(c for c in s if c in emoji.UNICODE_EMOJI['en'])


def add_multilingual_stopwords():
    """
    Handling Multilungual configuration here.
    : Add pages into folder with different languages
    : Currently Hindi and Marathi added
    """
    multilingul_list = []
    for file in os.listdir('stopwords'):
        stopword = open('stopwords/' + file, "r")
        for word in stopword:
            word = re.sub('[\n]', '', word)
            multilingul_list.append(word)
    return multilingul_list


def generate_word_cloud(text):
    """
    Generate Word Cloud for Text
    """
    # wordcloud = WordCloud(
    #   stopwords=stopwords, background_color="white").generate(text)
    wordcloud = WordCloud(
        stopwords=stopwords,
        font_path='Laila-Regular.ttf',
        random_state=1,
        collocations=False,
        background_color="white").generate(text)
    # Display the generated image:
    # the matplotlib way:
    plt.figure(figsize=(15, 3))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot()


# Preprocessing
URLPATTERN = r'(https?://\S+)'
stopwords = set(STOPWORDS).union(set(add_multilingual_stopwords()))

uploaded_file = st.file_uploader(
    "Choose a TXT file only",
    type=['txt'],
    accept_multiple_files=False)


def main():
    """
    Function will process the txt data and process
    into Pandas Dataframe items
    """
    if uploaded_file is not None:
        # Convert txt string to utf-8 Encoding
        data = uploaded_file.getvalue().decode("utf-8")
        # Compatible iOS and Android regex search
        # Different Phone have diffent format of export
        # Hence Iphone, Android and Samsung added in the case
        regex_iphone = re.findall(
            r'''(\[\d+/\d+/\d+, \d+:\d+:\d+ [A-Z]*\]) (.*?): (.*)''', data)
        if regex_iphone:
            regex_string = regex_iphone
        regex_android = re.findall(
            r'''(\d+/\d+/\d+, \d+:\d+\d+ [a-zA-Z]*) - (.*?): (.*)''', data)
        if regex_android:
            regex_string = regex_android
        regex_samsung = re.findall(
         r'''(\d+\-\d+\-\d+, \d+:\d+ [a-zA-Z].[a-zA-Z].*) - (.*?): (.*)''',
         data)
        if regex_samsung:
            regex_string = regex_samsung
        # Convert list to dataframe and name the columns
        raw_df = pd.DataFrame(
            regex_string, columns=['DateTime', 'Author', 'Message'])
        # Convert to dataframe date w.r.t to common DateTime format
        if regex_iphone:
            raw_df['DateTime'] = pd.to_datetime(
                raw_df['DateTime'], format="[%d/%m/%y, %H:%M:%S %p]")
        if regex_android:
            raw_df['DateTime'] = pd.to_datetime(
                raw_df['DateTime'], format="%d/%m/%y, %H:%M %p")
        # --------------------------------------------------------
        # Sumsung export has p.m and a.m in data time field
        # ---------------------------------------------------------
        if regex_samsung:
            raw_df['DateTime'] = raw_df['DateTime'].str.replace(
                r'[p].[m].', 'PM')
            raw_df['DateTime'] = raw_df['DateTime'].str.replace(
                    r'[a].[m].', 'AM')
            raw_df['DateTime'] = pd.to_datetime(
                raw_df['DateTime'], format='%Y-%m-%d, %I:%M %p')
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
        rdf = raw_df.iloc[1:, :]
        # Handle EMojis
        rdf = rdf.assign(Emojis=rdf["Message"].apply(extract_emojis))
        # Handle Links
        rdf = rdf.assign(
            Urlcount=rdf["Message"].apply(
                lambda x: re.findall(URLPATTERN, x)).str.len())
        # Handle Image ommited
        rdf = rdf.assign(Media=rdf['Message'].apply(
            lambda x: re.findall("omitted", x)).str.len())
        # Handling wordCount
        media_messages_df = rdf[rdf['Message'].str.contains("omitted")]
        df = rdf.drop(media_messages_df.index)
        # Letter Count
        df['Letter_Count'] =\
            df['Message'].apply(lambda s: len(s))
        df['Word_Count'] =\
            df['Message'].apply(lambda s: len(s.split(' ')))
        st.title(f"{group_name}")
        st.text("")
        st.markdown("""---""")
        st.subheader("Raw Dataset ▶")
        with st.beta_container():
            st.dataframe(raw_df[["Author", "Message", "DateTime"]])

        with st.beta_expander("Words and Phrases frequently used ▶"):
            col1, col2 = st.beta_columns(2)
            with col1:
                st.markdown(
                    f"**Total Messages:** {df.Message.shape[0]}")
            with col2:
                st.markdown(
                    f"**Total words:** {np.sum(df.Word_Count)}")
            # handling Cloud messages
            cloud_df = df[df["Message"].str.contains("<Media omitted>|\
                This message was deleted|You deleted this message|\
                    Missed voice call|Missed video call") == False]
            text = " ".join(review for review in cloud_df.Message)
            generate_word_cloud(text)

        with st.beta_expander("Individual Statistic ▶"):
            sorted_authors = df.groupby('Author')['Message'].count()\
                .sort_values(ascending=False).index
            st.markdown("### **Select Member:**")
            select_author = []

            select_author.append(st.selectbox('', sorted_authors))
            col1, col2 = st.beta_columns(2)
            with col1:
                st.markdown(f"**Total Messages:** {df.shape[0]}")
                st.markdown(f"""**Messages:**
                    {df[df['Author'] == select_author[0]].shape[0]}""")
                st.markdown(f'''**Media Shared:**
                    {df[df['Media'] == select_author[0]].shape[0]}''')

            with col2:
                st.markdown(f"**Highest Messanger:** {sorted_authors[0]}")
                st.markdown(f'''**Emoji's Shared:**
                    {sum(df[df['Author'] == select_author[0]]
                    .Emojis.str.len())}''')
                st.markdown(f"""**Link Shared:**
                    {df[df['Urlcount'] == select_author[0] ].shape[0]}""")
            dummy_df = df[df['Author'] == select_author[0]]
            # handling Cloud messages
            cloud_df = dummy_df[dummy_df["Message"].str.contains("<Media omitted>|\
                This message was deleted|You deleted this message|\
                    Missed voice call|Missed video call") == False]
            text = " ".join(review for review in cloud_df.Message)
            st.text("")
            fig = generate_word_cloud(text)
            st.write(fig)

        # View Data uploaded
        with st.beta_expander("Histogram Plot ▶"):
            st.text("")
            hour = st.slider("selected_hour", 0, 0, 23, 1)
            hist_data = df[df['DateTime'].dt.hour == hour]
            if st.checkbox('view_data'):
                st.subheader(f"Raw Data at {hour}")
                st.write(hist_data)
            st.subheader(f"Data by Minute at {hour}")
            st.bar_chart(
                np.histogram(
                 hist_data['DateTime'].dt.minute, bins=60, range=(0, 60))[0])

        st.text("")
        with st.beta_expander("Emojis distribution accross group▶"):
            total_emojis_list = list(set([a for b in df.Emojis for a in b]))
            st.subheader(f" Total Emoj's Shared: {len(total_emojis_list)}")

            total_emojis_list = list([a for b in df.Emojis for a in b])
            emoji_dict = dict(Counter(total_emojis_list))
            emoji_dict = sorted(
                emoji_dict.items(), key=lambda x: x[1], reverse=True)
            # for i in emoji_dict:
            #  print(i)
            emoji_df = pd.DataFrame(emoji_dict, columns=['emojis', 'count'])
            fig = px.pie(emoji_df, values='count', names='emojis')
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig)

        st.text("")
        with st.beta_expander("Message Distribution w.r.t DateTime ▶"):
            st.markdown("### **Select Member:**")
            selector = []
            selector.append(st.selectbox('', ['Date', "DateTime"]))
            z = df[selector[0]].value_counts()
            # converts to dictionary
            z1 = z.to_dict()
            df['Msg_count'] = df[selector[0]].map(z1)
            # Timeseries plot
            fig = px.line(x=df[selector[0]], y=df['Msg_count'])
            fig.update_layout(
                title='Analysis of number of messages using TimeSeries plot.',
                xaxis_title='Month',
                yaxis_title='No. of Messages')
            fig.update_xaxes(nticks=20)
            st.write(fig)


if __name__ == "__main__":
    main()
