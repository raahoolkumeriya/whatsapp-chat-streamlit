import re
import os
import _banner
import warnings
import streamlit as st
from chat_eda import WhatsApp
from typing import Dict, Any
from numpy import sum as npsum
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


st.set_option('deprecation.showPyplotGlobalUse', False)

warnings.filterwarnings(
    "ignore", message="Glyph 128584 missing from current font.")

TITLE = "WhatsApp Chat Processor"
# Initial page config
st.set_page_config(
    page_title=TITLE,
    page_icon="",
    # layout="wide",
    initial_sidebar_state="expanded",
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'copyrights © 2021 rahul.kumeriya@outlook.com [ DONT FORGET TO PLANT TREES ]' ;
                visibility: visible;
                display: block;
                position: fixed;
                #background-color: red;
                padding: 5px;
                top: 0;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.cache(allow_output_mutation=True)
padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)


st.title(TITLE)
st.header("Messages in your chat group, `says something?`...Let's find out")
st.subheader("**♟ General Statistics ♟**")
st.write('''* This app is meant as a playground to explore the whatsApp Chat.
    Try it out by `Uploading WITHOUT MEDIA whatsapp chat export` here.''')

st.sidebar.title("WhatsApp Chat Processor is a Data Science project for Fun")
st.sidebar.markdown('''This application is compatible with both `iOS` and\
    `Android` device exported chat.''')
st.sidebar.markdown('''** Application Feature: **
- Multilingual (Top-50 Languages) support in wordcloud
- Individual Messenger Statistics
- Activity Cluster
- Emoji's distrubution
- Time series analysis
- Sentiment Score of Member
''')



st.sidebar.markdown("`View Code on Github`")
st.sidebar.markdown('<iframe src="https://ghbtns.com/github-btn.html?user=raahoolkumeriya&repo=whatsapp-chat-streamlit&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>', unsafe_allow_html=True)
st.sidebar.markdown('<iframe src="https://ghbtns.com/github-btn.html?user=raahoolkumeriya&repo=whatsapp-chat-streamlit&type=star&count=true&size=large" frameborder="0" scrolling="0" width="150" height="30" title="GitHub"></iframe>', unsafe_allow_html=True)
st.sidebar.markdown('<iframe src="https://ghbtns.com/github-btn.html?user=raahoolkumeriya&repo=whatsapp-chat-streamlit&type=watch&count=true&size=large&v=2" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>', unsafe_allow_html=True)
st.sidebar.markdown('<iframe src="https://ghbtns.com/github-btn.html?user=raahoolkumeriya&repo=whatsapp-chat-streamlit&type=fork&count=true&size=large" frameborder="0" scrolling="0" width="170" height="30" title="GitHub"></iframe>', unsafe_allow_html=True)

class WordCloudDisplay:
    """
    Word Cloud Display
    """
    def add_multilingual_stopwords(self) -> Dict:
        """
        Function add Hindi, Marathi for the moment as
        Multilingula support for stopwords
        """
        multilingul_list = []
        for file in os.listdir('stopwords'):
            stopword = open('stopwords/' + file, "r")
            for word in stopword:
                word = re.sub('[\n]', '', word)
                multilingul_list.append(word)
        return set(STOPWORDS).union(set(multilingul_list))

    def generate_word_cloud(self, text: str, title: str) -> Any:
        """
        Generate Word Cloud for Text
        """
        # wordcloud = WordCloud(
        #   stopwords=stopwords, background_color="white").generate(text)
        wordcloud = WordCloud(
            scale=3,
            width=500,
            height=330,
            max_words=200,
            colormap='tab20c',
            stopwords=self.add_multilingual_stopwords(),
            collocations=True,
            contour_color='#5d0f24',
            contour_width=3,
            font_path='Laila-Regular.ttf',
            background_color="white").generate(text)
        # Display the generated image:
        # the matplotlib way:
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.title(title)
        st.pyplot()


gen_wordcloud = WordCloudDisplay()

uploaded_file = st.file_uploader(
    "Choose a TXT file only",
    type=['txt'],
    accept_multiple_files=False)


def next_page():
    st.session_state.page += 1


def prev_page():
    st.session_state.page -= 1


def main() -> None:
    """
    Function will process the txt data and process into
    Pandas Dataframe items
    """
    if uploaded_file is not None:
        # Convert txt string to utf-8 Encoding
        data = uploaded_file.getvalue().decode("utf-8")
        # Compatible iOS and Android regex search

        w = WhatsApp()
        message = w.apply_regex(data)
        raw_df = w.process_data(message)
        df = w.get_dataframe(raw_df)
        stats = w.statistics(raw_df, df)

        st.markdown(f'# {stats.get("group_name")}')

        st.markdown("----")

        # Pagination Code
        if "page" not in st.session_state:
            st.session_state.page = 0
        col1, _, _, col2, _, col3 = st.columns(6)
        if st.session_state.page < 4:
            col3.button("Next", on_click=next_page)
        else:
            col3.write("")  # this makes the empty column show up on mobile
        if st.session_state.page > 0:
            col1.button("Previous", on_click=prev_page)
        else:
            col1.write("")  # this makes the empty column show up on mobile
        col2.write(f"Page {1+st.session_state.page} of {5}")
        start = 10 * st.session_state.page
        end = start + 10
        st.write("")

        st.dataframe(raw_df[["name", "message", "datetime"]].iloc[start:end])

        st.markdown("#")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Total Messages", stats.get('total_messages'), delta="📦 📨")
        col2.metric(
            "Total Members", stats.get('total_members'), "💃🕺")
        col3.metric(
            "Total Media", stats.get('media_message'), delta="🎞️ 📷")
        col4.metric(
            "Link shared", int(stats.get('link_shared')), delta="🖇️ 🔗")
        st.text("")
        # col1, col2 = st.columns(2)
        # col1.metric(
        #     "Total Delted messages",
        #     stats.get('total_deleted_messages'), "+/-1%")
        # col2.metric(
        #     "Your Deleted Messages",
        #     stats.get('your_deleted_message'), "+/-1%")
        st.text("")
        cloud_df = w.cloud_data(raw_df)

        # SECTION 3: Frequenctly use words
        st.header("🔘 Frequently used words")
        sorted_authors = w.sorted_authors(cloud_df)
        select_author = []

        select_author.append(st.selectbox('', sorted_authors))
        dummy_df = cloud_df[cloud_df['name'] == select_author[0]]
        text = " ".join(review for review in dummy_df.message)

        col1, col2, col3, col4, col5 = st.columns(5)
    
        col1.metric(
        "Posted Messages", dummy_df[dummy_df['name'] == select_author[0]].shape[0])
        col2.metric("Emoji's Shared",
            sum(df[df.name.str.contains(select_author[0][-5:])]
            .emojis.str.len()))
    
        col3.metric("Link Shared", int(df[df.name == select_author[0]].urlcount.sum()))
        col4.metric("Total Words", int(df[df.name == select_author[0]].word_count.sum()))
        user_df = df[df.name.str.contains(select_author[0][-5:])]
        average = round(npsum(user_df.word_count)/user_df.shape[0], 2)

        col5.metric("Average words/Message", average)
            
        if len(text) != 0:
            gen_wordcloud.generate_word_cloud(
                text, "Word Cloud for individual Words")
        else:
            gen_wordcloud.generate_word_cloud(
                "NOWORD", "Word Cloud for individual Words")

        st.markdown("----")
        st.header("🔘 Words and Phrases frequently used in Chat")
        st.info("🔋 Frequently used words or phrases by all members in group chat.\
            Most dicussion occurs around below words or used frequently.")
        text = " ".join(review for review in cloud_df.message)
        gen_wordcloud.generate_word_cloud(
            text, "Word Cloud for Chat words")

        st.markdown("----")
        st.header("🔘 Most Active Member")
        st.info("🔋 Member comparision based on the number of messages\
            he/she posted in group chat")
        st.pyplot(w.most_active_member(df))

        st.markdown("----")
        st.header("🔘  Most Active Day")
        st.info("🔋 Member comparision based on the number of messages\
            he/she posted in group chat w.r.t Day")
        w.day_analysis(df)
        st.pyplot(w.most_active_day(df))

        st.markdown("----")
        st.header("🔘 Who uses more words in sentences")
        st.info("🔋 Member uses more number of sentences during the conversation")
        st.pyplot(w.max_words_used(df))

        st.markdown("----")
        st.header("🔘 Top-10 Media Contributor ")
        st.info("🔋 Comparision of members who contributes more number of Images,\
            Video or Documents")
        st.pyplot(w.top_media_contributor(raw_df))

        st.markdown("----")
        st.header("🔘 Who shares Links in group most? ")
        st.info("🔋 Members who shares internet links of information with others")
        st.pyplot(w.who_shared_links(df))

        st.markdown("----")
        st.header("🔘 Who has Positive Sentiment? ")
        st.info("🔋 Member sentiment analysis score base on the words used in\
            messages. Sentiment Score above 0.5 to 1 is consider as Positive.\
            Pure English words and Phrases is ideal for calcalation")
        st.pyplot(w.sentiment_analysis(cloud_df))

        # st.markdown("----")
        # st.header("🔘 Group highly Active time ")
        # st.pyplot(w.time_when_group_active(df))

        st.markdown("----")
        st.header("🔘 Most Active Day ")
        st.info("🔋 Member who active for suitable Day")
        st.pyplot(w.most_suitable_day(df))

        st.markdown("----")
        st.header("🔘 Most Active Hour")
        st.info("🔋 Member who active during suitable hours")
        st.pyplot(w.most_suitable_hour(df))

        st.markdown("----")
        st.header("🔘 Member activity Cluster")
        st.info("🔋 Cluster hover about the total messages, Emoji's, Links, Words\
            and Letter by individual member")
        st.write(w.message_cluster(df))

        st.markdown("----")
        st.header("🔘 Over the Time Analysis ")
        st.info("🔋 Group activity over the time w.r.t to number of messages")
        st.write(w.time_series_plot(df))

        st.markdown("----")
        st.header("🔘 Curious about Emoji's ?")
        st.info("🔋 The most use Emoji's in converstion is show with\
            larger sector")
        pie_display = w.pie_display_emojis(df)
        st.plotly_chart(pie_display)

        st.header("🔘 Take out some time to plant Trees 🌲🌳🌴🌵")
        st.success("🌳 I already did, now it's your turn ?\
            🌿🌾☘️")


if __name__ == "__main__":
    print(_banner.display_banner)
    main()
