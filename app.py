import re
import os
import warnings
import streamlit as st
from chat_eda import WhatsApp
from numpy import sum as npsum
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

st.set_option('deprecation.showPyplotGlobalUse', False)

warnings.filterwarnings(
    "ignore", message="Glyph 128584 missing from current font.")

# Initial page config
st.set_page_config(
    page_title="WhatsApp Chat Processor",
    page_icon="",
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
                padding: 5px;
                top: 5px;
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


st.title('WhatsApp Chat Processor')
st.markdown("Messages in your chat box, **says something?**...Let's find out")
st.markdown("**♟ General Statistics ♟**")
st.markdown('''* This app is meant as a playground to explore the whatsApp Chat.
    Try it out by **Uploading without-Media whatsapp chat export** here.''')

st.sidebar.title("WhatsApp Chat Analaysis is Data Science project:")
st.sidebar.markdown('''This application is compatible with both iOS and\
    Android exported chat.''')
st.sidebar.markdown('''** Application Feature: **
- English/Marathi/Hindi support in wordcloud
- Individual Messenger Statistics
- Graphs
- Emoji's distrubution
- Time series analysis
- Sentiment analysis
''')

link = '[GitHub](https://github.com/raahoolkumeriya/whatsapp-chat-streamlit)'
st.sidebar.markdown("Source code")
st.sidebar.markdown(link, unsafe_allow_html=True)


class WordCloudDisplay:
    """
    Word Cloud Display
    """
    def add_multilingual_stopwords(self):
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

    def generate_word_cloud(self, text, title):
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
            background_color="white").generate_from_text(text)
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


def main():
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
        col1, col2 = st.columns(2)
        col1.metric(
            "Total Delted messages",
            stats.get('total_deleted_messages'), "+/-1%")
        col2.metric(
            "Your Deleted Messages",
            stats.get('your_deleted_message'), "+/-1%")

        st.markdown("----")

        cloud_df = w.cloud_data(raw_df)

        # SECTION 3: Frequenctly use words
        st.header("🔘 Frequently used words")
        sorted_authors = w.sorted_authors(df)
        st.write("Select Member to see its Statistics")
        select_author = []

        select_author.append(st.selectbox('', sorted_authors))
        col1, col2 = st.columns([2, 2])
        with col1:
            st.markdown(f"**Total Messages:** {df.shape[0]}")
            st.markdown(f"""**Messages:**
                {df[df['name'] == select_author[0]].shape[0]}""")
            number = df.groupby(
                by=["name"], dropna=False).sum()['media'].get(
                    select_author[0])
            st.markdown(f"**Media Shared:** {number}")
            user_df = df[df.name == select_author[0]]
            average = round(npsum(user_df.word_count)/user_df.shape[0], 2)
            st.markdown(f"**Average words/Message:** {average}")

        with col2:
            st.markdown(f"**Highest Messanger:** {sorted_authors[0]}")
            st.markdown(f"""**Emoji's Shared:**
                {sum(df[df['name'] == select_author[0]]
                .emojis.str.len())}""")
            st.markdown(f"""**Link Shared:**
                {df[df['urlcount'] == select_author[0] ].shape[0]}""")
        dummy_df = cloud_df[cloud_df['name'] == select_author[0]]
        text = " ".join(review for review in dummy_df.message)
        st.text("")

        gen_wordcloud.generate_word_cloud(
            text, "Word Cloud for individual Words")

        st.markdown("----")
        st.header("🔘 Words and Phrases frequently used in Chat")
        text = " ".join(review for review in cloud_df.message)
        gen_wordcloud.generate_word_cloud(
            text, "Word Cloud for Chat words")

        st.header("🔘 Most Active Member")
        st.pyplot(w.most_active_member(df))

        st.header("🔘  Most Active Day")
        w.day_analysis(df)
        st.pyplot(w.most_active_day(df))

        st.header("🔘 Who uses more words in sentences")
        st.pyplot(w.max_words_used(df))

        st.header("🔘 Top-10 Media Contributor ")
        st.pyplot(w.top_media_contributor(raw_df))

        st.header("🔘 Who shares Links in group most? ")
        st.pyplot(w.who_shared_links(df))

        st.header("🔘 Who has Positive Sentiment? ")
        st.pyplot(w.sentiment_analysis(cloud_df))

        st.header("Group highly Active time ")
        st.pyplot(w.time_when_group_active(df))

        st.header("🔘 Most Active Day ")
        st.pyplot(w.most_suitable_day(df))

        st.header("🔘 Most Active Hour")
        st.pyplot(w.most_suitable_hour(df))

        st.header("🔘 Over the Time Analysis ")
        st.write(w.time_series_plot(df))

        st.header("🔘 Curious about Emoji's ?")
        st.write(
            "The most use Emoji's in converstion is show with larger sector")
        pie_display = w.pie_display_emojis(df)
        st.plotly_chart(pie_display)

        st.write("🔘 Much more to Come ...")


if __name__ == "__main__":
    import _banner
    main()