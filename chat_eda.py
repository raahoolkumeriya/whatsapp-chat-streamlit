# Import support labaraies
import re
import emoji
import logging
import numpy as np
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from typing import List, Dict
from collections import Counter
import matplotlib.pyplot as plt


class WhatsApp:
    """
    Exploratory Data Analysis for WhatsApp chat
    """
    def __init__(self):
        logging.info("WhatsApp Chat initiate")
        self.URLPATTERN = r'(https?://\S+)'
        self.weeks = {
            0: 'Monday',
            1: 'Tuesday',
            2: 'Wednesday',
            3: 'Thrusday',
            4: 'Friday',
            5: 'Saturday',
            6: 'Sunday'
        }
        # Regex for iOS, Android, Samsung chat export
        self.regex_list = [
            r'(\d+\-\d+\-\d+, \d+:\d+ [a-zA-Z].[a-zA-Z].*) - (.*?): (.*)',
            r'(\d+/\d+/\d+, \d+:\d+\d+ [a-zA-Z]*) - (.*?): (.*)',
            r'(\[\d+/\d+/\d+, \d+:\d+:\d+ [A-Z][A-Z]\]) (.*?): (.*)',
            r'(\d+/\d+/\d+, \d+:\d+ [a-zA-Z][a-zA-Z]) - (.*?): (.*)'
        ]
        self.ignore = [
            'Missed video call', 'Missed group video call',
            'Missed voice call', '<Media omitted>',
            'This message was deleted', 'image omitted',
            'You deleted this message']

    def apply_regex(self, data: str) -> List:
        """
        Attributes
        ----------
        data (str) : Input data string read from file

        Return
        ------
        list of data string
        """
        for reg in self.regex_list:
            out = re.findall(reg, data)
            if out != []:
                logging.info(f"Regex matched : {reg} ")
                return re.findall(reg, data)

    def extract_emojis(self, s: str) -> str:
        """
        Extract emojis from message string
        """
        logging.info("WhatsApp/extract_emojis()")
        return ''.join(c for c in s if c in emoji.UNICODE_EMOJI['en'])

    def process_data(self, messages):
        """
        Converting string messages into DataFrame
        """
        logging.info("WhatsApp/process_data()")
        raw_df = pd.DataFrame(messages, columns=['datetime', 'name', 'message'])
        # SAMSUNG Export time format
        try:
            raw_df['datetime'] = raw_df['datetime'].str.replace(
                r'[p].[m].', 'PM', regex=True)
            raw_df['datetime'] = raw_df['datetime'].str.replace(
                r'[a].[m].', 'AM', regex=True)
            raw_df['datetime'] = pd.to_datetime(
                raw_df['datetime'], format="%Y-%m-%d, %I:%M %p")
        except Exception:
            pass
        # IOS Export time format
        try:
            # Drop date enclosures from date column
            raw_df['datetime'] = raw_df['datetime'].map(
                lambda x: x.lstrip('[').rstrip(']'))
            raw_df['datetime'] = pd.to_datetime(
                raw_df['datetime'], format="%d/%m/%y, %I:%M:%S %p")
        except Exception:
            pass
        # OppO Export time format
        try:
            raw_df['datetime'] = pd.to_datetime(
                raw_df['datetime'], format="%d/%m/%Y, %I:%M %p")
        except Exception:
            pass
        # Android Export time format
        try:
            raw_df['datetime'] = pd.to_datetime(
                raw_df['datetime'], format="%d/%m/%y, %I:%M %p")
        except Exception:
            pass
        raw_df['date'] = pd.to_datetime(raw_df['datetime']).dt.date
        raw_df['time'] = pd.to_datetime(raw_df['datetime']).dt.time
        return raw_df

    def get_dataframe(self, raw_df):
        """
        Formation of Clean DataFrame
        """
        logging.info("WhatsApp/get_dataframe()")
        # FORMATION OF NEW DF for Analysis
        raw_df['media'] = raw_df['message'].apply(
            lambda x: re.findall("omitted", x)).str.len()
        df = raw_df.assign(emojis=raw_df["message"].apply(self.extract_emojis))
        df['urlcount'] = df.message.apply(
            lambda x: re.findall(self.URLPATTERN, x)).str.len()
        df['urlcount'].groupby(by=df['name']).sum()
        media_messages_df = df[df['message'].str.contains("omitted")]
        messages_df = df.drop(media_messages_df.index)
        messages_df['letter_count'] = messages_df['message'].apply(
            lambda s: len(s))
        messages_df['word_count'] = messages_df['message'].apply(
            lambda s: len(s.split(' ')))
        messages_df["message_count"] = 1
        return messages_df

    def get_members(self, df):
        """
        Return unique member list
        """
        logging.info("WhatsApp/get_members()")
        author_list = [author for author in df["name"].unique()]
        return author_list

    def sorted_authors(self, df):
        """
        Return sorted member list base on number of messages
        """
        logging.info("WhatsApp/sorted_authors()")
        sorted_authors = df.groupby('name')['message'].count()\
            .sort_values(ascending=False).index
        return sorted_authors

    def statistics(self, raw_df, df) -> Dict:
        """
        Statistics for summuary results
        """
        logging.info("WhatsApp/statistics()")
        author_list = self.get_members(df)
        return {
            "media_message": len(
                raw_df[raw_df['message'] == "<Media omitted>"]),
            "total_deleted_messages": len(
                raw_df[raw_df['message'] == "This message was deleted"]),
            "your_deleted_message": len(
                raw_df[raw_df['message'] == "You deleted this message"]),
            "group_name": raw_df.iloc[0:1]['name'][0],
            "total_messages": df.shape[0],
            'total_members': len(author_list),
            'link_shared': df['urlcount'].sum()}

    def day_analysis(self, df):
        """
        Exploratory Data Analysis on Dataframe
        """
        logging.info("WhatsApp/day_analysis()")
        lst = df.name.unique()
        for i in range(len(lst)):
            # Filtering out messages of particular user
            req_df = df[df["name"] == lst[i]]
            # req_df will contain messages of only one particular user
            print(f'{lst[i]} ->  {req_df.shape[0]}')
        df.groupby('name')['message'].count()\
            .sort_values(ascending=False).index
        df['day'] = df.datetime.dt.weekday.map(self.weeks)
        # Rearranging the columns for better understanding
        df = df[[
            'datetime', 'day', 'name', 'message',
            'date', 'time', 'emojis', 'urlcount']]
        df['day'] = df['day'].astype('category')
        lst = df.day.unique()
        # Day wise Message list
        for i in range(len(lst)):
            # Filtering out messages of particular user
            # req_df will contain messages of only one particular user
            print(f'{lst[i]} -> {df[df["day"] == lst[i]].shape[0]}')
        return df

    def cloud_data(self, raw_df):
        """
        Word Cloud DataFrame Formation
        """
        logging.info("WhatsApp/cloud_data()")
        sep = '|'
        cloud_df = raw_df[raw_df["message"].str.contains(
            sep.join(self.ignore)) == False]
        modified_df = cloud_df.copy()
        modified_df.message = cloud_df.loc[:,'message'].apply(
            lambda s: s.lower())\
            .str.replace('\n|\t', '', regex=True)\
            .str.replace(' {2,}', ' ', regex=True)\
            .str.strip().replace(r'http\S+', '', regex=True)\
            .replace(r'www\S+', '', regex=True)
        # Remove EMoji's
        final_df = modified_df.astype(str).apply(
            lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii'))
        return final_df

    def plot_data(self, x_value, y_value, tick_label, x_label, y_label, title):
        """
        Graph Plot function
        """
        logging.info("WhatsApp/plot_data()")
        fig, ax = plt.subplots()
        # Save the chart so we can loop through the bars below.
        bars = ax.bar(
            x=np.arange(x_value),
            height=y_value,
            tick_label=tick_label,
            color="#686868"
        )
        # Axis formatting.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#686868')
        ax.tick_params(bottom=False, left=False)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#EEEEEE')
        ax.xaxis.grid(False)
        # Grab the color of the bars so we can make the
        # text the same color.
        # bar_color = bars[0].get_facecolor()
        # Add text annotations to the top of the bars.
        # Note, you'll have to adjust this slightly (the 0.3)
        # with different data.
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                round(bar.get_height(), 1),
                horizontalalignment='center',
                color='green',  # bar_color
                # weight='bold'
            )
        ax.set_xlabel(x_label, labelpad=15, color='#333333')
        ax.set_ylabel(y_label, labelpad=15, color='#333333')
        ax.set_title(title, pad=15, color='#333333')  # weight='bold')
        return fig

    def max_words_used(self, df):
        """
        Maximum words used in sentence in group chat
        """
        logging.info("WhatsApp/max_words_used()")
        # Counting number of letters in each message
        # df['letters'] = df['message'].apply(lambda s: len(s))
        # # Counting number of word's in each message
        # df['words'] = df['message'].apply(lambda s: len(s.split(' ')))
        # # np.sum(df['words'])
        max_words = df[['name', 'word_count']].groupby('name').sum()
        m_w = max_words.sort_values('word_count', ascending=False).head(10)
        self.plot_data(
            m_w.size, m_w.word_count, m_w.index,
            'Name of Group Member',
            'Number of Words in Group Chat',
            'Analysis of members who has used more words in his/her messages',
            )

    def most_active_member(self, df):
        """
        Most active memeber as per number of messages in group
        """
        logging.info("WhatsApp/most_active_member()")
        # Mostly Active Author in the Group
        mostly_active = df['name'].value_counts()
        # Top 10 peoples that are mostly active in our Group
        m_a = mostly_active.head(10)
        self.plot_data(
            m_a.size, m_a, m_a.index,
            'Name of Group Member',
            'Number of Group Messages',
            'Mostly Active member in Group (based on messages)'
            )

    def most_active_day(self, df):
        """
        Most active day in Group as per messages numbers
        """
        logging.info("WhatsApp/most_active_day()")
        active_day = df['day'].value_counts()
        a_d = active_day.head(10)
        self.plot_data(
            a_d.size, a_d, a_d.index,
            'Name of Group Member',
            'Number of Group Messages',
            'Most active day of Week in the Group'
            )

    def top_media_contributor(self, df):
        """
        Top 10 members who shared media's in group
        """
        logging.info("WhatsApp/top_media_contributor()")
        # Top-10 Media Contributor of Group
        max_media = df[['name', 'media']].groupby('name').sum()
        mm = max_media.sort_values(
            'media', ascending=False).head(10)
        self.plot_data(
            mm.size, mm.media, mm.index,
            'Name of Group Member',
            'Number of Media Shared in Group',
            'Analysis of Top-10 Media shared in Group'
            )

    def who_shared_links(self, df):
        """
        Top 10 members Who shared maximum links in Group
        """
        logging.info("WhatsApp/who_shared_links()")
        # Member who has shared max numbers of link in Group
        max_words = df[['name', 'urlcount']].groupby('name').sum()
        m_w = max_words.sort_values('urlcount', ascending=False).head(10)
        self.plot_data(
            m_w.size, m_w.urlcount, m_w.index,
            'Name of Group Member',
            'Number of Links Shared in Group',
            'Analysis of members who has shared max no. of links in Group'
            )

    def time_series_plot(self, df):
        """
        Time analysis w.r.t to message in chat
        """
        logging.info("WhatsApp/time_series_plot()")
        z = df['datetime'].value_counts()
        z1 = z.to_dict()  # converts to dictionary
        df['Msg_count'] = df['datetime'].map(z1)
        # Timeseries plot
        fig = px.line(x=df['datetime'], y=df['Msg_count'])
        fig.update_layout(
            title='Analysis of number of messages using TimeSeries plot.',
            xaxis_title='Month',
            yaxis_title='No. of Messages')
        fig.update_xaxes(nticks=30)
        return fig

    def pie_display_emojis(self, df):
        """
        Pie chart formation for Emoji's Distrubution
        """
        logging.info("WhatsApp/pie_display_emojis()")
        total_emojis_list = list(set([a for b in df.emojis for a in b]))
        total_emojis_list = list([a for b in df.emojis for a in b])
        emoji_dict = dict(Counter(total_emojis_list))
        emoji_dict = sorted(
            emoji_dict.items(), key=lambda x: x[1], reverse=True)
        # for i in emoji_dict:
        #     print(i)
        emoji_df = pd.DataFrame(emoji_dict, columns=['emojis', 'count'])
        fig = px.pie(emoji_df, values='count', names='emojis')
        fig.update_traces(textposition='inside', textinfo='percent+label')
        return fig

    def time_when_group_active(self, df):
        """
        Most Messages Analsyis w.r.t to Time
        """
        logging.info("WhatsApp/time_when_group_active()")
        # Time whenever the group was highly active
        active_time = df.datetime.dt.time.value_counts().head(10)
        self.plot_data(
            active_time.size, active_time.values, active_time.index,
            'Time',
            'Number of Messages',
            'Analysis of time when group was highly active'
            )

    def most_suitable_hour(self, df):
        """
        Most Messages Analsyis w.r.t to Hour
        """
        logging.info("WhatsApp/most_suitable_hour()")
        # Time whenever the group was highly active
        active_hour = df.datetime.dt.hour.value_counts().head(20)
        self.plot_data(
            active_hour.size, active_hour.values, active_hour.index,
            'Hour',
            'Number of Messages',
            'Analysis of hour when group was highly active'
            )

    def most_suitable_day(self, df):
        """
        Most Messages Analsyis w.r.t to Day 
        """
        logging.info("WhatsApp/most_suitable_day()")
        # Time whenever the group was highly active
        active_day = df.datetime.dt.day.value_counts().head(20)
        self.plot_data(
            active_day.size, active_day.values, active_day.index,
            'Day',
            'Number of Messages',
            'Analysis of Day when group was highly active'
            )

    def sentiment_analysis(self, cloud_df):
        cloud_df['sentiment'] = cloud_df.message.apply(
            lambda text: TextBlob(text).sentiment.polarity)
        sentiment = cloud_df[['name', 'sentiment']].groupby('name').mean()
        s_a = sentiment.sort_values('sentiment', ascending=False).head(10)
        self.plot_data(
            s_a.size, s_a.sentiment, s_a.index,
            'Name of Group Member',
            'Positive Sentiment in Group',
            'Analysis of members having higher score in Positive Sentiment'
            )
