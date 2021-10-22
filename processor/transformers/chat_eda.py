"""Exploratory Data Analysis"""
import re
import logging
from typing import List, Dict, Any
import pandas as pd
import emoji
from pandas.errors import EmptyDataError


def get_members(data_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Return unique member list

    Attributes
    ----------
    Dataframe (pandas DF)

    Retrurns
    --------
    DataFrame Series
    """
    logging.info("WhatsApp/get_members()")
    return data_frame["name"].unique()


def sorted_authors_df(data_frame: pd.DataFrame) -> List:
    """
    Return sorted member list base on number of messages

    Attributes
    ----------
    Dataframe (pandas DF)

    Retrurns
    --------
    List : List of memebrs
    """
    logging.info("WhatsApp/sorted_authors()")
    sorted_authors = data_frame.groupby('name')['message'].count()\
        .sort_values(ascending=False).index
    return sorted_authors


def statistics(raw_df: pd.DataFrame, data_frame: pd.DataFrame) -> Dict:
    """
    Statistics for summuary results

    Attributes
    ----------
    Dataframe (pandas DF) : raw Dataframe
    Dataframe (pandas DF) : cleaned dataframe

    Retrurns
    --------
    Dict: Calculated features of members
    """
    logging.info("WhatsApp/statistics()")
    author_list = get_members(data_frame)
    return {
        "media_message": len(
            raw_df[raw_df.message.str.contains("omitted")]),
        "total_deleted_messages": len(
            raw_df[raw_df['message'] == "This message was deleted"]),
        "your_deleted_message": len(
            raw_df[raw_df['message'] == "You deleted this message"]),
        "group_name": raw_df.iloc[0:1]['name'][0],
        "total_messages": data_frame.shape[0],
        'total_members': len(author_list),
        'link_shared': data_frame['urlcount'].sum()}


def extract_emojis(string: str) -> str:
    """
    Extract emojis from message string

    Attributes
    ----------
    string (str): text with Emoji's content message

    Retrurns
    --------
    str: Emoji's extracted from message
    """
    return ''.join(c for c in string if c in emoji.UNICODE_EMOJI['en'])


def give_emoji_free_text(text: str) -> str:
    """
    Emojis free string

    Attributes
    ----------
    string (str): text with Emoji's content message

    Retrurns
    --------
    str: Emoji's extracted from message
    """
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text


def process_data(messages: str) -> Any:
    """
    Converting string messages into DataFrame

    Attributes
    ----------
    message (str): String of text

    Retrurns
    --------
    DataFrame (pandas DF)
    """
    logging.info("WhatsApp/process_data()")
    raw_df = pd.DataFrame(
        messages, columns=['datetime', 'name', 'message'])
    # SAMSUNG Export time format
    try:
        raw_df['datetime'] = raw_df['datetime'].str.replace(
            r'[p].[m].', 'PM', regex=True)
        raw_df['datetime'] = raw_df['datetime'].str.replace(
            r'[a].[m].', 'AM', regex=True)
        raw_df['datetime'] = pd.to_datetime(
            raw_df['datetime'], format="%Y-%m-%d, %I:%M %p")
    except Exception as diag:
        # IOS Export time format
        print(diag)
        try:
            # Drop date enclosures from date column
            raw_df['datetime'] = raw_df['datetime'].map(
                lambda x: x.lstrip('[').rstrip(']'))
            raw_df['datetime'] = pd.to_datetime(
                raw_df['datetime'], format="%d/%m/%y, %I:%M:%S %p")
        except Exception as diag:
            print(diag)
            # OppO Export time format
            try:
                raw_df['datetime'] = pd.to_datetime(
                    raw_df['datetime'], format="%d/%m/%Y, %I:%M %p")
            except Exception as diag:
                print(diag)
                # Android Export time format
                try:
                    raw_df['datetime'] = pd.to_datetime(
                        raw_df['datetime'], format="%d/%m/%y, %I:%M %p")
                except EmptyDataError as diag:
                    raise diag

    raw_df['date'] = pd.to_datetime(raw_df['datetime']).dt.date
    raw_df['time'] = pd.to_datetime(raw_df['datetime']).dt.time
    return raw_df


class WhatsAppProcess:
    """
    Exploratory Data Analysis for WhatsApp chat
    """
    def __init__(self):
        """Constructor object"""
        logging.info("WhatsApp Chat initiate")
        self.url_pattern = r'(https?://\S+)'
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
        # Messages which want to remove from Dataframe
        self.ignore = [
            'Missed video call', 'Missed group video call',
            'Missed voice call', '<Media omitted>',
            'This message was deleted', 'image omitted',
            'video omitted', 'You deleted this message',
            'sticker omitted']
        self.emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U0001F1F2-\U0001F1F4"  # Macau flag
            u"\U0001F1E6-\U0001F1FF"  # flags
            u"\U0001F600-\U0001F64F"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U0001F1F2"
            u"\U0001F1F4"
            u"\U0001F620"
            u"\u200d"
            u"\u2640-\u2642"
            "]+", flags=re.UNICODE)

    def apply_regex(self, data: str) -> List:
        """
        Attributes
        ----------
        data (str) : Input data string read from file

        Return
        ------
        list of data string
        """
        matches = []
        for reg in self.regex_list:
            matches += re.findall(reg, data)
        return matches

    def get_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Formation of Clean DataFrame

        Attributes
        ----------
        Dataframe (pandas DF)

        Retrurns
        --------
        Dataframe (pandas DF)
        """
        logging.info("WhatsApp/get_dataframe()")
        # FORMATION OF NEW DF for Analysis
        raw_df['media'] = raw_df['message'].apply(
            lambda x: re.findall("omitted", x)).str.len()
        data_frame = raw_df.assign(
            emojis=raw_df["message"].apply(extract_emojis))
        data_frame['urlcount'] = data_frame.message.apply(
            lambda x: re.findall(self.url_pattern, x)).str.len()
        data_frame['urlcount'].groupby(by=data_frame['name']).sum()
        media_messages_df = data_frame[
            data_frame['message'].str.contains("omitted")]
        messages_df = data_frame.drop(media_messages_df.index)
        messages_df['letter_count'] = messages_df['message'].apply(
            lambda s: len(s))
        messages_df['word_count'] = messages_df['message'].str.len()
        messages_df["message_count"] = 1
        return messages_df

    def day_analysis(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Exploratory Data Analysis on Dataframe

        Attributes
        ----------
        Dataframe (pandas DF)

        Retrurns
        --------
        Dataframe (pandas DF)
        """
        logging.info("WhatsApp/day_analysis()")
        # lst = data_frame.name.unique()
        # for i in range(len(lst)):
        #     # Filtering out messages of particular user
        #     req_df = data_frame[data_frame["name"] == lst[i]]
        #     # req_df will contain messages of only one particular user
        #     # print(f'{lst[i]} ->  {req_df.shape[0]}')
        data_frame.groupby('name')['message'].count()\
            .sort_values(ascending=False).index
        data_frame['day'] = data_frame.datetime.dt.weekday.map(self.weeks)
        # Rearranging the columns for better understanding
        data_frame = data_frame[[
            'datetime', 'day', 'name', 'message',
            'date', 'time', 'emojis', 'urlcount']]
        data_frame['day'] = data_frame['day'].astype('category')
        # lst = data_frame.day.unique()
        # Day wise Message list
        # for i in range(len(lst)):
        #     # Filtering out messages of particular user
        #     # req_df will contain messages of only one particular user
        #     # print(f'{lst[i]}->{data_frame[data_frame["day"] ==\
        #        lst[i]].shape[0]}')
        #     pass
        return data_frame

    def cloud_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Word Cloud DataFrame Formation

        Attributes
        ----------
        Dataframe (pandas DF)

        Retrurns
        --------
        Dataframe (pandas DF)
        """
        logging.info("WhatsApp/cloud_data()")
        sep = '|'
        cloud_df = raw_df[(raw_df["message"].str.contains(
            sep.join(self.ignore)) == False)]
        modified_df = cloud_df.copy()
        modified_df.message = cloud_df.loc[:, 'message'].apply(
            lambda s: s.lower())\
            .apply(lambda s: self.emoji_pattern.sub(r'', s))\
            .str.replace('\n|\t', '', regex=True)\
            .str.replace(' {2,}', ' ', regex=True)\
            .str.strip().replace(r'http\S+', '', regex=True)\
            .replace(r'www\S+', '', regex=True)
        return  modified_df
