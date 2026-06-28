"""Exploratory Data Analysis"""
import re
import logging
from typing import List, Dict, Any, NamedTuple
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
    return ''.join(item['emoji'] for item in emoji.emoji_list(string))


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
    return emoji.replace_emoji(text, replace='')


def process_data(messages: list) -> pd.DataFrame:
    """
    Converting parsed messages into DataFrame

    Attributes
    ----------
    messages (list): List of tuples (datetime, name, message)

    Retrurns
    --------
    DataFrame (pandas DF)
    """
    logging.info("WhatsApp/process_data()")
    raw_df = pd.DataFrame(
        messages, columns=['datetime', 'name', 'message'])
    
    if raw_df.empty:
        raise EmptyDataError("No WhatsApp chat data parsed. Check format.")

    # Clean brackets and normalize a.m./p.m./am/pm to AM/PM
    raw_df['datetime'] = raw_df['datetime'].astype(str)
    raw_df['datetime'] = raw_df['datetime'].str.replace('[', '', regex=False)
    raw_df['datetime'] = raw_df['datetime'].str.replace(']', '', regex=False)
    raw_df['datetime'] = raw_df['datetime'].str.replace(r'(?i)[a]\.?[m]\.?', 'AM', regex=True)
    raw_df['datetime'] = raw_df['datetime'].str.replace(r'(?i)[p]\.?[m]\.?', 'PM', regex=True)
    raw_df['datetime'] = raw_df['datetime'].str.strip()

    # Try parsing with format='mixed' first
    try:
        raw_df['datetime'] = pd.to_datetime(raw_df['datetime'], format='mixed')
    except Exception:
        # Fallback to try individual formats
        parsed = None
        formats = [
            "%d/%m/%y, %I:%M:%S %p",
            "%d/%m/%Y, %I:%M:%S %p",
            "%d/%m/%y, %H:%M:%S",
            "%d/%m/%Y, %H:%M:%S",
            "%d/%m/%y, %I:%M %p",
            "%d/%m/%Y, %I:%M %p",
            "%d/%m/%y, %H:%M",
            "%d/%m/%Y, %H:%M",
            "%m/%d/%y, %I:%M:%S %p",
            "%m/%d/%Y, %I:%M:%S %p",
            "%m/%d/%y, %I:%M %p",
            "%m/%d/%Y, %I:%M %p",
            "%Y-%m-%d, %H:%M:%S",
            "%Y-%m-%d, %H:%M",
            "%d.%m.%y, %H:%M:%S",
            "%d.%m.%Y, %H:%M:%S",
            "%d.%m.%y, %H:%M",
            "%d.%m.%Y, %H:%M",
        ]
        for fmt in formats:
            try:
                parsed = pd.to_datetime(raw_df['datetime'], format=fmt)
                break
            except Exception:
                continue
        if parsed is not None:
            raw_df['datetime'] = parsed
        else:
            raw_df['datetime'] = pd.to_datetime(raw_df['datetime'], errors='coerce')

    raw_df['date'] = pd.to_datetime(raw_df['datetime']).dt.date
    raw_df['time'] = pd.to_datetime(raw_df['datetime']).dt.time
    return raw_df


class WhatsAppConfig(NamedTuple):
    """
    class for Whatsapp Configuration
    
    url_pattern: https url pattern
    weeks: Week days dict
    regex_list: regex list for chat formatting
    ignore: Text to ignore in whatsapp caht
    """
    url_pattern: str
    weeks: dict
    regex_list: list
    ignore: list 


class WhatsAppProcess():
    """
    Read and Transform whatsapp messages to analytical format
    """
    def __init__(self, app_config):
        """
        Constructor for WhatsAppProcess
        
        :param app_config: NamedTuple class or dict with whatsapp configuratin data
        """
        self._logger = logging.getLogger(__name__)
        if isinstance(app_config, dict):
            self.app_config = WhatsAppConfig(**app_config)
        else:
            self.app_config = app_config
        self.ignore = self.app_config.ignore
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
            u"\U0001F926-\U0001F937"
            u"\U0001F1F2"
            u"\U0001F1F4"
            u"\U0001F620"
            u"\u200d"
            u"\u2640-\u2642"
            "]+", flags=re.UNICODE)

    def apply_regex(self, data: str) -> List:
        """
        Read the messages data and parse them into a list of tuples (datetime, name, message).
        This parser handles various date formats, iOS/Android styles, and multi-line messages.

        :returns:
            list: List of parsed messages as tuples
        """
        # Universal timestamp prefix pattern
        timestamp_pattern = re.compile(
            r'^(\[?'
            r'\d{1,4}[/\-\.]\d{1,4}[/\-\.]\d{1,4}'
            r'(?:,\s*|\s+)'
            r'\d{1,2}:\d{2}(?::\d{2})?'
            r'(?:\s*[aApP]\.?[mM]\.?)?'
            r'\]?'
            r')'
            r'(?:\s+-\s+|\s+)'
        )
        
        matches = []
        lines = data.split('\n')
        current_message = None
        
        for line in lines:
            stripped_line = line.rstrip('\r\n')
            
            # Check if this line starts a new message
            match = timestamp_pattern.match(stripped_line.strip())
            if match:
                # Save previous message
                if current_message:
                    matches.append((
                        current_message['datetime'],
                        current_message['name'],
                        current_message['message']
                    ))
                
                timestamp_str = match.group(1)
                # The text after timestamp and separator
                trimmed_line = stripped_line.strip()
                rest = trimmed_line[match.end():].strip()
                
                # Check for author separator: ": "
                if ': ' in rest:
                    author, msg = rest.split(': ', 1)
                    if len(author) <= 60:
                        current_message = {
                            'datetime': timestamp_str,
                            'name': author,
                            'message': msg
                        }
                    else:
                        current_message = None
                else:
                    current_message = None
            else:
                # Continuation line
                if current_message:
                    current_message['message'] += '\n' + stripped_line
                    
        # Append last message
        if current_message:
            matches.append((
                current_message['datetime'],
                current_message['name'],
                current_message['message']
            ))
            
        return matches

    def get_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Read the raw dataframe and trasform it to clean dataframe

        :param raw_df: Pandas Dataframe as input

        :returns:
            messages_df: Transformed Pandas DataFrame as Output
        """
        logging.info("WhatsApp/get_dataframe()")
        # FORMATION OF NEW DF for Analysis
        raw_df['media'] = raw_df['message'].apply(
            lambda x: re.findall("omitted", x)).str.len()
        data_frame = raw_df.assign(
            emojis=raw_df["message"].apply(extract_emojis))
        data_frame['urlcount'] = data_frame.message.apply(
            lambda x: re.findall(self.app_config.url_pattern, x)).str.len()
        data_frame['urlcount'].groupby(by=data_frame['name']).sum()
        media_messages_df = data_frame[
            data_frame['message'].str.contains("omitted")]
        messages_df = data_frame.drop(media_messages_df.index)
        messages_df['letter_count'] = messages_df['message'].apply(
            lambda s: len(s))
        messages_df['word_count'] = messages_df['message'].str.len()
        messages_df["message_count"] = 1
        self._logger.info("Extractig Raw Dataframe")
        return messages_df

    def day_analysis(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Exploratory Data Analysis on Dataframe

        :param data_frame: Pandas Dataframe as input

        :returns:
            data_frame: Transformed Pandas DataFrame as Output
        """
        # lst = data_frame.name.unique()
        # for i in range(len(lst)):
        #     # Filtering out messages of particular user
        #     req_df = data_frame[data_frame["name"] == lst[i]]
        #     # req_df will contain messages of only one particular user
        #     # print(f'{lst[i]} ->  {req_df.shape[0]}')
        data_frame.groupby('name')['message'].count()\
            .sort_values(ascending=False).index
        data_frame['day'] = data_frame.datetime.dt.weekday.map(self.app_config.weeks).astype('category')
        # Rearranging the columns for better understanding
        data_frame = data_frame[[
            'datetime', 'day', 'name', 'message',
            'date', 'time', 'emojis', 'urlcount']].copy()
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

        :param raw_df: Pandas Dataframe as input

        :returns:
            modified_df: Transformed Pandas DataFrame as Output
        """
        sep = '|'
        cloud_df = raw_df[(raw_df["message"].str.contains(
            sep.join(self.app_config.ignore)) == False)]
        modified_df = cloud_df.copy()
        modified_df.message = cloud_df.loc[:, 'message'].apply(
            lambda s: s.lower())\
            .apply(lambda s: emoji.replace_emoji(s, replace=''))\
            .str.replace('\n|\t', '', regex=True)\
            .str.replace(' {2,}', ' ', regex=True)\
            .str.strip().replace(r'http\S+', '', regex=True)\
            .replace(r'www\S+', '', regex=True)
        self._logger.info("Extracting information for Cloud Words")
        return modified_df
