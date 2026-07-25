"""Exploratory Data Analysis for WhatsApp Chat Logs."""
import re
import time
import logging
from typing import List, Dict, Any, NamedTuple
import pandas as pd
import emoji
from pandas.errors import EmptyDataError

logger = logging.getLogger(__name__)


def extract_emojis(string: str) -> str:
    """Extract emojis from message string."""
    return ''.join(item['emoji'] for item in emoji.emoji_list(string))


def give_emoji_free_text(text: str) -> str:
    """Return text stripped of emojis."""
    return emoji.replace_emoji(text, replace='')


def get_members(data_frame: pd.DataFrame) -> pd.Series:
    """Return unique member list from DataFrame."""
    logger.info("Extracting unique members list from DataFrame (Total unique: %d)", data_frame["name"].nunique() if 'name' in data_frame.columns else 0)
    return data_frame["name"].unique()


def sorted_authors_df(data_frame: pd.DataFrame):
    """Return member list sorted by message count descending."""
    logger.info("Computing sorted author message counts")
    return data_frame.groupby('name')['message'].count()\
        .sort_values(ascending=False).index


def statistics(raw_df: pd.DataFrame, data_frame: pd.DataFrame) -> Dict[str, Any]:
    """Calculate summary statistics for chat dataset."""
    logger.info("Calculating summary statistics across raw (%d rows) and cleaned (%d rows) DataFrames", len(raw_df), len(data_frame))
    author_list = get_members(data_frame)
    media_count = len(raw_df[raw_df.message.str.contains("omitted")])
    deleted_count = len(raw_df[raw_df['message'] == "This message was deleted"])
    your_deleted_count = len(raw_df[raw_df['message'] == "You deleted this message"])
    group_name = raw_df.iloc[0:1]['name'][0] if not raw_df.empty else "Group"
    link_count = data_frame['urlcount'].sum() if 'urlcount' in data_frame.columns else 0

    stats = {
        "media_message": media_count,
        "total_deleted_messages": deleted_count,
        "your_deleted_message": your_deleted_count,
        "group_name": group_name,
        "total_messages": data_frame.shape[0],
        'total_members': len(author_list),
        'link_shared': link_count
    }
    logger.info("Summary stats calculated: Group='%s', Messages=%d, Members=%d, Media=%d, Links=%d",
                group_name, data_frame.shape[0], len(author_list), media_count, link_count)
    return stats


def process_data(messages: list) -> pd.DataFrame:
    """Convert parsed messages list of tuples into DataFrame."""
    start_time = time.perf_counter()
    logger.info("Converting %d parsed message tuples into DataFrame", len(messages))
    raw_df = pd.DataFrame(messages, columns=['datetime', 'name', 'message'])
    
    if raw_df.empty:
        logger.error("Parsed message list is empty. Check chat file format or regex rules.")
        raise EmptyDataError("No WhatsApp chat data parsed. Check format.")

    # Clean brackets and normalize am/pm indicators
    raw_df['datetime'] = raw_df['datetime'].astype(str)
    raw_df['datetime'] = raw_df['datetime'].str.replace('[', '', regex=False)
    raw_df['datetime'] = raw_df['datetime'].str.replace(']', '', regex=False)
    raw_df['datetime'] = raw_df['datetime'].str.replace(r'(?i)[a]\.?[m]\.?', 'AM', regex=True)
    raw_df['datetime'] = raw_df['datetime'].str.replace(r'(?i)[p]\.?[m]\.?', 'PM', regex=True)
    raw_df['datetime'] = raw_df['datetime'].str.strip()

    try:
        raw_df['datetime'] = pd.to_datetime(raw_df['datetime'], format='mixed')
        logger.debug("Successfully parsed datetimes using format='mixed'")
    except Exception as parse_err:
        logger.warning("pd.to_datetime format='mixed' failed: %s. Attempting fallback format list.", parse_err)
        parsed = None
        formats = [
            "%d/%m/%y, %I:%M:%S %p", "%d/%m/%Y, %I:%M:%S %p",
            "%d/%m/%y, %H:%M:%S", "%d/%m/%Y, %H:%M:%S",
            "%d/%m/%y, %I:%M %p", "%d/%m/%Y, %I:%M %p",
            "%d/%m/%y, %H:%M", "%d/%m/%Y, %H:%M",
            "%m/%d/%y, %I:%M:%S %p", "%m/%d/%Y, %I:%M:%S %p",
            "%m/%d/%y, %I:%M %p", "%m/%d/%Y, %I:%M %p",
            "%Y-%m-%d, %H:%M:%S", "%Y-%m-%d, %H:%M",
            "%d.%m.%y, %H:%M:%S", "%d.%m.%Y, %H:%M:%S",
            "%d.%m.%y, %H:%M", "%d.%m.%Y, %H:%M",
        ]
        for fmt in formats:
            try:
                parsed = pd.to_datetime(raw_df['datetime'], format=fmt)
                logger.debug("Successfully parsed datetimes with format '%s'", fmt)
                break
            except Exception:
                continue
        if parsed is not None:
            raw_df['datetime'] = parsed
        else:
            logger.warning("Fallback formats failed. Coercing invalid datetimes to NaT.")
            raw_df['datetime'] = pd.to_datetime(raw_df['datetime'], errors='coerce')

    raw_df['date'] = pd.to_datetime(raw_df['datetime']).dt.date
    raw_df['time'] = pd.to_datetime(raw_df['datetime']).dt.time
    
    elapsed = time.perf_counter() - start_time
    logger.info("DataFrame processing complete in %.3f sec (Shape: %s, Datetime range: %s to %s)",
                elapsed, raw_df.shape, raw_df['date'].min(), raw_df['date'].max())
    return raw_df


class WhatsAppConfig(NamedTuple):
    """Configuration container for WhatsApp parsing parameters."""
    url_pattern: str
    weeks: dict
    regex_list: list
    ignore: list


class WhatsAppProcess:
    """
    Object-Oriented Processor for reading, cleaning, and transforming
    WhatsApp chat logs into analytical structures.
    """
    def __init__(self, app_config: Any):
        self._logger = logging.getLogger(__name__)
        if isinstance(app_config, dict):
            self.app_config = WhatsAppConfig(**app_config)
        else:
            self.app_config = app_config
        self.ignore = self.app_config.ignore
        self.url_pattern = re.compile(self.app_config.url_pattern)
        self._logger.debug("WhatsAppProcess initialized with %d ignore rules", len(self.ignore))

    def apply_regex(self, data: str) -> List[tuple]:
        """
        Parse raw chat log text into list of tuples (datetime, author, message).
        Supports multi-line messages and multiple device export formats.
        """
        start_time = time.perf_counter()
        lines = data.split('\n')
        self._logger.info("Starting regex chat log parsing across %d lines (Raw size: %d bytes)", len(lines), len(data))

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
        current_message = None
        multi_line_count = 0
        
        for line in lines:
            stripped_line = line.rstrip('\r\n')
            match = timestamp_pattern.match(stripped_line.strip())
            
            if match:
                if current_message:
                    matches.append((
                        current_message['datetime'],
                        current_message['name'],
                        current_message['message']
                    ))
                
                timestamp_str = match.group(1)
                trimmed_line = stripped_line.strip()
                rest = trimmed_line[match.end():].strip()
                
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
                if current_message:
                    current_message['message'] += '\n' + stripped_line
                    multi_line_count += 1
                    
        if current_message:
            matches.append((
                current_message['datetime'],
                current_message['name'],
                current_message['message']
            ))
            
        elapsed = time.perf_counter() - start_time
        self._logger.info("Regex parsing complete in %.3f sec: Extracted %d messages (%d multi-line continuations)",
                          elapsed, len(matches), multi_line_count)
        return matches

    def get_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Transform raw message DataFrame into enriched analytical DataFrame."""
        start_time = time.perf_counter()
        self._logger.info("Transforming raw DataFrame (%d rows) into enriched analytics DataFrame", len(raw_df))
        
        raw_df['media'] = raw_df['message'].apply(
            lambda x: len(re.findall("omitted", str(x))))
        data_frame = raw_df.copy()
        data_frame['emojis'] = data_frame["message"].apply(extract_emojis)
        data_frame['urlcount'] = data_frame.message.apply(
            lambda x: len(self.url_pattern.findall(str(x))))

        media_messages_df = data_frame[data_frame['message'].str.contains("omitted")]
        messages_df = data_frame.drop(media_messages_df.index).copy()
        
        messages_df['letter_count'] = messages_df['message'].apply(len)
        messages_df['word_count'] = messages_df['message'].str.len()
        messages_df["message_count"] = 1

        elapsed = time.perf_counter() - start_time
        total_urls = messages_df['urlcount'].sum()
        total_words = messages_df['word_count'].sum()
        self._logger.info("Analytics DataFrame transformed in %.3f sec (Media messages dropped: %d, Standard messages: %d, Total URLs: %d, Total words: %d)",
                          elapsed, len(media_messages_df), len(messages_df), total_urls, total_words)
        return messages_df

    def day_analysis(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Perform day-of-week feature mapping on DataFrame."""
        self._logger.info("Performing day-of-week feature mapping on %d records", len(data_frame))
        data_frame['day'] = data_frame.datetime.dt.weekday.map(self.app_config.weeks).astype('category')
        return data_frame

    def cloud_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and extract message data for WordCloud analysis."""
        start_time = time.perf_counter()
        sep = '|'
        ignore_pattern = sep.join(re.escape(w) for w in self.app_config.ignore)
        cloud_df = raw_df[~raw_df["message"].str.contains(ignore_pattern, regex=True)].copy()
        
        ignored_count = len(raw_df) - len(cloud_df)
        cloud_df['message'] = (
            cloud_df['message']
            .apply(lambda s: s.lower())
            .apply(give_emoji_free_text)
            .str.replace('\n|\t', '', regex=True)
            .str.replace(' {2,}', ' ', regex=True)
            .str.strip()
            .replace(r'http\S+', '', regex=True)
            .replace(r'www\S+', '', regex=True)
        )
        elapsed = time.perf_counter() - start_time
        self._logger.info("Extracted WordCloud dataset in %.3f sec (System messages filtered out: %d, Final records: %d)",
                          elapsed, ignored_count, len(cloud_df))
        return cloud_df


WhatsAppAnalyzer = WhatsAppProcess
