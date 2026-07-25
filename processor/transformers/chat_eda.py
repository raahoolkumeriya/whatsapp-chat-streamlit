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
    total_emojis = sum(len(em) for em in data_frame['emojis']) if 'emojis' in data_frame.columns else 0

    # Calculate active days & average messages per day
    unique_dates = data_frame['datetime'].dt.date.nunique() if ('datetime' in data_frame.columns and not data_frame.empty) else 1
    avg_msgs_per_day = round(data_frame.shape[0] / max(unique_dates, 1), 1)

    # Calculate peak day of week
    peak_day = "N/A"
    if 'day' in data_frame.columns and not data_frame['day'].dropna().empty:
        peak_day = str(data_frame['day'].value_counts().index[0])

    # Calculate busiest hour
    busiest_hour = "N/A"
    if 'datetime' in data_frame.columns and not data_frame.empty:
        top_h = data_frame['datetime'].dt.hour.value_counts().index[0]
        ampm = "AM" if top_h < 12 else "PM"
        display_h = top_h if top_h <= 12 else top_h - 12
        if display_h == 0:
            display_h = 12
        busiest_hour = f"{display_h} {ampm}"

    stats = {
        "media_message": media_count,
        "total_deleted_messages": deleted_count,
        "your_deleted_message": your_deleted_count,
        "group_name": group_name,
        "total_messages": data_frame.shape[0],
        'total_members': len(author_list),
        'link_shared': link_count,
        'total_emojis': total_emojis,
        'avg_msgs_per_day': avg_msgs_per_day,
        'peak_day': peak_day,
        'busiest_hour': busiest_hour
    }
    logger.info("Summary stats calculated: Group='%s', Messages=%d, Members=%d, Media=%d, Links=%d, PeakDay=%s, PeakHour=%s",
                group_name, data_frame.shape[0], len(author_list), media_count, link_count, peak_day, busiest_hour)
    return stats


def process_data(messages: list) -> pd.DataFrame:
    """Convert parsed messages list of tuples into DataFrame."""
    start_time = time.perf_counter()
    logger.info("Converting %d parsed message tuples into DataFrame", len(messages))
    raw_df = pd.DataFrame(messages, columns=['datetime', 'name', 'message'])
    
    if raw_df.empty:
        logger.error("Parsed message list is empty. Check chat file format or regex rules.")
        raise EmptyDataError("No WhatsApp chat data parsed. Check format.")

    # Clean brackets, translate Devanagari digits (०-९ -> 0-9), and normalize AM/PM indicators
    raw_df['datetime'] = raw_df['datetime'].astype(str)
    devanagari_trans = str.maketrans('०१२३४५६७८९', '0123456789')
    raw_df['datetime'] = raw_df['datetime'].str.translate(devanagari_trans)
    raw_df['datetime'] = raw_df['datetime'].str.replace('[', '', regex=False)
    raw_df['datetime'] = raw_df['datetime'].str.replace(']', '', regex=False)
    raw_df['datetime'] = raw_df['datetime'].str.replace('पूर्वाह्न', 'AM', regex=False)
    raw_df['datetime'] = raw_df['datetime'].str.replace('अपराह्न', 'PM', regex=False)
    raw_df['datetime'] = raw_df['datetime'].str.replace('प.पू.', 'AM', regex=False)
    raw_df['datetime'] = raw_df['datetime'].str.replace('प.उ.', 'PM', regex=False)
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
            r'[\d\u0966-\u096F]{1,4}[/\-\.][\d\u0966-\u096F]{1,4}[/\-\.][\d\u0966-\u096F]{1,4}'
            r'(?:,\s*|\s+)'
            r'[\d\u0966-\u096F]{1,2}:[\d\u0966-\u096F]{2}(?::[\d\u0966-\u096F]{2})?'
            r'(?:\s*(?:[aApP]\.?[mM]\.?|पूर्वाह्न|अपराह्न|प\.पू\.|प\.उ\.))?'
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
        """Clean and extract message data for WordCloud analysis supporting Hindi & Marathi."""
        start_time = time.perf_counter()
        sep = '|'
        ignore_pattern = sep.join(re.escape(w) for w in self.app_config.ignore)
        cloud_df = raw_df[~raw_df["message"].str.contains(ignore_pattern, regex=True)].copy()
        
        ignored_count = len(raw_df) - len(cloud_df)
        cloud_df['message'] = (
            cloud_df['message']
            .apply(lambda s: str(s).lower())
            .apply(give_emoji_free_text)
            .str.replace(r'http\S+', '', regex=True)
            .str.replace(r'www\S+', '', regex=True)
            .str.replace(r'[\n\t]', ' ', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )
        elapsed = time.perf_counter() - start_time
        self._logger.info("Extracted WordCloud dataset in %.3f sec (System messages filtered out: %d, Final records: %d)",
                          elapsed, ignored_count, len(cloud_df))
        return cloud_df

    def extract_top_domains(self, data_frame: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Extract top shared web domain names from message URLs."""
        domain_pattern = re.compile(r'https?://(?:www\.)?([^/\s:]+)', re.IGNORECASE)
        domains = []
        for msg in data_frame['message'].dropna():
            found = domain_pattern.findall(str(msg))
            domains.extend(d.lower() for d in found)
        if not domains:
            return pd.DataFrame(columns=['domain', 'count'])
        counts = pd.Series(domains).value_counts().head(top_n).reset_index()
        counts.columns = ['domain', 'count']
        return counts

    def calculate_reply_latency(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Calculate average response speed (reply latency in minutes) per participant."""
        if 'datetime' not in data_frame.columns or 'name' not in data_frame.columns or data_frame.empty:
            return pd.DataFrame(columns=['name', 'avg_reply_time_mins'])
        
        df = data_frame.sort_values('datetime').copy()
        df['prev_name'] = df['name'].shift(1)
        df['prev_time'] = df['datetime'].shift(1)
        
        # Filter for messages that are replies to a different author
        replies = df[(df['name'] != df['prev_name']) & df['prev_name'].notna()].copy()
        replies['reply_time_mins'] = (replies['datetime'] - replies['prev_time']).dt.total_seconds() / 60.0
        
        # Keep realistic active conversation replies (under 2 hours / 120 mins)
        active_replies = replies[replies['reply_time_mins'] <= 120]
        if active_replies.empty:
            return pd.DataFrame(columns=['name', 'avg_reply_time_mins'])
            
        latency = active_replies.groupby('name')['reply_time_mins'].mean().round(1).reset_index()
        latency.columns = ['name', 'avg_reply_time_mins']
        return latency.sort_values('avg_reply_time_mins')

    def hourly_activity_profile(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Extract message count distribution across all 24 hours (0-23)."""
        if 'datetime' not in data_frame.columns or data_frame.empty:
            return pd.DataFrame({'hour': range(24), 'message_count': [0]*24})
        
        hour_counts = data_frame['datetime'].dt.hour.value_counts()
        profile = pd.DataFrame({'hour': range(24)})
        profile['message_count'] = profile['hour'].map(hour_counts).fillna(0).astype(int)
        profile['hour_label'] = profile['hour'].apply(
            lambda h: f"{h if h <= 12 else h-12} {'AM' if h < 12 else 'PM'}" if h != 0 else "12 AM"
        )
        return profile

    def analyze_message_sentiment(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Calculate sentiment polarity distribution across messages."""
        try:
            from textblob import TextBlob
        except ImportError:
            TextBlob = None

        categories = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
        for msg in data_frame['message'].dropna():
            text = str(msg).strip()
            if not text:
                categories['Neutral'] += 1
                continue
            if TextBlob:
                polarity = TextBlob(text).sentiment.polarity
            else:
                polarity = 0.0
            
            if polarity > 0.05:
                categories['Positive'] += 1
            elif polarity < -0.05:
                categories['Negative'] += 1
            else:
                categories['Neutral'] += 1

        res = pd.DataFrame(list(categories.items()), columns=['Sentiment', 'Count'])
        return res

    def message_length_distribution(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Categorize messages into word-length buckets."""
        if 'word_count' not in data_frame.columns or data_frame.empty:
            return pd.DataFrame(columns=['Bucket', 'Count'])
            
        def bucketize(cnt):
            if cnt <= 3:
                return '1-3 Words (Short)'
            elif cnt <= 10:
                return '4-10 Words (Medium)'
            elif cnt <= 25:
                return '11-25 Words (Long)'
            else:
                return '25+ Words (Detailed)'

        buckets = data_frame['word_count'].apply(bucketize).value_counts().reset_index()
        buckets.columns = ['Bucket', 'Count']
        return buckets


WhatsAppAnalyzer = WhatsAppProcess
