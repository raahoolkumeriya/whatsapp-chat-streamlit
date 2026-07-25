"""
Analytics and Visualization Chart Generator Module for WhatsApp Chat Analysis.
Provides an Object-Oriented ChatVisualizer class and DRY helper routines.
"""
import os
import time
import logging
from typing import Dict, Any, Union
from collections import Counter
import numpy as np
import pandas as pd
import plotly.express as px
from textblob import TextBlob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

logger = logging.getLogger(__name__)

FONT_PATH = 'Laila-Regular.ttf'
DEVANAGARI_FONT = fm.FontProperties(fname=FONT_PATH) if os.path.exists(FONT_PATH) else None


class ChatVisualizer:
    """
    Object-Oriented visualization builder for WhatsApp chat metrics using
    Matplotlib and Plotly engines.
    """
    def __init__(self, bar_color: str = "#4361ee", text_color: str = "#2b2d42"):
        self.bar_color = bar_color
        self.text_color = text_color
        self._logger = logging.getLogger(__name__)

    def build_bar_chart(
        self,
        x_count: int,
        y_values: Any,
        tick_labels: Any,
        x_label: str,
        y_label: str,
        title: str
    ) -> plt.Figure:
        """DRY helper method to construct standardized, styled Matplotlib bar charts."""
        start_time = time.perf_counter()
        fig, ax = plt.subplots(figsize=(10, 5))
        y_values_clean = np.asarray(y_values)
        if y_values_clean.dtype == bool or y_values_clean.dtype == np.bool_:
            y_values_clean = y_values_clean.astype(np.float64)
        else:
            y_values_clean = np.asarray(y_values_clean, dtype=np.float64)
        tick_labels_clean = [str(lbl) for lbl in tick_labels]
        bars = ax.bar(
            x=np.arange(len(y_values_clean)),
            height=y_values_clean,
            tick_label=tick_labels_clean,
            color=self.bar_color,
            alpha=0.85,
            edgecolor="none"
        )
        # Modern axis styling
        for spine in ['top', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#cccccc')
        ax.tick_params(bottom=False, left=False)
        ax.tick_params(axis='x', labelrotation=45, labelsize=9)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color='#e9ecef', linestyle='--')
        ax.xaxis.grid(False)

        # Bar top value labels
        max_y = float(np.max(y_values_clean)) if len(y_values_clean) > 0 else 1.0
        for bar in bars:
            height = float(bar.get_height())
            if not np.isnan(height) and height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + (max_y * 0.01),
                    f"{round(height, 1):g}",
                    ha='center',
                    va='bottom',
                    color='#2b2d42',
                    fontsize=9,
                    fontweight='bold'
                )

        if DEVANAGARI_FONT:
            ax.set_xlabel(x_label, labelpad=10, color=self.text_color, fontproperties=DEVANAGARI_FONT, fontsize=10)
            ax.set_ylabel(y_label, labelpad=10, color=self.text_color, fontproperties=DEVANAGARI_FONT, fontsize=10)
            ax.set_title(title, pad=15, color=self.text_color, fontproperties=DEVANAGARI_FONT, fontsize=12)
            for tick in ax.get_xticklabels():
                tick.set_fontproperties(DEVANAGARI_FONT)
                tick.set_fontsize(9)
        else:
            ax.set_xlabel(x_label, labelpad=10, color=self.text_color, fontweight='medium')
            ax.set_ylabel(y_label, labelpad=10, color=self.text_color, fontweight='medium')
            ax.set_title(title, pad=15, color=self.text_color, fontsize=12, fontweight='bold')
        fig.tight_layout()
        
        elapsed = time.perf_counter() - start_time
        self._logger.debug("Rendered Matplotlib bar chart '%s' (Bars: %d, Max Value: %.1f) in %.3f sec",
                           title, len(y_values_clean), max_y, elapsed)
        return fig

    def message_cluster(self, data_frame: pd.DataFrame):
        """Display scatter plot cluster of message stats by member."""
        import re
        start_time = time.perf_counter()
        df_temp = data_frame.copy()
        if 'media' not in df_temp.columns:
            df_temp['media'] = df_temp['message'].apply(lambda x: len(re.findall("omitted", str(x))))
        if 'emojis' not in df_temp.columns:
            from processor.transformers.chat_eda import extract_emojis
            df_temp['emojis'] = df_temp['message'].apply(extract_emojis)
        if 'urlcount' not in df_temp.columns:
            df_temp['urlcount'] = df_temp['message'].apply(lambda x: len(re.findall(r'(https?://\S+)', str(x))))
        if 'letter_count' not in df_temp.columns:
            df_temp['letter_count'] = df_temp['message'].apply(lambda s: len(str(s)))
        if 'word_count' not in df_temp.columns:
            df_temp['word_count'] = df_temp['message'].astype(str).str.len()

        grouped = df_temp.groupby('name')
        new_df = pd.DataFrame(grouped['message'].count())
        new_df['media_count'] = grouped['media'].sum()
        new_df['emoji_count'] = grouped['emojis'].apply(lambda x: sum(len(e) for e in x))
        new_df['urlcount_count'] = grouped['urlcount'].sum()
        new_df['letter_count'] = grouped['letter_count'].sum()
        new_df['words_count'] = grouped['word_count'].sum()
        new_df.reset_index(level=0, inplace=True)

        fig = px.scatter(
            new_df,
            x="message",
            y="words_count",
            size="letter_count",
            color="name",
            hover_name="name",
            hover_data=["media_count", "urlcount_count"],
            log_x=True,
            size_max=50,
            title="Member Activity Cluster (Messages vs Words Used)"
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        elapsed = time.perf_counter() - start_time
        self._logger.info("Rendered Member Activity Cluster scatter plot across %d members in %.3f sec", len(new_df), elapsed)
        return fig

    def pie_display_emojis(self, data_frame: pd.DataFrame):
        """Pie chart distribution for Emojis in conversation."""
        start_time = time.perf_counter()
        total_emojis_list = [a for b in data_frame.emojis for a in b] if 'emojis' in data_frame.columns else []
        emoji_dict = dict(Counter(total_emojis_list))
        emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
        
        emoji_df = pd.DataFrame(emoji_dict[:15], columns=['emojis', 'count'])
        fig = px.pie(
            emoji_df,
            values='count',
            names='emojis',
            title="Top 15 Emojis Used in Chat",
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        elapsed = time.perf_counter() - start_time
        top_emoji = emoji_dict[0][0] if emoji_dict else "None"
        self._logger.info("Rendered Emoji Distribution pie chart (Total unique emojis: %d, Top emoji: %s) in %.3f sec",
                          len(emoji_dict), top_emoji, elapsed)
        return fig

    def time_series_plot(self, data_frame: pd.DataFrame):
        """Time series analysis plot of messages over time."""
        start_time = time.perf_counter()
        df_copy = data_frame.copy()
        msg_counts = df_copy.groupby('date')['message'].count().reset_index()
        msg_counts.columns = ['date', 'msg_count']

        fig = px.line(
            msg_counts,
            x='date',
            y='msg_count',
            title='Activity Timeline (Messages per Day)',
            markers=True
        )
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Number of Messages',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        elapsed = time.perf_counter() - start_time
        max_row = msg_counts.loc[msg_counts['msg_count'].idxmax()] if not msg_counts.empty else None
        peak_date = max_row['date'] if max_row is not None else "N/A"
        peak_count = max_row['msg_count'] if max_row is not None else 0
        self._logger.info("Rendered Time Series timeline across %d dates (Peak date: %s with %d msgs) in %.3f sec",
                          len(msg_counts), peak_date, peak_count, elapsed)
        return fig


# Global default visualizer instance
default_visualizer = ChatVisualizer()


def plot_data(data_string: Dict[str, Any]) -> plt.Figure:
    """Wrapper function for custom data_string bar plots."""
    logger.debug("plot_data wrapper called for title '%s'", data_string.get('title', ''))
    return default_visualizer.build_bar_chart(
        x_count=data_string.get('x_value', 0),
        y_values=data_string.get('y_value', []),
        tick_labels=data_string.get('tick_label', []),
        x_label=data_string.get('x_label', ''),
        y_label=data_string.get('y_label', ''),
        title=data_string.get('title', '')
    )


def message_cluster(data_frame: pd.DataFrame):
    """Display Message Cluster plot."""
    return default_visualizer.message_cluster(data_frame)


def pie_display_emojis(data_frame: pd.DataFrame):
    """Display Emoji distribution pie chart."""
    return default_visualizer.pie_display_emojis(data_frame)


def time_series_plot(data_frame: pd.DataFrame):
    """Display activity time series line chart."""
    return default_visualizer.time_series_plot(data_frame)


def max_words_used(data_frame: pd.DataFrame) -> plt.Figure:
    """Plot members with maximum words used."""
    logger.info("Calculating top word count contributors")
    max_words = data_frame[['name', 'word_count']].groupby('name').sum()
    m_w = max_words.sort_values('word_count', ascending=False).head(10)
    top_speaker = m_w.index[0] if not m_w.empty else "N/A"
    top_words = m_w.iloc[0, 0] if not m_w.empty else 0
    logger.info("Top word contributor: '%s' with %d words", top_speaker, top_words)
    return plot_data({
        'x_value': m_w.shape[0],
        'y_value': m_w.word_count.values,
        'tick_label': m_w.index,
        'x_label': 'Group Member',
        'y_label': 'Total Words',
        'title': 'Top Members by Total Words Spoken'
    })


def most_active_member(data_frame: pd.DataFrame) -> plt.Figure:
    """Plot mostly active member by message count."""
    logger.info("Calculating most active group members by message count")
    mostly_active = data_frame['name'].value_counts().head(10)
    top_active = mostly_active.index[0] if not mostly_active.empty else "N/A"
    top_count = mostly_active.iloc[0] if not mostly_active.empty else 0
    logger.info("Most active member: '%s' with %d messages", top_active, top_count)
    return plot_data({
        'x_value': mostly_active.shape[0],
        'y_value': mostly_active.values,
        'tick_label': mostly_active.index,
        'x_label': 'Group Member',
        'y_label': 'Message Count',
        'title': 'Most Active Group Members (by Message Count)'
    })


def most_active_day(data_frame: pd.DataFrame) -> plt.Figure:
    """Plot most active day of week."""
    logger.info("Calculating active days of the week")
    if 'day' in data_frame.columns:
        active_day = data_frame['day'].value_counts()
    else:
        active_day = data_frame.datetime.dt.day_name().value_counts()
    a_d = active_day.head(7)
    peak_day = a_d.index[0] if not a_d.empty else "N/A"
    logger.info("Peak active weekday: '%s'", peak_day)
    return plot_data({
        'x_value': a_d.shape[0],
        'y_value': a_d.values,
        'tick_label': a_d.index,
        'x_label': 'Day of Week',
        'y_label': 'Message Count',
        'title': 'Most Active Days of the Week'
    })


def top_media_contributor(data_frame: pd.DataFrame) -> plt.Figure:
    """Plot top media contributors."""
    import re
    logger.info("Calculating top media contributors")
    df_temp = data_frame.copy()
    if 'media' not in df_temp.columns:
        df_temp['media'] = df_temp['message'].apply(lambda x: len(re.findall("omitted", str(x))))
    max_media = df_temp[['name', 'media']].groupby('name').sum()
    m_m = max_media.sort_values('media', ascending=False).head(10)
    top_media_user = m_m.index[0] if not m_m.empty else "N/A"
    logger.info("Top media contributor: '%s'", top_media_user)
    return plot_data({
        'x_value': m_m.shape[0],
        'y_value': m_m.media.values,
        'tick_label': m_m.index,
        'x_label': 'Group Member',
        'y_label': 'Media Shared',
        'title': 'Top 10 Media Contributors (Photos/Videos/Files)'
    })


def who_shared_links(data_frame: pd.DataFrame) -> plt.Figure:
    """Plot members who shared maximum links."""
    logger.info("Calculating top web link sharing contributors")
    max_urls = data_frame[['name', 'urlcount']].groupby('name').sum()
    m_w = max_urls.sort_values('urlcount', ascending=False).head(10)
    top_link_user = m_w.index[0] if not m_w.empty else "N/A"
    logger.info("Top link sharing contributor: '%s'", top_link_user)
    return plot_data({
        'x_value': m_w.shape[0],
        'y_value': m_w.urlcount.values,
        'tick_label': m_w.index,
        'x_label': 'Group Member',
        'y_label': 'Links Shared',
        'title': 'Top Members Sharing Web Links'
    })


def time_when_group_active(data_frame: pd.DataFrame) -> plt.Figure:
    """Plot active time breakdown."""
    logger.info("Calculating active timestamp breakdown")
    active_time = data_frame.datetime.dt.time.value_counts().head(10)
    return plot_data({
        'x_value': active_time.shape[0],
        'y_value': active_time.values,
        'tick_label': [str(t) for t in active_time.index],
        'x_label': 'Time',
        'y_label': 'Message Count',
        'title': 'Peak Group Activity Timestamps'
    })


def most_suitable_hour(data_frame: pd.DataFrame) -> plt.Figure:
    """Plot activity breakdown by hour of day."""
    logger.info("Calculating hourly activity breakdown (0-23 hours)")
    active_hour = data_frame.datetime.dt.hour.value_counts().sort_index()
    return plot_data({
        'x_value': active_hour.shape[0],
        'y_value': active_hour.values,
        'tick_label': [f"{h:02d}:00" for h in active_hour.index],
        'x_label': 'Hour of Day (24-hour)',
        'y_label': 'Message Count',
        'title': 'Group Activity by Hour of Day'
    })


def most_suitable_day(data_frame: pd.DataFrame) -> plt.Figure:
    """Plot activity breakdown by day of month."""
    logger.info("Calculating day-of-month activity breakdown (1-31 days)")
    active_day = data_frame.datetime.dt.day.value_counts().sort_index().head(31)
    return plot_data({
        'x_value': active_day.shape[0],
        'y_value': active_day.values,
        'tick_label': [str(d) for d in active_day.index],
        'x_label': 'Day of Month',
        'y_label': 'Message Count',
        'title': 'Group Activity by Day of Month'
    })


def sentiment_analysis(cloud_df: pd.DataFrame) -> plt.Figure:
    """Plot positive sentiment analysis by member."""
    start_time = time.perf_counter()
    logger.info("Running TextBlob sentiment analysis across %d clean messages", len(cloud_df))
    df_temp = cloud_df.copy()
    df_temp['sentiment'] = df_temp['message'].apply(lambda text: TextBlob(str(text)).sentiment.polarity)
    sentiment = df_temp[['name', 'sentiment']].groupby('name').mean()
    s_a = sentiment.sort_values('sentiment', ascending=False).head(10)
    top_positive = s_a.index[0] if not s_a.empty else "N/A"
    top_score = s_a.iloc[0, 0] if not s_a.empty else 0.0
    elapsed = time.perf_counter() - start_time
    logger.info("Sentiment analysis complete in %.3f sec (Top positive member: '%s' with score %.3f)",
                elapsed, top_positive, top_score)
    return plot_data({
        'x_value': s_a.shape[0],
        'y_value': s_a.sentiment.values,
        'tick_label': s_a.index,
        'x_label': 'Group Member',
        'y_label': 'Average Polarity Score (0.0 to 1.0)',
        'title': 'Member Sentiment Polarity (Most Positive)'
    })


def plot_top_domains(domain_df: pd.DataFrame) -> plt.Figure:
    """Plot horizontal bar chart of top shared web domain names."""
    if domain_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No Web Links Found in Chat", ha='center', va='center', fontsize=12, color='#888888')
        ax.axis('off')
        return fig

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = np.arange(len(domain_df))
    ax.barh(y_pos, domain_df['count'], color='#06B6D4', alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(domain_df['domain'], fontsize=10)
    ax.invert_yaxis()  # top-down
    ax.set_xlabel('Links Shared Count', fontweight='bold', fontsize=10)
    ax.set_title('🌐 Top Shared Web Domains & Websites', fontweight='bold', fontsize=12, pad=12)
    
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.xaxis.grid(True, color='#e9ecef', linestyle='--')
    
    for i, v in enumerate(domain_df['count']):
        ax.text(v + 0.1, i, str(v), va='center', fontweight='bold', color='#333333')
        
    fig.tight_layout()
    return fig


def plot_reply_latency(latency_df: pd.DataFrame) -> plt.Figure:
    """Plot bar chart of average response speed (reply latency in minutes) per member."""
    if latency_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Insufficient Reply Data", ha='center', va='center', fontsize=12, color='#888888')
        ax.axis('off')
        return fig

    return plot_data({
        'x_value': latency_df.shape[0],
        'y_value': latency_df['avg_reply_time_mins'].values,
        'tick_label': latency_df['name'].values,
        'x_label': 'Group Participant',
        'y_label': 'Average Reply Speed (Minutes)',
        'title': '⏱️ Member Response Speed (Fastest Responders)'
    })


def plot_hourly_profile(profile_df: pd.DataFrame) -> plt.Figure:
    """Plot 24-hour activity distribution curve."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = profile_df['hour']
    y = profile_df['message_count']
    
    ax.plot(x, y, color='#6366F1', linewidth=2.5, marker='o', markersize=5, markerfacecolor='#818CF8')
    ax.fill_between(x, y, color='#6366F1', alpha=0.15)
    
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=9)
    ax.set_xlabel('Hour of Day (24-Hour)', fontweight='bold', fontsize=10)
    ax.set_ylabel('Message Count', fontweight='bold', fontsize=10)
    ax.set_title('⚡ 24-Hour Hourly Activity Profile (Night Owls vs Early Birds)', fontweight='bold', fontsize=12, pad=12)
    
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#cccccc')
    ax.yaxis.grid(True, color='#e9ecef', linestyle='--')
    fig.tight_layout()
    return fig


def plot_sentiment_distribution(sentiment_df: pd.DataFrame) -> plt.Figure:
    """Plot pie chart of group message sentiment polarity distribution."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ['#10B981', '#94A3B8', '#F43F5E']  # Green, Slate, Red
    counts = sentiment_df['Count'].values
    labels = sentiment_df['Sentiment'].values
    
    if sum(counts) == 0:
        counts = [1]
        labels = ['Neutral']

    wedges, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors[:len(counts)],
        textprops=dict(color="#333333", fontweight='bold'),
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
    )
    ax.set_title('🎭 Group Sentiment Distribution', fontweight='bold', fontsize=12, pad=12)
    fig.tight_layout()
    return fig


def plot_message_length_distribution(length_df: pd.DataFrame) -> plt.Figure:
    """Plot bar chart of message word-length categories."""
    if length_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No Message Length Data", ha='center', va='center', fontsize=12, color='#888888')
        ax.axis('off')
        return fig

    return plot_data({
        'x_value': length_df.shape[0],
        'y_value': length_df['Count'].values,
        'tick_label': length_df['Bucket'].values,
        'x_label': 'Message Word-Length Category',
        'y_label': 'Message Count',
        'title': '📏 Message Length & Complexity Distribution'
    })