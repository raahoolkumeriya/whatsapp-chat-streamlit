"""
DuckDB In-Memory High-Performance Analytics Engine for WhatsApp Chat Exports.
Accelerates dataset queries, aggregations, timelines, and statistics using DuckDB vectorized C++ SQL execution.
Compatible with Streamlit Cloud deployments.
"""
import time
import logging
from typing import Dict, Any
import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


class DuckDBChatEngine:
    """
    In-memory DuckDB analytical query engine for ultra-fast WhatsApp chat aggregations.
    """
    def __init__(self, data_frame: pd.DataFrame, raw_df: pd.DataFrame = None):
        self.start_time = time.perf_counter()
        self.conn = duckdb.connect(database=':memory:')
        self._logger = logging.getLogger(__name__)
        
        # Register Pandas DataFrames into DuckDB in-memory tables instantly
        self.conn.register('data_frame_view', data_frame)
        if raw_df is not None:
            self.conn.register('raw_df_view', raw_df)

        # Create optimized physical DuckDB tables with vectorized columnar storage
        self.conn.execute("""
            CREATE TABLE chat_messages AS 
            SELECT 
                CAST(datetime AS TIMESTAMP) as datetime,
                CAST(name AS VARCHAR) as name,
                CAST(message AS VARCHAR) as message,
                CAST(letter_count AS INTEGER) as letter_count,
                CAST(word_count AS INTEGER) as word_count,
                CAST(urlcount AS INTEGER) as urlcount,
                CAST(media AS INTEGER) as media,
                DAYNAME(CAST(datetime AS TIMESTAMP)) as day
            FROM data_frame_view
        """)
        
        if raw_df is not None:
            self.conn.execute("""
                CREATE TABLE raw_messages AS 
                SELECT 
                    CAST(datetime AS TIMESTAMP) as datetime,
                    CAST(name AS VARCHAR) as name,
                    CAST(message AS VARCHAR) as message
                FROM raw_df_view
            """)

        elapsed = time.perf_counter() - self.start_time
        self._logger.info("DuckDB in-memory engine initialized with %d rows in %.4f sec", len(data_frame), elapsed)

    def get_statistics(self, group_name: str = "Group") -> Dict[str, Any]:
        """Execute unified DuckDB SQL query to calculate all summary statistics in one pass."""
        t0 = time.perf_counter()
        
        sql = """
            SELECT 
                COUNT(*) as total_messages,
                COUNT(DISTINCT name) as total_members,
                COALESCE(SUM(urlcount), 0) as link_shared,
                COALESCE(SUM(media), 0) as media_message,
                COUNT(DISTINCT CAST(datetime AS DATE)) as active_days
            FROM chat_messages
        """
        res = self.conn.execute(sql).fetchone()
        total_messages, total_members, link_shared, media_message, active_days = res

        # Peak Day of week
        peak_day_res = self.conn.execute("""
            SELECT day, COUNT(*) as cnt 
            FROM chat_messages 
            WHERE day IS NOT NULL 
            GROUP BY day 
            ORDER BY cnt DESC 
            LIMIT 1
        """).fetchone()
        peak_day = peak_day_res[0] if peak_day_res else "N/A"

        # Peak Hour
        peak_hour_res = self.conn.execute("""
            SELECT EXTRACT(hour FROM datetime) as hr, COUNT(*) as cnt 
            FROM chat_messages 
            GROUP BY hr 
            ORDER BY cnt DESC 
            LIMIT 1
        """).fetchone()
        if peak_hour_res:
            top_h = int(peak_hour_res[0])
            ampm = "AM" if top_h < 12 else "PM"
            disp_h = top_h if top_h <= 12 else top_h - 12
            if disp_h == 0:
                disp_h = 12
            busiest_hour = f"{disp_h} {ampm}"
        else:
            busiest_hour = "N/A"

        avg_msgs_per_day = round(total_messages / max(active_days, 1), 1)

        # Deleted messages from raw
        deleted_count = 0
        your_deleted_count = 0
        try:
            d_res = self.conn.execute("""
                SELECT 
                    SUM(CASE WHEN message = 'This message was deleted' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN message = 'You deleted this message' THEN 1 ELSE 0 END)
                FROM raw_messages
            """).fetchone()
            deleted_count = d_res[0] or 0
            your_deleted_count = d_res[1] or 0
        except Exception:
            pass

        elapsed = time.perf_counter() - t0
        self._logger.info("DuckDB get_statistics executed in %.4f sec", elapsed)

        return {
            "media_message": int(media_message),
            "total_deleted_messages": int(deleted_count),
            "your_deleted_message": int(your_deleted_count),
            "group_name": group_name,
            "total_messages": int(total_messages),
            "total_members": int(total_members),
            "link_shared": int(link_shared),
            "avg_msgs_per_day": avg_msgs_per_day,
            "peak_day": peak_day,
            "busiest_hour": busiest_hour
        }

    def get_most_active_members(self, top_n: int = 10) -> pd.DataFrame:
        """SQL query for top active members by message count."""
        return self.conn.execute(f"""
            SELECT name, COUNT(*) as message_count
            FROM chat_messages
            GROUP BY name
            ORDER BY message_count DESC
            LIMIT {top_n}
        """).df()

    def get_hourly_profile(self) -> pd.DataFrame:
        """SQL query for 24-hour activity distribution."""
        df = self.conn.execute("""
            SELECT 
                EXTRACT(hour FROM datetime) as hour,
                COUNT(*) as message_count
            FROM chat_messages
            GROUP BY hour
            ORDER BY hour ASC
        """).df()
        
        # Ensure complete 0-23 hours
        full_df = pd.DataFrame({'hour': range(24)})
        merged = full_df.merge(df, on='hour', how='left').fillna(0)
        merged['message_count'] = merged['message_count'].astype(int)
        merged['hour_label'] = merged['hour'].apply(
            lambda h: f"{h if h <= 12 else h-12} {'AM' if h < 12 else 'PM'}" if h != 0 else "12 AM"
        )
        return merged

    def get_reply_latency(self) -> pd.DataFrame:
        """DuckDB SQL Window query for reply latency per participant."""
        return self.conn.execute("""
            WITH ordered AS (
                SELECT 
                    name,
                    datetime,
                    LAG(name) OVER (ORDER BY datetime) as prev_name,
                    LAG(datetime) OVER (ORDER BY datetime) as prev_time
                FROM chat_messages
            ),
            replies AS (
                SELECT 
                    name,
                    EPOCH(datetime - prev_time) / 60.0 as reply_mins
                FROM ordered
                WHERE name != prev_name 
                  AND prev_name IS NOT NULL
                  AND EPOCH(datetime - prev_time) / 60.0 <= 120
            )
            SELECT 
                name, 
                ROUND(AVG(reply_mins), 1) as avg_reply_time_mins
            FROM replies
            GROUP BY name
            ORDER BY avg_reply_time_mins ASC
        """).df()

    def get_member_metrics(self, member_name: str) -> Dict[str, Any]:
        """Ultra-fast DuckDB query for individual member metrics."""
        res = self.conn.execute("""
            SELECT 
                COUNT(*) as msgs,
                COALESCE(SUM(urlcount), 0) as links_cnt,
                COALESCE(SUM(word_count), 0) as words_cnt,
                ROUND(COALESCE(AVG(word_count), 0), 2) as avg_words
            FROM chat_messages
            WHERE name = ?
        """, [member_name]).fetchone()
        
        if res:
            msgs, links_cnt, words_cnt, avg_words = res
        else:
            msgs, links_cnt, words_cnt, avg_words = 0, 0, 0, 0.0

        return {
            "msgs": int(msgs or 0),
            "links_cnt": int(links_cnt or 0),
            "words_cnt": int(words_cnt or 0),
            "avg_words": float(avg_words or 0.0)
        }

    def close(self):
        """Close DuckDB connection."""
        try:
            self.conn.close()
        except Exception:
            pass
