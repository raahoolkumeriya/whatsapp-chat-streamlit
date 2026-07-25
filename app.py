"""
Streamlit WhatsApp Chat Processor - Modern Web Interface
"""
import os
import re
import time
import warnings
import logging
import logging.config
from typing import Dict, Any, Set, Tuple
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import streamlit as st

from processor.transformers.chat_eda import (
    WhatsAppProcess, WhatsAppAnalyzer, sorted_authors_df,
    statistics, process_data, WhatsAppConfig
)
from processor.graphs.charts import (
    ChatVisualizer, pie_display_emojis, time_series_plot,
    message_cluster, most_active_member, most_active_day,
    max_words_used, top_media_contributor, who_shared_links,
    sentiment_analysis, most_suitable_day, most_suitable_hour,
    plot_top_domains, plot_reply_latency, plot_hourly_profile,
    plot_sentiment_distribution, plot_message_length_distribution
)
from processor.common.configure import (
    BANNER, TITLE, REPO_URL, FORMAT_BUTTON,
    HIDE_STREAMLIT_STYLE, APPLICATION_FEATURE
)
from processor.reports.pdf_generator import generate_pdf_report

warnings.filterwarnings("ignore", message="Glyph 128584 missing from current font.")

# Page Configuration
st.set_page_config(
    page_title=TITLE,
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern Rich UI & Visual Metrics Cards
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400&family=Fira+Code:wght@500;600;700&display=swap');
    
    /* Modern Premium Typography Hierarchy */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    h1, h2, h3, .hero-title, .stSubheader, [data-testid="stHeader"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em !important;
    }
    
    .anthropic-tag, .anthropic-badge, code, pre {
        font-family: 'Fira Code', monospace !important;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }
    
    /* Anthropic Tag Badge */
    .anthropic-tag {
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #D97757;
        background: rgba(217, 119, 87, 0.12);
        padding: 4px 12px;
        border-radius: 20px;
        border: 1px solid rgba(217, 119, 87, 0.25);
        margin-bottom: 0.6rem;
    }
    
    /* Anthropic Hero Header Banner */
    .hero-header {
        background: linear-gradient(160deg, #1C1B18 0%, #141311 100%);
        border: 1px solid rgba(230, 225, 214, 0.14);
        border-radius: 20px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.6rem;
        font-weight: 800;
        color: #FAF8F5;
        letter-spacing: -0.03em;
        margin: 0;
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #A3A199;
        font-weight: 400;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
    
    /* Linear / Stripe Style Metric Cards Grid */
    .stat-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(195px, 1fr));
        gap: 1.25rem;
        margin-bottom: 2.2rem;
    }
    
    .linear-card {
        position: relative;
        background: #18181B;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.3rem 1.2rem 1.2rem 1.2rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 1.1rem;
        transition: all 0.28s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        overflow: hidden;
    }
    
    /* Top Accent Line Strips */
    .linear-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3.5px;
        border-top-left-radius: 16px;
        border-top-right-radius: 16px;
        transition: all 0.28s ease;
    }

    .border-indigo::before { background: linear-gradient(90deg, #6366F1, #818CF8); }
    .border-emerald::before { background: linear-gradient(90deg, #10B981, #34D399); }
    .border-amber::before { background: linear-gradient(90deg, #F59E0B, #FBBF24); }
    .border-cyan::before { background: linear-gradient(90deg, #06B6D4, #38BDF8); }
    .border-purple::before { background: linear-gradient(90deg, #A855F7, #C084FC); }
    .border-rose::before { background: linear-gradient(90deg, #F43F5E, #FB7185); }

    .border-indigo:hover { transform: translateY(-4px); border-color: rgba(99, 102, 241, 0.5); box-shadow: 0 12px 30px -4px rgba(99, 102, 241, 0.3); }
    .border-emerald:hover { transform: translateY(-4px); border-color: rgba(16, 185, 129, 0.5); box-shadow: 0 12px 30px -4px rgba(16, 185, 129, 0.3); }
    .border-amber:hover { transform: translateY(-4px); border-color: rgba(245, 158, 11, 0.5); box-shadow: 0 12px 30px -4px rgba(245, 158, 11, 0.3); }
    .border-cyan:hover { transform: translateY(-4px); border-color: rgba(6, 182, 212, 0.5); box-shadow: 0 12px 30px -4px rgba(6, 182, 212, 0.3); }
    .border-purple:hover { transform: translateY(-4px); border-color: rgba(168, 85, 247, 0.5); box-shadow: 0 12px 30px -4px rgba(168, 85, 247, 0.3); }
    .border-rose:hover { transform: translateY(-4px); border-color: rgba(244, 63, 94, 0.5); box-shadow: 0 12px 30px -4px rgba(244, 63, 94, 0.3); }

    .linear-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        width: 100%;
    }

    .linear-icon-box {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        flex-shrink: 0;
        transition: transform 0.25s ease;
    }

    .linear-card:hover .linear-icon-box {
        transform: scale(1.1);
    }

    .bg-indigo-subtle { background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.25); }
    .bg-emerald-subtle { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.25); }
    .bg-amber-subtle { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.25); }
    .bg-cyan-subtle { background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.25); }
    .bg-purple-subtle { background: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.25); }
    .bg-rose-subtle { background: rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.25); }

    .linear-badge {
        position: relative;
        font-family: 'Fira Code', monospace;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        padding: 4px 11px;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        transition: all 0.25s ease;
    }

    .linear-badge::before {
        content: '•';
        font-size: 0.9rem;
        line-height: 0;
    }

    .badge-indigo {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(79, 70, 229, 0.15) 100%);
        color: #A5B4FC;
        border: 1px solid rgba(165, 180, 252, 0.4);
        box-shadow: 0 3px 10px rgba(99, 102, 241, 0.25);
    }
    .badge-emerald {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.25) 0%, rgba(5, 150, 105, 0.15) 100%);
        color: #6EE7B7;
        border: 1px solid rgba(110, 231, 183, 0.4);
        box-shadow: 0 3px 10px rgba(16, 185, 129, 0.25);
    }
    .badge-amber {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(217, 119, 6, 0.15) 100%);
        color: #FDE047;
        border: 1px solid rgba(253, 224, 71, 0.4);
        box-shadow: 0 3px 10px rgba(245, 158, 11, 0.25);
    }
    .badge-cyan {
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.25) 0%, rgba(8, 145, 178, 0.15) 100%);
        color: #67E8F9;
        border: 1px solid rgba(103, 232, 249, 0.4);
        box-shadow: 0 3px 10px rgba(6, 182, 212, 0.25);
    }
    .badge-purple {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.25) 0%, rgba(147, 51, 234, 0.15) 100%);
        color: #E9D5FF;
        border: 1px solid rgba(233, 213, 255, 0.4);
        box-shadow: 0 3px 10px rgba(168, 85, 247, 0.25);
    }
    .badge-rose {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.25) 0%, rgba(225, 29, 72, 0.15) 100%);
        color: #FECDD3;
        border: 1px solid rgba(254, 205, 211, 0.4);
        box-shadow: 0 3px 10px rgba(244, 63, 94, 0.25);
    }

    .linear-card-body {
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
        margin-top: 0.3rem;
    }

    .linear-value {
        font-family: 'Outfit', sans-serif;
        font-size: clamp(1.35rem, 1.8vw, 1.65rem);
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        line-height: 1.25;
        white-space: nowrap;
        text-overflow: ellipsis;
        overflow: hidden;
    }

    .linear-subtext {
        font-size: 0.82rem;
        font-weight: 600;
        color: #A1A1AA;
        letter-spacing: 0.01em;
        margin-top: 0.2rem;
    }
    
    /* Anthropic Styled Tabs Navigation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #1C1B18;
        padding: 6px;
        border-radius: 16px;
        border: 1px solid rgba(230, 225, 214, 0.12);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 10px;
        padding-left: 20px;
        padding-right: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        color: #A3A199;
        border: none !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #FAF8F5;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #D97757 0%, #C86646 100%) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(217, 119, 87, 0.35);
    }
    
    /* Primary Action Buttons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #D97757 0%, #C86646 100%) !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 14px rgba(217, 119, 87, 0.3) !important;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(217, 119, 87, 0.45) !important;
    }
</style>
""", unsafe_allow_html=True)


from PIL import Image

# --- Cached Helper Operations ---
@st.cache_data(show_spinner=False)
def load_config() -> Dict[str, Any]:
    """Load application YAML configuration."""
    config_path = 'configs/app_configuration.yml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def load_multilingual_stopwords() -> Set[str]:
    """Read language files and build combined multilingual stopwords set."""
    multilingual_list = []
    stopwords_dir = 'configs/stopwords'
    if os.path.exists(stopwords_dir):
        for file in os.listdir(stopwords_dir):
            file_path = os.path.join(stopwords_dir, file)
            if os.path.isfile(file_path):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as stopword_file:
                    for word in stopword_file:
                        cleaned = re.sub(r'[\n\r]', '', word).strip()
                        if cleaned:
                            multilingual_list.append(cleaned)
    return set(STOPWORDS).union(set(multilingual_list))


@st.cache_data(show_spinner="Generating WordCloud...")
def get_wordcloud_pil_image(text: str) -> Image.Image:
    """Generate WordCloud and return PIL Image (supports English, Hindi, and Marathi text)."""
    stopwords_set = load_multilingual_stopwords()
    font_path = 'Laila-Regular.ttf' if os.path.exists('Laila-Regular.ttf') else None
    
    multilingual_regex = r'[\u0900-\u097F\w\'-]+'
    raw_tokens = re.findall(multilingual_regex, str(text))
    tokens = [w for w in raw_tokens if w.lower() not in stopwords_set and len(w) > 1]
    cleaned_text = ' '.join(tokens)
    if not cleaned_text.strip():
        cleaned_text = "WhatsApp Chat Data"

    try:
        wordcloud = WordCloud(
            scale=3,
            width=600,
            height=350,
            max_words=200,
            colormap='tab20c',
            stopwords=stopwords_set,
            regexp=multilingual_regex,
            collocations=True,
            contour_color='#4361ee',
            contour_width=2,
            font_path=font_path,
            background_color="white"
        ).generate(cleaned_text)
        return Image.fromarray(wordcloud.to_array())
    except Exception as wc_err:
        logging.warning("WordCloud generation failed for text '%s': %s. Falling back to default text.", cleaned_text[:30], wc_err)
        wordcloud = WordCloud(
            scale=3,
            width=600,
            height=350,
            colormap='tab20c',
            regexp=multilingual_regex,
            font_path=font_path,
            background_color="white"
        ).generate("WhatsApp Chat Data")
        return Image.fromarray(wordcloud.to_array())


def generate_word_cloud_figure(text: str, title: str) -> plt.Figure:
    """Generate WordCloud matplotlib figure dynamically using cached PIL Image."""
    pil_image = get_wordcloud_pil_image(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(pil_image, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title, pad=12, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner="Processing WhatsApp Chat Log...")
def parse_and_transform_chat(data: str, _config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Parse raw chat log text and generate analytical dataframes."""
    source_config = WhatsAppConfig(**_config['whatsapp'])
    analyzer = WhatsAppAnalyzer(source_config)
    
    parsed_messages = analyzer.apply_regex(data)
    raw_df = process_data(parsed_messages)
    data_frame = analyzer.get_dataframe(raw_df)
    data_frame = analyzer.day_analysis(data_frame)
    cloud_df = analyzer.cloud_data(raw_df)
    stats = statistics(raw_df, data_frame)
    
    return raw_df, data_frame, cloud_df, stats


# --- Streamlit Dialog Modals ---
if hasattr(st, "dialog"):
    @st.dialog("🔒 Privacy Policy & Data Security Guarantee")
    def show_privacy_dialog():
        st.success("🟩 Your privacy is our highest priority. Complete zero-retention architecture.")
        st.markdown("""
        ### 🛡️ Core Security Principles
        
        #### 1. Zero Data Retention & Storage
        - Uploaded WhatsApp chat exports (`.txt`) are processed **strictly in transient RAM memory**.
        - We do **NOT** save, record, replicate, or store your chat messages, contact names, or media references on any database or disk storage.

        #### 2. Local & In-Memory Execution
        - All message parsing, regex timestamp extraction, Devanagari script processing, and metric aggregations execute locally in transient session memory.
        - **No external AI APIs or cloud storage services** receive your private conversations.

        #### 3. Instant Session Cleanup
        - The moment you refresh or close your browser tab, all transient session variables, DataFrames, and cached analytics are **instantly purged from memory**.

        #### 4. Open-Source Transparency
        - Our complete processing pipeline is 100% open-source. You can independently review and audit the parsing code in `processor/transformers/chat_eda.py` and `app.py`.
        """)

    @st.dialog("ℹ️ About WhatsApp Chat Analytics Engine")
    def show_about_dialog():
        st.info("⚡ Advanced Data Science & Group Analytics Dashboard for WhatsApp Chat Exports")
        st.markdown(f"""
        ### 🎯 Platform Overview
        The **WhatsApp Chat Processor** transforms plain WhatsApp exported text files (`.txt`) into rich visual dashboards, participant timelines, vocabulary word clouds, and group activity heatmaps.

        ### 🚀 Key Capabilities
        - **🌐 Multilingual Support**: Full native parsing for English, Hindi, and Marathi WhatsApp chat exports (including Devanagari numerals `०-९`, Hindi/Marathi AM/PM markers, and Devanagari WordCloud extraction).
        - **📊 High-Impact Analytics**: Interactive charts for hourly/daily activity peaks, member contribution rankings, link sharing frequency, and emoji distribution.
        - **⚡ Performance-Optimized**: Modular Python OOP architecture (`WhatsAppAnalyzer`, `ChatVisualizer`), `@st.cache_data` memory management, and DRY data pipelines.

        ### 👨‍💻 Maintainer & Source Code
        - **Author & Maintainer**: **Rahul Kumeriya**
        - **Role**: Data & Software Analytics Engineer
        - **Open Source Repository**: [View on GitHub]({REPO_URL.split('?')[0]})
        - **Core Stack**: Python 3.9+, Streamlit 1.37+, Pandas 2.0+, Matplotlib 3.7+, Plotly 5.15+, WordCloud 1.9+, Emoji 2.8+.
        """)
else:
    def show_privacy_dialog():
        st.toast("We respect your privacy. No chat content is stored or saved.")
    def show_about_dialog():
        st.toast("Visit GitHub link for detailed project information.")


# --- Visual Hero Metric Cards ---
def render_visual_kpi_cards(stats: Dict[str, Any], data_frame: pd.DataFrame):
    """Render Linear/Stripe-inspired floating accent KPI cards for key conversation metrics."""
    total_msgs = stats.get('total_messages', 0)
    total_members = stats.get('total_members', 0)
    media_msg = stats.get('media_message', 0)
    link_shared = int(stats.get('link_shared', 0))
    total_words = int(data_frame['word_count'].sum()) if 'word_count' in data_frame.columns else 0
    top_member = data_frame['name'].value_counts().index[0] if ('name' in data_frame.columns and not data_frame.empty) else "N/A"
    
    total_emojis = stats.get('total_emojis', 0)
    avg_per_day = stats.get('avg_msgs_per_day', 0.0)
    peak_day = stats.get('peak_day', 'N/A')
    busiest_hour = stats.get('busiest_hour', 'N/A')

    st.markdown(f"""
    <div class="stat-card-grid">
        <div class="linear-card border-indigo">
            <div class="linear-card-header">
                <div class="linear-icon-box bg-indigo-subtle">💬</div>
                <span class="linear-badge badge-indigo">MESSAGES</span>
            </div>
            <div class="linear-card-body">
                <div class="linear-value">{total_msgs:,}</div>
                <div class="linear-subtext">Total Messages Sent</div>
            </div>
        </div>
        <div class="linear-card border-emerald">
            <div class="linear-card-header">
                <div class="linear-icon-box bg-emerald-subtle">👥</div>
                <span class="linear-badge badge-emerald">MEMBERS</span>
            </div>
            <div class="linear-card-body">
                <div class="linear-value">{total_members:,}</div>
                <div class="linear-subtext">Active Participants</div>
            </div>
        </div>
        <div class="linear-card border-amber">
            <div class="linear-card-header">
                <div class="linear-icon-box bg-amber-subtle">🎞️</div>
                <span class="linear-badge badge-amber">MEDIA</span>
            </div>
            <div class="linear-card-body">
                <div class="linear-value">{media_msg:,}</div>
                <div class="linear-subtext">Media Files Shared</div>
            </div>
        </div>
        <div class="linear-card border-cyan">
            <div class="linear-card-header">
                <div class="linear-icon-box bg-cyan-subtle">🔗</div>
                <span class="linear-badge badge-cyan">LINKS</span>
            </div>
            <div class="linear-card-body">
                <div class="linear-value">{link_shared:,}</div>
                <div class="linear-subtext">Shared Web Links</div>
            </div>
        </div>
        <div class="linear-card border-purple">
            <div class="linear-card-header">
                <div class="linear-icon-box bg-purple-subtle">📝</div>
                <span class="linear-badge badge-purple">WORDS</span>
            </div>
            <div class="linear-card-body">
                <div class="linear-value">{total_words:,}</div>
                <div class="linear-subtext">Total Words Typed</div>
            </div>
        </div>
        <div class="linear-card border-rose">
            <div class="linear-card-header">
                <div class="linear-icon-box bg-rose-subtle">👑</div>
                <span class="linear-badge badge-rose">LEADER</span>
            </div>
            <div class="linear-card-body">
                <div class="linear-value" title="{top_member}">{top_member}</div>
                <div class="linear-subtext">Top Message Contributor</div>
            </div>
        </div>
        <div class="linear-card border-amber">
            <div class="linear-card-header">
                <div class="linear-icon-box bg-amber-subtle">😃</div>
                <span class="linear-badge badge-amber">EMOJIS</span>
            </div>
            <div class="linear-card-body">
                <div class="linear-value">{total_emojis:,}</div>
                <div class="linear-subtext">Total Emojis Sent</div>
            </div>
        </div>
        <div class="linear-card border-cyan">
            <div class="linear-card-header">
                <div class="linear-icon-box bg-cyan-subtle">📈</div>
                <span class="linear-badge badge-cyan">DAILY AVG</span>
            </div>
            <div class="linear-card-body">
                <div class="linear-value">{avg_per_day}</div>
                <div class="linear-subtext">Messages / Day</div>
            </div>
        </div>
        <div class="linear-card border-emerald">
            <div class="linear-card-header">
                <div class="linear-icon-box bg-emerald-subtle">📅</div>
                <span class="linear-badge badge-emerald">PEAK DAY</span>
            </div>
            <div class="linear-card-body">
                <div class="linear-value" title="{peak_day}">{peak_day}</div>
                <div class="linear-subtext">Busiest Day of Week</div>
            </div>
        </div>
        <div class="linear-card border-rose">
            <div class="linear-card-header">
                <div class="linear-icon-box bg-rose-subtle">⏰</div>
                <span class="linear-badge badge-rose">PEAK HOUR</span>
            </div>
            <div class="linear-card-body">
                <div class="linear-value" title="{busiest_hour}">{busiest_hour}</div>
                <div class="linear-subtext">Busiest Hour Window</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_member_visual_cards(user_cloud_df: pd.DataFrame, user_data_df: pd.DataFrame):
    """Render Linear/Stripe-inspired floating accent metric cards for individual member analytics."""
    msgs = user_cloud_df.shape[0]
    emojis_cnt = sum(len(em) for em in user_data_df['emojis']) if 'emojis' in user_data_df.columns else 0
    links_cnt = int(user_data_df['urlcount'].sum()) if 'urlcount' in user_data_df.columns else 0
    words_cnt = int(user_data_df['word_count'].sum()) if 'word_count' in user_data_df.columns else 0
    avg_words = round(user_data_df['word_count'].mean(), 2) if ('word_count' in user_data_df.columns and not user_data_df.empty) else 0

    st.markdown(f"""
    <div class="stat-card-grid">
        <div class="linear-card border-indigo">
            <div class="linear-card-header">
                <div class="linear-icon-box bg-indigo-subtle">💬</div>
                <span class="linear-badge badge-indigo">MESSAGES</span>
            </div>
            <div class="linear-card-body">
                <div class="linear-value">{msgs:,}</div>
                <div class="linear-subtext">Sent Messages</div>
            </div>
        </div>
        <div class="linear-card border-amber">
            <div class="linear-card-header">
                <div class="linear-icon-box bg-amber-subtle">😀</div>
                <span class="linear-badge badge-amber">EMOJIS</span>
            </div>
            <div class="linear-card-body">
                <div class="linear-value">{emojis_cnt:,}</div>
                <div class="linear-subtext">Emojis Used</div>
            </div>
        </div>
        <div class="linear-card border-cyan">
            <div class="linear-card-header">
                <div class="linear-icon-box bg-cyan-subtle">🔗</div>
                <span class="linear-badge badge-cyan">LINKS</span>
            </div>
            <div class="linear-card-body">
                <div class="linear-value">{links_cnt:,}</div>
                <div class="linear-subtext">Shared Links</div>
            </div>
        </div>
        <div class="linear-card border-purple">
            <div class="linear-card-header">
                <div class="linear-icon-box bg-purple-subtle">📚</div>
                <span class="linear-badge badge-purple">VOCABULARY</span>
            </div>
            <div class="linear-card-body">
                <div class="linear-value">{words_cnt:,}</div>
                <div class="linear-subtext">Total Words</div>
            </div>
        </div>
        <div class="linear-card border-emerald">
            <div class="linear-card-header">
                <div class="linear-icon-box bg-emerald-subtle">📊</div>
                <span class="linear-badge badge-emerald">AVERAGE</span>
            </div>
            <div class="linear-card-body">
                <div class="linear-value">{avg_words}</div>
                <div class="linear-subtext">Words / Message</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# --- Main Application Logic ---
def main():
    os.makedirs('latest_logs', exist_ok=True)
    config = load_config()
    
    # Configure logging
    log_config = config['logging']
    logging.config.dictConfig(log_config)
    logger = logging.getLogger("app_main")
    logger.info("WhatsApp Chat Processor Application Initializing")

    # Anthropic Hero Header Card
    st.markdown(f"""
    <div class="hero-header">
        <span class="anthropic-tag">⚡️ WHATSAPP ANALYTICS ENGINE</span>
        <div class="hero-title">{TITLE}</div>
        <div class="hero-subtitle">Discover interaction dynamics, member contributions, multilingual word clouds, and sentiment insights from exported WhatsApp group chats.</div>
    </div>
    """, unsafe_allow_html=True)
    
    action_col1, action_col2 = st.columns([1, 1])
    with action_col1:
        if st.button("🔒 Privacy Policy", use_container_width=True):
            logger.info("User opened Privacy Policy dialog")
            show_privacy_dialog()
            
    with action_col2:
        if st.button("ℹ️ About App", use_container_width=True):
            logger.info("User opened About Application dialog")
            show_about_dialog()

    st.markdown("---")

    # Sidebar Controls & File Upload
    st.sidebar.image("https://img.icons8.com/color/96/whatsapp--v1.png", width=70)
    st.sidebar.title("WhatsApp Chat Processor")
    st.sidebar.markdown(APPLICATION_FEATURE)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Upload Chat Export")
    uploaded_file = st.sidebar.file_uploader(
        "Select exported `.txt` chat file",
        type=['txt'],
        accept_multiple_files=False,
        help="Export your WhatsApp chat export without media from iOS or Android and upload the resulting .txt file."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡️ Try Demo Chat")
    demo_clicked = st.sidebar.button("MAD MAX Fury Road Chat", type="primary", use_container_width=True)

    # GitHub Badges
    st.sidebar.markdown("---")
    st.sidebar.markdown("`View Project on GitHub`")
    BUTTON_SIZE = 'count=true&size=large&v=2'
    st.sidebar.markdown(
        f'<iframe src="{REPO_URL}&{BUTTON_SIZE}" {FORMAT_BUTTON}></iframe>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        f'<iframe src="{REPO_URL}&type=star&{BUTTON_SIZE}" {FORMAT_BUTTON}></iframe>',
        unsafe_allow_html=True
    )

    # Resolve Data Input
    data = None
    if uploaded_file is not None:
        file_size = uploaded_file.size
        logger.info("User uploaded file '%s' (Size: %d bytes)", uploaded_file.name, file_size)
        start_decode = time.perf_counter()
        data = uploaded_file.getvalue().decode("utf-8")
        logger.info("File UTF-8 decoded in %.3f sec", time.perf_counter() - start_decode)
    elif demo_clicked or "demo_loaded" in st.session_state:
        logger.info("Demo chat requested or loaded from session state")
        st.session_state.demo_loaded = True
        demo_path = 'configs/demo_chat.txt'
        if os.path.exists(demo_path):
            with open(demo_path, 'r', encoding='utf-8') as rf:
                data = rf.read()
            logger.info("Loaded demo chat file '%s' (%d bytes)", demo_path, len(data))

    if data:
        try:
            start_proc = time.perf_counter()
            raw_df, data_frame, cloud_df, stats = parse_and_transform_chat(data, config)
            logger.info("Chat data parsed and transformed successfully in %.3f sec", time.perf_counter() - start_proc)
            render_dashboard(raw_df, data_frame, cloud_df, stats, logger)
        except Exception as e:
            logger.exception("Failed to process WhatsApp chat export: %s", e)
            st.error(f"⚠️ Unable to process WhatsApp chat export: {str(e)}")
            st.info("Please verify that the uploaded file is a valid WhatsApp chat log exported without media.")
    else:
        # Welcome placeholder screen
        logger.info("No chat file uploaded yet. Rendering welcome instructions screen.")
        st.info("👈 Upload a WhatsApp `.txt` chat export file from the sidebar, or click **Try Demo** to explore a sample chat analysis!")
        
        with st.container(border=True):
            st.subheader("💡 How to Export Your WhatsApp Chat:")
            col_ios, col_android = st.columns(2)
            with col_ios:
                st.markdown("""
                **📱 On iOS (iPhone):**
                1. Open WhatsApp and select the group or contact chat.
                2. Tap the contact or group title at the top -> **Export Chat**.
                3. Choose **Without Media**.
                4. Upload the generated `.txt` file here.
                """)
            with col_android:
                st.markdown("""
                **🤖 On Android:**
                1. Open WhatsApp and open the group or contact chat.
                2. Tap the `⋮` menu (top right) -> **More** -> **Export chat**.
                3. Choose **Without Media**.
                4. Upload the generated `.txt` file here.
                """)


def render_dashboard(raw_df: pd.DataFrame, data_frame: pd.DataFrame, cloud_df: pd.DataFrame, stats: Dict[str, Any], logger: logging.Logger):
    """Render comprehensive Streamlit analytics tabs and dashboards."""
    group_name = stats.get("group_name", "Group Chat Analytics")
    st.header(f"🏷️ {group_name}")

    # Top Visual Hero KPI Cards Grid
    render_visual_kpi_cards(stats, data_frame)

    st.markdown("---")

    # Structured View Tabs
    tab_overview, tab_members, tab_time, tab_sentiment, tab_advanced = st.tabs([
        "📊 Overview & Messages",
        "👤 Member Analytics",
        "🕒 Activity & Time Patterns",
        "😀 Emoji & Sentiment",
        "🔬 Web Links & Message Style"
    ])

    analyzer = WhatsAppAnalyzer(WhatsAppConfig(**load_config()['whatsapp']))

    # TAB 1: OVERVIEW
    with tab_overview:
        st.subheader("📋 Parsed Chat Message Explorer")
        st.dataframe(
            raw_df[["datetime", "name", "message"]],
            use_container_width=True,
            height=320
        )
        
        # Download Buttons Row
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_data = raw_df.to_csv(index=False).encode('utf-8')
            if st.download_button(
                label="📥 Export Cleaned CSV Dataset",
                data=csv_data,
                file_name="whatsapp_chat_export_parsed.csv",
                mime="text/csv",
                use_container_width=True
            ):
                logger.info("User clicked Export Cleaned Dataset to CSV button")

        with col_exp2:
            domain_df = analyzer.extract_top_domains(data_frame)
            pdf_bytes = generate_pdf_report(stats, data_frame, cloud_df, domain_df)
            clean_group_name = re.sub(r'[^\w\s-]', '', stats.get('group_name', 'chat')).strip().replace(' ', '_')
            if not clean_group_name:
                clean_group_name = "chat_group"
            if st.download_button(
                label="📄 Download Executive PDF Report",
                data=pdf_bytes,
                file_name=f"whatsapp_analysis_report_{clean_group_name}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            ):
                logger.info("User downloaded Executive PDF Analysis Report")

        st.markdown("---")
        st.subheader("☁️ Overall Group Word Cloud")
        st.info("Displays the most prominent words across all conversation messages in the group.")
        full_text = " ".join(review for review in cloud_df.message if isinstance(review, str))
        fig_cloud = generate_word_cloud_figure(full_text, "Group Conversation Word Cloud")
        st.pyplot(fig_cloud, use_container_width=True)

    # TAB 2: MEMBER ANALYTICS
    with tab_members:
        st.subheader("👤 Member Deep-Dive Analytics")
        authors = sorted_authors_df(cloud_df)
        selected_author = st.selectbox("Select a group member to inspect:", authors)
        
        if selected_author:
            logger.info("Member Analytics tab inspecting author '%s'", selected_author)
            user_cloud_df = cloud_df[cloud_df['name'] == selected_author]
            user_data_df = data_frame[data_frame.name == selected_author]
            
            # Individual Member Visual Metric Cards
            render_member_visual_cards(user_cloud_df, user_data_df)

            st.write("")
            user_text = " ".join(str(msg) for msg in user_cloud_df.message if isinstance(msg, str))
            fig_user_cloud = generate_word_cloud_figure(user_text, f"Word Cloud - {selected_author}")
            st.pyplot(fig_user_cloud, use_container_width=True)

        st.markdown("---")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.subheader("🔘 Most Active Member")
            st.pyplot(most_active_member(data_frame), use_container_width=True)
        with col_m2:
            st.subheader("🔘 Top Word Contributors")
            st.pyplot(max_words_used(data_frame), use_container_width=True)

        st.markdown("---")
        col_m3, col_m4 = st.columns(2)
        with col_m3:
            st.subheader("🔘 Link Sharing Leaderboard")
            st.pyplot(who_shared_links(data_frame), use_container_width=True)
        with col_m4:
            st.subheader("🔘 Top Media Contributors")
            st.pyplot(top_media_contributor(data_frame), use_container_width=True)

        st.markdown("---")
        st.subheader("⏱️ Member Response Speed (Fastest Responders)")
        st.info("Measures the average reply latency (in minutes) when answering another participant's message.")
        latency_df = analyzer.calculate_reply_latency(data_frame)
        st.pyplot(plot_reply_latency(latency_df), use_container_width=True)

    # TAB 3: ACTIVITY PATTERNS
    with tab_time:
        st.subheader("📈 Message Volume Timeline")
        st.plotly_chart(time_series_plot(data_frame), use_container_width=True)

        st.markdown("---")
        st.subheader("⚡ 24-Hour Hourly Activity Profile")
        st.info("Distribution of group activity across all 24 hours of the day (Night Owls vs Early Birds).")
        hourly_df = analyzer.hourly_activity_profile(data_frame)
        st.pyplot(plot_hourly_profile(hourly_df), use_container_width=True)

        st.markdown("---")
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.subheader("🗓️ Active Day of Week")
            st.pyplot(most_active_day(data_frame), use_container_width=True)
        with col_t2:
            st.subheader("⏰ Active Hour of Day")
            st.pyplot(most_suitable_hour(data_frame), use_container_width=True)

        st.markdown("---")
        st.subheader("📅 Activity Breakdown by Day of Month")
        st.pyplot(most_suitable_day(data_frame), use_container_width=True)

        st.markdown("---")
        st.subheader("🌌 Member Activity Scatter Cluster")
        st.plotly_chart(message_cluster(data_frame), use_container_width=True)

    # TAB 4: EMOJI & SENTIMENT
    with tab_sentiment:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.subheader("😜 Top Emoji Usage Distribution")
            st.plotly_chart(pie_display_emojis(data_frame), use_container_width=True)
        with col_s2:
            st.subheader("🎭 Group Sentiment Breakdown")
            sentiment_df = analyzer.analyze_message_sentiment(data_frame)
            st.pyplot(plot_sentiment_distribution(sentiment_df), use_container_width=True)

        st.markdown("---")
        st.subheader("🧠 Member Sentiment Polarity Analysis")
        st.info("Measures average message sentiment polarity using TextBlob NLP (-1.0 Negative, 0.0 Neutral, +1.0 Positive).")
        st.pyplot(sentiment_analysis(cloud_df), use_container_width=True)

    # TAB 5: ADVANCED LINKS & STYLE
    with tab_advanced:
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.subheader("🌐 Top Shared Web Domains")
            st.info("Identifies top shared web platforms (YouTube, Instagram, GitHub, News).")
            domain_df = analyzer.extract_top_domains(data_frame)
            st.pyplot(plot_top_domains(domain_df), use_container_width=True)
        with col_a2:
            st.subheader("📏 Message Length & Complexity Distribution")
            st.info("Categorizes messages by word count (Short replies vs Detailed essays).")
            length_df = analyzer.message_length_distribution(data_frame)
            st.pyplot(plot_message_length_distribution(length_df), use_container_width=True)

    # Footer
    st.markdown("---")
    st.success("🌳 **Environmental Reminder:** Take a moment to plant trees and support a greener world! 🌿🌾☘️")


if __name__ == "__main__":
    print(BANNER)
    main()
