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
    sentiment_analysis, most_suitable_day, most_suitable_hour
)
from processor.common.configure import (
    BANNER, TITLE, REPO_URL, FORMAT_BUTTON,
    HIDE_STREAMLIT_STYLE, APPLICATION_FEATURE
)

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
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1250px;
    }
    
    /* Hero Header Banner */
    .hero-header {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.12) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 18px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(12px);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        font-weight: 500;
        margin-top: 0.3rem;
    }
    
    /* Visual Stat Cards Grid */
    .stat-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 1rem;
        margin-bottom: 1.8rem;
    }
    
    .stat-card {
        background: linear-gradient(145deg, #1e1e2f 0%, #151522 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.2rem 1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    .stat-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.25);
    }
    
    .stat-icon {
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        flex-shrink: 0;
    }
    
    .icon-messages { background: linear-gradient(135deg, #6366F1, #4F46E5); }
    .icon-members { background: linear-gradient(135deg, #EC4899, #D946EF); }
    .icon-media { background: linear-gradient(135deg, #F59E0B, #D97706); }
    .icon-links { background: linear-gradient(135deg, #10B981, #059669); }
    .icon-words { background: linear-gradient(135deg, #06B6D4, #0891B2); }
    .icon-top { background: linear-gradient(135deg, #8B5CF6, #7C3AED); }
    
    .stat-info {
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    .stat-value {
        font-size: 1.4rem;
        font-weight: 800;
        color: #f8fafc;
        line-height: 1.2;
        letter-spacing: -0.5px;
        white-space: nowrap;
        text-overflow: ellipsis;
        overflow: hidden;
    }
    
    .stat-label {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-top: 3px;
    }
    
    /* Modern Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(255, 255, 255, 0.03);
        padding: 6px;
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        border-radius: 10px;
        padding-left: 22px;
        padding-right: 22px;
        font-weight: 600;
        font-size: 0.92rem;
        color: #94a3b8;
        border: none !important;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366F1, #4F46E5) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
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
    """Generate WordCloud and return PIL Image (safely serialized by @st.cache_data)."""
    stopwords_set = load_multilingual_stopwords()
    font_path = 'Laila-Regular.ttf' if os.path.exists('Laila-Regular.ttf') else None
    
    tokens = [w for w in str(text).split() if w.lower() not in stopwords_set]
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
    @st.dialog("🔒 Privacy Statement")
    def show_privacy_dialog():
        st.success("🟩 We strictly protect user privacy.")
        st.write("""
        - **Zero Data Retention**: Uploaded chat exports are processed entirely in transient session memory. We do **NOT** store or replicate any chat logs.
        - **Private & Local Processing**: Text data is parsed dynamically in-memory and discarded upon session closing.
        - **Visual Insights Only**: Transforms plain raw text into clean, informative interactive graphics and statistical metrics.
        """)

    @st.dialog("ℹ️ About WhatsApp Chat Processor")
    def show_about_dialog():
        st.info("🧊 Data Science & Analytics Dashboard for WhatsApp Chat Exports")
        st.markdown(f"""
        This web application allows group administrators, researchers, and users to analyze interaction dynamics, top contributors, sentiment polarity, and activity peaks in WhatsApp group chats.
        
        - **Author / Maintainer**: Rahul Kumeriya
        - **Source Code**: [View on GitHub]({REPO_URL.split('?')[0]})
        """)
else:
    def show_privacy_dialog():
        st.toast("We respect your privacy. No chat content is stored or saved.")
    def show_about_dialog():
        st.toast("Visit GitHub link for detailed project information.")


# --- Visual Hero Metric Cards ---
def render_visual_kpi_cards(stats: Dict[str, Any], data_frame: pd.DataFrame):
    """Render high-impact visual hero stat cards for key conversation metrics."""
    total_msgs = stats.get('total_messages', 0)
    total_members = stats.get('total_members', 0)
    media_msg = stats.get('media_message', 0)
    link_shared = int(stats.get('link_shared', 0))
    total_words = int(data_frame['word_count'].sum()) if 'word_count' in data_frame.columns else 0
    top_member = data_frame['name'].value_counts().index[0] if ('name' in data_frame.columns and not data_frame.empty) else "N/A"

    st.markdown(f"""
    <div class="stat-card-grid">
        <div class="stat-card">
            <div class="stat-icon icon-messages">📨</div>
            <div class="stat-info">
                <div class="stat-value">{total_msgs:,}</div>
                <div class="stat-label">Total Messages</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon icon-members">💃🕺</div>
            <div class="stat-info">
                <div class="stat-value">{total_members:,}</div>
                <div class="stat-label">Active Members</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon icon-media">🎞️</div>
            <div class="stat-info">
                <div class="stat-value">{media_msg:,}</div>
                <div class="stat-label">Media Shared</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon icon-links">🔗</div>
            <div class="stat-info">
                <div class="stat-value">{link_shared:,}</div>
                <div class="stat-label">Links Shared</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon icon-words">📝</div>
            <div class="stat-info">
                <div class="stat-value">{total_words:,}</div>
                <div class="stat-label">Total Words</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon icon-top">👑</div>
            <div class="stat-info">
                <div class="stat-value" title="{top_member}">{top_member}</div>
                <div class="stat-label">Top Speaker</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_member_visual_cards(user_cloud_df: pd.DataFrame, user_data_df: pd.DataFrame):
    """Render visual metric cards for individual member analytics."""
    msgs = user_cloud_df.shape[0]
    emojis_cnt = sum(len(em) for em in user_data_df['emojis']) if 'emojis' in user_data_df.columns else 0
    links_cnt = int(user_data_df['urlcount'].sum()) if 'urlcount' in user_data_df.columns else 0
    words_cnt = int(user_data_df['word_count'].sum()) if 'word_count' in user_data_df.columns else 0
    avg_words = round(user_data_df['word_count'].mean(), 2) if ('word_count' in user_data_df.columns and not user_data_df.empty) else 0

    st.markdown(f"""
    <div class="stat-card-grid">
        <div class="stat-card">
            <div class="stat-icon icon-messages">💬</div>
            <div class="stat-info">
                <div class="stat-value">{msgs:,}</div>
                <div class="stat-label">Messages</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon icon-members">😜</div>
            <div class="stat-info">
                <div class="stat-value">{emojis_cnt:,}</div>
                <div class="stat-label">Emojis Shared</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon icon-links">🔗</div>
            <div class="stat-info">
                <div class="stat-value">{links_cnt:,}</div>
                <div class="stat-label">Links Shared</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon icon-words">📚</div>
            <div class="stat-info">
                <div class="stat-value">{words_cnt:,}</div>
                <div class="stat-label">Total Words</div>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-icon icon-media">📊</div>
            <div class="stat-info">
                <div class="stat-value">{avg_words}</div>
                <div class="stat-label">Avg Words/Msg</div>
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

    # Header Hero Card
    title_col, action_col1, action_col2 = st.columns([6, 2, 2])
    with title_col:
        st.title(f"💬 {TITLE}")
        st.caption("Discover member activity, conversation dynamics, word clouds, and sentiment trends in your WhatsApp group chats")
    
    with action_col1:
        st.write("")
        if st.button("🔒 Privacy Policy", use_container_width=True):
            logger.info("User opened Privacy Policy dialog")
            show_privacy_dialog()
            
    with action_col2:
        st.write("")
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
    tab_overview, tab_members, tab_time, tab_sentiment = st.tabs([
        "📊 Overview & Messages",
        "👤 Member Analytics",
        "🕒 Activity & Time Patterns",
        "😀 Emoji & Sentiment"
    ])

    # TAB 1: OVERVIEW
    with tab_overview:
        st.subheader("📋 Parsed Chat Message Explorer")
        st.dataframe(
            raw_df[["datetime", "name", "message"]],
            use_container_width=True,
            height=320
        )
        
        # Download Data Button
        csv_data = raw_df.to_csv(index=False).encode('utf-8')
        if st.download_button(
            label="📥 Export Cleaned Dataset to CSV",
            data=csv_data,
            file_name="whatsapp_chat_export_parsed.csv",
            mime="text/csv"
        ):
            logger.info("User clicked Export Cleaned Dataset to CSV button")

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

    # TAB 3: ACTIVITY PATTERNS
    with tab_time:
        st.subheader("📈 Message Volume Timeline")
        st.plotly_chart(time_series_plot(data_frame), use_container_width=True)

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
            st.subheader("🧠 Member Sentiment Polarity Analysis")
            st.info("Measures average message sentiment polarity using TextBlob NLP (-1.0 Negative, 0.0 Neutral, +1.0 Positive).")
            st.pyplot(sentiment_analysis(cloud_df), use_container_width=True)

    # Footer
    st.markdown("---")
    st.success("🌳 **Environmental Reminder:** Take a moment to plant trees and support a greener world! 🌿🌾☘️")


if __name__ == "__main__":
    print(BANNER)
    main()
