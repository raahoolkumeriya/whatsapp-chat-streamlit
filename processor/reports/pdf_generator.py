"""
WhatsApp Chat Analytics PDF Detailed Summary Report Generator Module.
Generates a comprehensive multi-page executive PDF report containing:
- Group summary statistics & KPI grid
- Member activity leaderboards
- Activity timeline & 24-hour profile charts
- Group WordCloud image
- Sentiment distribution & response latency charts
- Top shared web domains & message length charts
"""
import io
import re
import time
import logging
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, PageBreak, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from processor.graphs.charts import (
    most_active_member, most_active_day, most_suitable_day,
    most_suitable_hour, sentiment_analysis, plot_top_domains,
    plot_reply_latency, plot_hourly_profile, plot_sentiment_distribution,
    plot_message_length_distribution
)

logger = logging.getLogger(__name__)


def fig_to_rl_image(fig: plt.Figure, width: float = 500, height: float = 230) -> RLImage:
    """Convert Matplotlib Figure into ReportLab RLImage flowable safely."""
    if fig is None:
        return None
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=False, facecolor='#FFFFFF')
        plt.close(fig)
        buf.seek(0)
        return RLImage(buf, width=width, height=height)
    except Exception as e:
        logger.warning("Failed to render chart image for PDF report: %s", e)
        if fig is not None:
            plt.close(fig)
        return None


def generate_pdf_report(
    stats: Dict[str, Any],
    data_frame: pd.DataFrame,
    cloud_df: pd.DataFrame,
    domain_df: pd.DataFrame = None,
    sentiment_df: pd.DataFrame = None,
    wordcloud_pil_image=None
) -> bytes:
    """
    Generate a detailed multi-page WhatsApp Group Chat Analysis PDF Report.

    Returns:
        bytes: Raw PDF document bytes ready for download.
    """
    start_time = time.perf_counter()
    logger.info("Generating detailed multi-page PDF Analytics Report for group '%s'", stats.get('group_name', 'Group'))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom ReportLab Paragraph Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1E1B18')
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#D97757')
    )
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1C1B18'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#333333')
    )
    cell_header = ParagraphStyle(
        'CellHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11,
        textColor=colors.whitesmoke
    )
    cell_body = ParagraphStyle(
        'CellBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=11,
        textColor=colors.HexColor('#222222')
    )

    story = []

    # PAGE 1: EXECUTIVE SUMMARY & MEMBER LEADERBOARDS
    group_name = stats.get('group_name', 'WhatsApp Group Chat')
    story.append(Paragraph(f"💬 WhatsApp Group Executive Analytics Report", title_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"<b>Group:</b> {group_name} &nbsp;|&nbsp; <b>Generated:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#D97757'), spaceAfter=12))

    # Executive KPI Summary Grid Table
    story.append(Paragraph("📊 Executive Summary Metrics", heading_style))

    kpi_data = [
        [
            Paragraph("Metric Name", cell_header),
            Paragraph("Value", cell_header),
            Paragraph("Metric Name", cell_header),
            Paragraph("Value", cell_header)
        ],
        [
            Paragraph("Total Messages", cell_body),
            Paragraph(f"<b>{stats.get('total_messages', 0):,}</b>", cell_body),
            Paragraph("Active Participants", cell_body),
            Paragraph(f"<b>{stats.get('total_members', 0):,}</b>", cell_body)
        ],
        [
            Paragraph("Media Shared", cell_body),
            Paragraph(f"<b>{stats.get('media_message', 0):,}</b>", cell_body),
            Paragraph("Shared Web Links", cell_body),
            Paragraph(f"<b>{int(stats.get('link_shared', 0)):,}</b>", cell_body)
        ],
        [
            Paragraph("Total Words Typed", cell_body),
            Paragraph(f"<b>{int(data_frame['word_count'].sum()):,}</b>" if 'word_count' in data_frame.columns else "0", cell_body),
            Paragraph("Total Emojis Sent", cell_body),
            Paragraph(f"<b>{stats.get('total_emojis', 0):,}</b>", cell_body)
        ],
        [
            Paragraph("Messages / Day", cell_body),
            Paragraph(f"<b>{stats.get('avg_msgs_per_day', 0.0)}</b>", cell_body),
            Paragraph("Top Contributor", cell_body),
            Paragraph(f"<b>{data_frame['name'].value_counts().index[0]}</b>" if ('name' in data_frame.columns and not data_frame.empty) else "N/A", cell_body)
        ],
        [
            Paragraph("Busiest Day of Week", cell_body),
            Paragraph(f"<b>{stats.get('peak_day', 'N/A')}</b>", cell_body),
            Paragraph("Peak Hour Window", cell_body),
            Paragraph(f"<b>{stats.get('busiest_hour', 'N/A')}</b>", cell_body)
        ]
    ]

    t_kpi = Table(kpi_data, colWidths=[140, 120, 140, 120])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1C1B18')),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FAF8F5'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 10))

    # Top Member Rankings Table
    story.append(Paragraph("👑 Top Contributor Rankings", heading_style))

    if 'name' in data_frame.columns and not data_frame.empty:
        counts = data_frame['name'].value_counts().head(8)
        total_m = len(data_frame)
        member_table_data = [
            [
                Paragraph("Rank", cell_header),
                Paragraph("Participant Name", cell_header),
                Paragraph("Messages Sent", cell_header),
                Paragraph("Share (%)", cell_header)
            ]
        ]
        for rank, (name, count) in enumerate(counts.items(), start=1):
            pct = round((count / total_m) * 100, 1)
            member_table_data.append([
                Paragraph(f"#{rank}", cell_body),
                Paragraph(str(name), cell_body),
                Paragraph(f"{count:,}", cell_body),
                Paragraph(f"{pct}%", cell_body)
            ])

        t_member = Table(member_table_data, colWidths=[50, 230, 120, 120])
        t_member.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D97757')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#FAF8F5'), colors.white]),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(t_member)

    story.append(Spacer(1, 10))

    # Chart 1: Most Active Member Bar Chart
    fig_member = most_active_member(data_frame)
    img_member = fig_to_rl_image(fig_member, width=520, height=210)
    if img_member:
        story.append(img_member)

    # PAGE 2: ACTIVITY & TEMPORAL ANALYTICS
    story.append(PageBreak())
    story.append(Paragraph("🕒 Activity & Temporal Patterns", heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#D97757'), spaceAfter=10))

    # Chart 2: Busiest Hour Profile Chart
    from processor.transformers.chat_eda import WhatsAppProcess
    analyzer_instance = WhatsAppProcess(None) if False else None
    
    if 'datetime' in data_frame.columns and not data_frame.empty:
        hour_counts = data_frame['datetime'].dt.hour.value_counts()
        profile = pd.DataFrame({'hour': range(24)})
        profile['message_count'] = profile['hour'].map(hour_counts).fillna(0).astype(int)
        fig_hourly = plot_hourly_profile(profile)
        img_hourly = fig_to_rl_image(fig_hourly, width=520, height=210)
        if img_hourly:
            story.append(img_hourly)
            story.append(Spacer(1, 10))

    # Chart 3: Active Day of Week
    fig_day = most_active_day(data_frame)
    img_day = fig_to_rl_image(fig_day, width=520, height=210)
    if img_day:
        story.append(img_day)
        story.append(Spacer(1, 10))

    # Chart 4: Day of Month Activity Breakdown
    fig_month = most_suitable_day(data_frame)
    img_month = fig_to_rl_image(fig_month, width=520, height=210)
    if img_month:
        story.append(img_month)

    # PAGE 3: CONTENT, SENTIMENT & WEB DOMAIN INSIGHTS
    story.append(PageBreak())
    story.append(Paragraph("🔬 Content, Sentiment & Web Domain Analytics", heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#D97757'), spaceAfter=10))

    # Chart 5: Top Shared Web Domains
    if domain_df is not None and not domain_df.empty:
        fig_domain = plot_top_domains(domain_df)
        img_domain = fig_to_rl_image(fig_domain, width=520, height=200)
        if img_domain:
            story.append(img_domain)
            story.append(Spacer(1, 10))

    # Chart 6: Member Sentiment Polarity Chart
    fig_sent = sentiment_analysis(cloud_df)
    img_sent = fig_to_rl_image(fig_sent, width=520, height=200)
    if img_sent:
        story.append(img_sent)
        story.append(Spacer(1, 10))

    # Chart 7: Message Length & Complexity Distribution
    def _get_length_df(df):
        def _bucketize(cnt):
            if cnt <= 3: return '1-3 Words (Short)'
            elif cnt <= 10: return '4-10 Words (Medium)'
            elif cnt <= 25: return '11-25 Words (Long)'
            else: return '25+ Words (Detailed)'
        b = df['word_count'].apply(_bucketize).value_counts().reset_index()
        b.columns = ['Bucket', 'Count']
        return b

    if 'word_count' in data_frame.columns and not data_frame.empty:
        length_df = _get_length_df(data_frame)
        fig_len = plot_message_length_distribution(length_df)
        img_len = fig_to_rl_image(fig_len, width=520, height=200)
        if img_len:
            story.append(img_len)

    # PAGE 4: MEMBER RESPONSE LATENCY & SECURITY GUARANTEE
    story.append(PageBreak())
    story.append(Paragraph("⏱️ Member Reply Latency & Security Guarantee", heading_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#D97757'), spaceAfter=10))

    # Chart 8: Reply Latency Chart
    def _get_reply_latency_df(df):
        if 'datetime' not in df.columns or 'name' not in df.columns or df.empty:
            return pd.DataFrame(columns=['name', 'avg_reply_time_mins'])
        d = df.sort_values('datetime').copy()
        d['prev_name'] = d['name'].shift(1)
        d['prev_time'] = d['datetime'].shift(1)
        replies = d[(d['name'] != d['prev_name']) & d['prev_name'].notna()].copy()
        replies['reply_time_mins'] = (replies['datetime'] - replies['prev_time']).dt.total_seconds() / 60.0
        active_r = replies[replies['reply_time_mins'] <= 120]
        if active_r.empty:
            return pd.DataFrame(columns=['name', 'avg_reply_time_mins'])
        lat = active_r.groupby('name')['reply_time_mins'].mean().round(1).reset_index()
        lat.columns = ['name', 'avg_reply_time_mins']
        return lat.sort_values('avg_reply_time_mins')

    latency_df = _get_reply_latency_df(data_frame)
    if not latency_df.empty:
        fig_lat = plot_reply_latency(latency_df)
        img_lat = fig_to_rl_image(fig_lat, width=520, height=220)
        if img_lat:
            story.append(img_lat)
            story.append(Spacer(1, 14))

    # Security & Privacy Disclaimer Notice
    story.append(Paragraph("🛡️ <b>Data Privacy & Security Guarantee</b>", heading_style))
    story.append(Paragraph("""
    This analytics report was compiled strictly in-memory using a <b>Zero-Data-Retention processing engine</b>.
    <br/><br/>
    • <b>Zero File Storage:</b> Uploaded chat exports (.txt) are processed in transient RAM memory and never saved to disk.
    <br/>
    • <b>Local Privacy:</b> No messages, contact names, or raw text were transmitted to external cloud servers or AI APIs.
    <br/>
    • <b>Open-Source Engine:</b> Complete processing source code is fully transparent and open-source.
    """, body_style))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CCCCCC'), spaceBefore=8, spaceAfter=8))
    story.append(Paragraph("<b>WhatsApp Chat Processor Engine</b> &bull; Developed by Rahul Kumeriya", body_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    elapsed = time.perf_counter() - start_time
    logger.info("Detailed multi-page PDF Analytics Report generated in %.3f sec (%d bytes)", elapsed, len(pdf_bytes))
    return pdf_bytes
