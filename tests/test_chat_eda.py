import unittest
import yaml
import pandas as pd
from processor.transformers.chat_eda import WhatsAppProcess, WhatsAppAnalyzer, process_data,\
    statistics, extract_emojis, sorted_authors_df, WhatsAppConfig


class TestWhatsAppProcess(unittest.TestCase):
    def setUp(self):
        with open('configs/demo_chat.txt', 'r') as read_file:
            self.data = read_file.read()
        # Parsing YAML file
        config = 'configs/app_configuration.yml'
        config = yaml.safe_load(open(config))
        # reading source configuration
        whatsapp_config = WhatsAppConfig(**config['whatsapp'])
        self.whatsapp = WhatsAppProcess(whatsapp_config)
        self.message = self.whatsapp.apply_regex(self.data)
        self.raw_df = process_data(self.message)
        self.data_frame = self.whatsapp.get_dataframe(self.raw_df)
        self.stats = statistics(self.raw_df, self.data_frame)

    def test_regex_on_loaded_chat(self):
        assert self.message != [], "REGEX list Does NOT matched"

    def test_value_exists_in_varible(self):
        assert "Missed video call" in self.whatsapp.ignore,\
            "Message does not found in Ignore list"

    def test_get_dataframe(self):
        assert str(type(self.raw_df)) ==\
            "<class 'pandas.core.frame.DataFrame'>",\
            "Return response is not Pandas DataFrame"

    def test_get_day_analysis_dataframe(self):
        df = self.whatsapp.day_analysis(self.data_frame)
        assert str(type(df)) == "<class 'pandas.core.frame.DataFrame'>"
        assert "word_count" in df.columns, "word_count column missing after day_analysis"

    def test_get_cloud_dataframe(self):
        assert str(type(self.whatsapp.cloud_data(self.raw_df))) ==\
            "<class 'pandas.core.frame.DataFrame'>",\
            "Return response is not Pandas DataFrame"

    def test_get_unique_member(self):
        assert self.data_frame.size == 363,\
            "DataFrame size is not Matched"

    def test_statistics_data(self):
        expected_keys = {
            'group_name', 'link_shared', 'media_message', 'total_deleted_messages',
            'total_members', 'total_messages', 'your_deleted_message', 'total_emojis',
            'avg_msgs_per_day', 'peak_day', 'busiest_hour'
        }
        assert set(self.stats.keys()) == expected_keys,\
            "Statistics keys do not match expected metrics"
        assert self.stats['group_name'] == 'MAD MAX Fury Road 🛻'
        assert self.stats['total_messages'] == 33
        assert self.stats['total_members'] == 9
        assert self.stats['total_emojis'] == 31

    def test_emojis_function(self):
        assert extract_emojis(self.data[-19:]) == "💀🛻🔥",\
            "Emojis extraction failed"

    def test_sorted_authors_df(self):
        list_member = [
            'Nux', 'Slit', 'Imperator Furiosa', 'War Boys', 'Max Rockatansky',
            'Morsov', 'The Rock Rider Chief', 'War Boy',
            'MAD MAX Fury Road 🛻']
        assert sorted_authors_df(self.data_frame).any() in list_member,\
            "Member does not found in List"

    def test_whatsapp_analyzer_alias(self):
        analyzer = WhatsAppAnalyzer(self.whatsapp.app_config)
        assert analyzer is not None
        matches = analyzer.apply_regex(self.data)
        assert len(matches) > 0

    def test_hindi_marathi_multilingual_parsing(self):
        hindi_marathi_data = """25/07/24, 2:30 pm - राहुल: नमस्कार कसा आहेस?
२५/०७/२४, २:३० PM - अमित: मैं ठीक हूँ, आप कैसे हैं?
25.07.24, 2:30 अपराह्न - सुप्रिया: हाँ जरूर मिलेंगे"""
        matches = self.whatsapp.apply_regex(hindi_marathi_data)
        assert len(matches) == 3, f"Expected 3 multilingual matches, got {len(matches)}"
        df = process_data(matches)
        assert len(df) == 3
        assert df['name'].tolist() == ['राहुल', 'अमित', 'सुप्रिया']
        assert df['datetime'].iloc[0] == pd.Timestamp('2024-07-25 14:30:00')

    def test_hindi_marathi_wordcloud_extraction(self):
        matches = [
            ('25/07/24, 2:30 pm', 'राहुल', 'नमस्कार कसा आहेस?'),
            ('25/07/24, 2:31 pm', 'अमित', 'मैं ठीक हूँ, आप कैसे हैं?')
        ]
        raw_df = pd.DataFrame(matches, columns=['datetime', 'name', 'message'])
        cloud_df = self.whatsapp.cloud_data(raw_df)
        assert 'नमस्कार' in cloud_df['message'].iloc[0]
        assert 'ठीक' in cloud_df['message'].iloc[1]

    def test_new_analytics_transformers(self):
        # Top domains test
        domains_df = self.whatsapp.extract_top_domains(self.data_frame)
        assert isinstance(domains_df, pd.DataFrame)
        
        # Reply latency test
        latency_df = self.whatsapp.calculate_reply_latency(self.data_frame)
        assert isinstance(latency_df, pd.DataFrame)

        # Hourly activity profile test
        hourly_df = self.whatsapp.hourly_activity_profile(self.data_frame)
        assert len(hourly_df) == 24

        # Sentiment analysis test
        sentiment_df = self.whatsapp.analyze_message_sentiment(self.data_frame)
        assert 'Sentiment' in sentiment_df.columns

        # Message length distribution test
        length_df = self.whatsapp.message_length_distribution(self.data_frame)
        assert isinstance(length_df, pd.DataFrame)

    def test_generate_pdf_report(self):
        from processor.reports.pdf_generator import generate_pdf_report
        cloud_df = self.whatsapp.cloud_data(self.raw_df)
        pdf_bytes = generate_pdf_report(self.stats, self.data_frame, cloud_df)
        assert isinstance(pdf_bytes, bytes)
        assert len(pdf_bytes) > 500
        assert pdf_bytes.startswith(b'%PDF')
