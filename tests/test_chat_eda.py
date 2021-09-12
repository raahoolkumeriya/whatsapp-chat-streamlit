
import unittest
from chat_eda import WhatsAppProcess, process_data, statistics, extract_emojis, sorted_authors_df


class TestWhatsAppProcess(unittest.TestCase):
    def setUp(self):
        with open('demo_chat.txt', 'r') as read_file:
            self.data = read_file.read()
        self.whatsapp = WhatsAppProcess()
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
        assert str(type(self.raw_df)) == "<class 'pandas.core.frame.DataFrame'>",\
            "Return response is not Pandas DataFrame"

    def test_get_day_analysis_dataframe(self):
        assert str(type(self.whatsapp.day_analysis(self.data_frame))) ==\
            "<class 'pandas.core.frame.DataFrame'>", "Return response is not Pandas DataFrame"

    def test_get_cloud_dataframe(self):
        assert str(type(self.whatsapp.cloud_data(self.raw_df))) ==\
            "<class 'pandas.core.frame.DataFrame'>", "Return response is not Pandas DataFrame"

    def test_get_unique_member(self):
        assert self.data_frame.size == 352, "DataFrame size is not Matched"
    
    def test_statistics_data(self):
        data_string = {
            'group_name': 'MAD MAX Fury Road 🛻',
            'link_shared': 1,
            'media_message': 0,
            'total_deleted_messages': 0,
            'total_members': 9,
            'total_messages': 32,
            'your_deleted_message': 0
        }
        assert self.stats == data_string, "Statistics return from DataFrame analysis"

    def test_emojis_function(self):
        assert extract_emojis(self.data[-20:]) == "💀🛻🔥", "Emojis extraction failed"

    def test_sorted_authors_df(self):
        list_member = [
            'Nux', 'Slit', 'Imperator Furiosa', 'War Boys', 'Max Rockatansky',
            'Morsov', 'The Rock Rider Chief', 'War Boy',
            'MAD MAX Fury Road 🛻']
        assert sorted_authors_df(self.data_frame).any() in list_member,\
            "Member does not found in List"