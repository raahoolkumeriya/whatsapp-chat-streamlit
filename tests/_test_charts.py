
import unittest
from chat_eda import WhatsAppProcess, process_data
from charts import message_cluster, plot_data, max_words_used
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt



class TestCharts(unittest.TestCase):
    def setUp(self):
        with open('demo_chat.txt', 'r') as read_file:
            self.data = read_file.read()
        self.whatsapp = WhatsAppProcess()
        self.message = self.whatsapp.apply_regex(self.data)
        self.raw_df = process_data(self.message)
        self.data_frame = self.whatsapp.get_dataframe(self.raw_df)
        self.dataF = self.whatsapp.day_analysis(self.data_frame)


    @image_comparison(baseline_images=['spines_axes_positions'])
    def test_spines_axes_positions(self):
        max_words_used(self.dataF)