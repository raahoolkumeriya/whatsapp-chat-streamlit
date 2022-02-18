
import unittest
import yaml
from processor.transformers.chat_eda import WhatsAppProcess, process_data
from unittest import mock


class TestCharts(unittest.TestCase):
    def setUp(self):
        with open('configs/demo_chat.txt', 'r') as read_file:
            self.data = read_file.read()
        # Parsing YAML file
        config = 'configs/app_configuration.yml'
        config = yaml.safe_load(open(config))
        # configure logging
        whatsapp_config = config['whatsapp']
        self.whatsapp = WhatsAppProcess(whatsapp_config)
        self.message = self.whatsapp.apply_regex(self.data)
        self.raw_df = process_data(self.message)
        self.data_frame = self.whatsapp.get_dataframe(self.raw_df)
        self.dataF = self.whatsapp.day_analysis(self.data_frame)

    @mock.patch("processor.graphs.charts.max_words_used")
    def test_charts_max_words(self, mock_plot):
        mock_plot.max_words_used(self.data_frame)
        assert mock_plot

    @mock.patch("processor.graphs.charts.message_cluster")
    def test_charts_message_cluster(self, mock_plot):
        mock_plot.message_cluster(self.data_frame)
        assert mock_plot

    @mock.patch("processor.graphs.charts.pie_display_emojis")
    def test_charts_pie_display_emojis(self, mock_plot):
        mock_plot.pie_display_emojis(self.data_frame)
        assert mock_plot

    @mock.patch("processor.graphs.charts.time_series_plot")
    def test_charts_time_series_plot(self, mock_plot):
        mock_plot.time_series_plot(self.data_frame)
        assert mock_plot

    @mock.patch("processor.graphs.charts.plot_data")
    def test_charts_plot_data(self, mock_plot):
        data_string = {
            'x_value': 4,
            'y_value': [1, 2, 3, 4],
            'tick_label': ['A', 'B', 'C', 'D'],
            'x_label': 'Name of Group Member',
            'y_label': 'Number of Group Messages',
            'title': 'Mostly Active member in Group (based on messages)'
        }
        mock_plot.plot_data(data_string)
        assert mock_plot.figure.show
