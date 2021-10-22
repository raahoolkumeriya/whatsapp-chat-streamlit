
import unittest
import processor.common.configure as configure

HIDE_STREAMLIT_STYLE = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
footer:after {
    content:'copyrights © 2021 rahul.kumeriya@outlook.com\
        [ DONT FORGET TO PLANT TREES ]' ;
    visibility: visible;
    display: block;
    position: fixed;
    #background-color: red;
    padding: 1px;
    bottom: 0;
}
</style>"""


class TestConfigure(unittest.TestCase):

    def test_configurational_variable_title(self):
        assert configure.TITLE == "WhatsApp Chat Processor",\
            "TITLE is not set in configuration file"

    def test_configurational_variable_Repo_url(self):
        assert configure.REPO_URL == \
            "https://ghbtns.com/github-btn.html?user=raahoolkumeriya&repo=whatsapp-chat-streamlit",\
            "REPO_URL is not set in configuration file"

    def test_configurational_variable_format_button(self):
        assert configure.FORMAT_BUTTON ==\
            'frameborder="0" scrolling="0" width="170" height="30" title="GitHub"',\
            "FORMAT_BUTTON is not set in configuration file"

    def test_configurational_variable_hide_streamlit_style(self):
        assert configure.HIDE_STREAMLIT_STYLE == HIDE_STREAMLIT_STYLE,\
            "HIDE_STREAMLIT_STYLE is not set in configuration file"
