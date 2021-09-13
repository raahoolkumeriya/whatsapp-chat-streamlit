import time
from seleniumbase import BaseCase
import cv2


class ComponentsTest(BaseCase):
    def test_basic(self):

        # open the app and take a screenshot
        self.open(
            "https://share.streamlit.io/raahoolkumeriya/\
                whatsapp-chat-streamlit/main/app.py")

        time.sleep(10)  # give leaflet time to load from web
        self.save_screenshot("current-screenshot.png")

        self.check_window(name="first_test", baseline=True)

        # self.assert_text("WhatsApp Chat Processor")

        # test screenshots look exactly the same
        original = cv2.imread(
            "visual_baseline/test_basic/first_test/screenshot.png")
        duplicate = cv2.imread("current-screenshot.png")

        assert original.shape == duplicate.shape

        difference = cv2.subtract(original, duplicate)
        b, g, r = cv2.split(difference)
        assert cv2.countNonZero(b) == \
            cv2.countNonZero(g) == cv2.countNonZero(r) == 0
