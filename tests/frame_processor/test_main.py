import unittest
from unittest.mock import patch, MagicMock
from src.frame_processor.rtsp_client import RTSPClient
from src.frame_processor.preprocessing import preprocess_frame
import numpy as np

class TestFrameProcessor(unittest.TestCase):

    @patch('cv2.VideoCapture')
    def test_rtsp_client_connect_success(self, mock_video_capture):
        mock_capture_instance = MagicMock()
        mock_capture_instance.isOpened.return_value = True
        mock_video_capture.return_value = mock_capture_instance

        client = RTSPClient("rtsp://dummy.url")
        self.assertTrue(client.connect())

    @patch('cv2.VideoCapture')
    def test_rtsp_client_connect_fail(self, mock_video_capture):
        mock_capture_instance = MagicMock()
        mock_capture_instance.isOpened.return_value = False
        mock_video_capture.return_value = mock_capture_instance

        client = RTSPClient("rtsp://dummy.url")
        self.assertFalse(client.connect())

    def test_preprocess_frame(self):
        dummy_frame = np.zeros((100, 200, 3), dtype=np.uint8)
        target_size = (640, 480)
        preprocessed = preprocess_frame(dummy_frame, target_size)

        self.assertEqual(preprocessed.shape, (1, 480, 640, 3))
        self.assertEqual(preprocessed.dtype, np.float32)

if __name__ == '__main__':
    unittest.main()
