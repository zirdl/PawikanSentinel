import unittest
from src.detection_analyzer.post_processing import post_process_detections
from src.detection_analyzer.object_tracker import ObjectTracker
import numpy as np

class TestDetectionAnalyzer(unittest.TestCase):

    def test_post_process_detections(self):
        dummy_output = {
            'output_1': np.array([
                [0.5, 0.5, 0.1, 0.1, 0.9, 0.1, 0.9],  # Example detection 1 (x,y,w,h,conf,class_prob)
                [0.2, 0.2, 0.3, 0.3, 0.7, 0.8, 0.2]   # Example detection 2
            ], dtype=np.float32).reshape(1, 2, 7) # Reshape to (1, N, 7) to simulate model output
        }
        # Assuming a dummy frame size for testing
        original_frame_width = 640
        original_frame_height = 480
        detections = post_process_detections(dummy_output, 0.8, 0.5, original_frame_width, original_frame_height)
        self.assertEqual(len(detections), 1)

    def test_object_tracker(self):
        tracker = ObjectTracker()
        rects = [(10, 10, 20, 20)]
        objects = tracker.update(rects)
        self.assertEqual(len(objects), 1)

if __name__ == '__main__':
    unittest.main()
