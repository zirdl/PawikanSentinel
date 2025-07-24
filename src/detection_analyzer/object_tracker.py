from collections import OrderedDict
import numpy as np

def calculate_iou(boxA, boxB):
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

class ObjectTracker:
    """A simple object tracker that assigns unique IDs to detected objects."""

    def __init__(self, max_disappeared=50):
        """Initializes the object tracker."""
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        """Registers a new object with the next available object ID."""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Deregisters an object by its ID."""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """Updates the tracker with a new set of bounding boxes."""
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.zeros((len(object_centroids), len(input_centroids)))

            for i in range(0, len(object_centroids)):
                for j in range(0, len(input_centroids)):
                    D[i, j] = np.linalg.norm(object_centroids[i] - input_centroids[j])

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects

if __name__ == '__main__':
    # --- Example Usage ---

    # 1. Initialize the object tracker
    tracker = ObjectTracker(max_disappeared=5)

    # 2. Simulate a few frames of detections
    frames_detections = [
        [(10, 10, 50, 50), (100, 100, 150, 150)],  # Frame 1
        [(12, 12, 52, 52), (105, 105, 155, 155)],  # Frame 2
        [(108, 108, 158, 158)],                    # Frame 3 (one object disappears)
        [],                                        # Frame 4 (all objects disappear)
        [(200, 200, 250, 250)]                     # Frame 5 (new object appears)
    ]

    for i, rects in enumerate(frames_detections):
        objects = tracker.update(rects)
        print(f"Frame {i+1} Objects: {objects}")
