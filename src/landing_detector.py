import yolov5


class LandingDetector:
    def __init__(self):
        self.name = 'LandingDetector'

    def detect(self, img):
        # load model
        model = yolov5.load('./models/trained/best.pt')

        # inference
        results = model(img)

        # parse results
        predictions = results.pred[0]
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        # x1 = float(boxes[0][0])
        # y1 = float(boxes[0][1])
        # x2 = float(boxes[0][2])
        # y2 = float(boxes[0][3])

        x1 = int(boxes[0][0])
        y1 = int(boxes[0][1])
        x2 = int(boxes[0][2])
        y2 = int(boxes[0][3])

        return x1, y1, x2, y2
