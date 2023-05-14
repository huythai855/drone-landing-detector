import torch


class LandingDetector:
    def __init__(self):
        self.name = 'LandingDetector'

    def detect(self, img):
        model = torch.hub.load('.', 'custom',
                               path='models/trained/best.pt',
                               source='local',
                               force_reload=True,
                               trust_repo=True)

        # inference
        results = model(img)

        # parse results
        final_res = results.pandas().xyxy[0]
        x1 = int(final_res.xmin[0])
        y1 = int(final_res.ymin[0])
        x2 = int(final_res.xmax[0])
        y2 = int(final_res.ymax[0])

        return x1, y1, x2, y2
