import yolov5
from yolov5 import train, val

EPOCHS = 100
BATCH_SIZE = 32

model = yolov5.load('yolov5s.pt')

train.run(imgsz=640,
          data='./data/data.yaml',
          epochs=EPOCHS,
          batch_size=BATCH_SIZE
          )

val.run(imgsz=640,
        data='../data/data.yaml',
        weights='../models/trained/best.pt'
        )
