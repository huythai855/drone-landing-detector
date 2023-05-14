Drone landing detector
==========================

### Description
- A simple model to detect the landing zone for drone.
- My team's term-end project for the course **Image Processing INT3404 2** at UET.

### Group number 12
- Members:
    - [20020099 Lê Xuân Dương](https://www.facebook.com/leduongO19)
    - [21020035 Nguyễn Huy Thái](https://www.facebook.com/huythai855/)
    - [20020398 Vũ Văn Hào](https://www.facebook.com/profile.php?id=100006279257590)
    - [20020392 Cao Hải Đăng](https://www.facebook.com/haidang.uet.2203)
    - [20020041 Nguyễn Văn Khánh Duy](https://www.facebook.com/duytuoiit)

### How to run
- Clone the repository:
    ```bash
    git clone https://github.com/huythai855/drone-landing-detector
    ```
  
- Get into the folder:
    ```bash
    cd drone-landing-detector
    ```
- Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

- Copy folder `test_images` and file `labels.csv` to the root folder of the project (as the example). ⚠️ **PLEASE NOTE** that the folder structure should be:
  ```
  (root)
    │
    ├── README.md
    ├── requirements.txt
    │
    ├── models
    │   ├── trained
    │   |   └── best.pt                <- The trained model
    │   └── common.py
    │   └── experimental.py
    │   └── yolo.py
    │
    ├── segment
    │   └── predict.py
    │
    ├── utils/...
    │
    ├── test_images                     <- Images folder
    │   └── img_example_001.jpg        
    │   └── img_example_002.jpg
    │
    ├── labels.csv                      <- Label of the images
    │
    ├── detect.py
    ├── hubconf.py
    ├── trained.py
    ├── main.py
    ├── landing_detector.py             <- Load trained model and predict
    ├── test.py                         <- Scoring file
    │
    └── yolov5s.pt                      <- The pretrained model
    ```
 
- Run the scoring program:
    ```bash
    python python test.py test_images labels.csv
    ```

### Demo
<img src="resources/images/img_119.jpg" width="447" height="448">

- Demo result's scoring:
    ```
    Python 3.10.9 (main, Mar  1 2023, 18:23:06) [GCC 11.2.0] on linux
    Run time in: 0.00 s
    Total test images:  5
    filename:  img_train_593.jpg
    Fusing layers... 
    Adding AutoShape... 
    {'x1': 239, 'y1': 300, 'x2': 381, 'y2': 420} 
    
    filename:  img_train_451.jpg
    Fusing layers... 
    Adding AutoShape... 
    {'x1': 330, 'y1': 370, 'x2': 424, 'y2': 478} 
    
    filename:  img_train_330.jpg
    Fusing layers... 
    Adding AutoShape... 
    {'x1': 284, 'y1': 238, 'x2': 370, 'y2': 264} 
    
    filename:  img_train_156.jpg
    Fusing layers... 
    Adding AutoShape... 
    {'x1': 256, 'y1': 270, 'x2': 508, 'y2': 414} 
    
    filename:  img_train_500.jpg
    Fusing layers... 
    Adding AutoShape... 
    {'x1': 212, 'y1': 282, 'x2': 242, 'y2': 382} 
    
    [0.9501630181648812, 0.9255329318420801, 0.8578161822466615, 0.9271523178807947, 0.8490566037735849]
    Map score: 0.860000
    Run time:  1.7516562938690186
    ```