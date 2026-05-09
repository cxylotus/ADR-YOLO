import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/path/to/your/weights/best.pt')
    result = model.val( 
                        # data='ultralytics/cfg/dataset/DIOR.yaml',
                        # data='ultralytics/cfg/dataset/TinyPerson.yaml',
                        data='ultralytics/cfg/dataset/VisDrone-DET.yaml',
                        split='test', 
                        imgsz=640, # 640 / only 1280 for TinyPerson 
                        batch=16,
                        conf=0.001,
                        device=0,
                        workers=2,
                        # save_json=True, # if you need to cal coco metrice
                        project='work_dirs',
                        name='ADR_yolo_val_',
                        )
    