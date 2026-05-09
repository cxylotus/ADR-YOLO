import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch

if __name__ == '__main__':
    torch.cuda.empty_cache()

    model = YOLO('ultralytics/cfg/models/ADR-YOLO.yaml')
    model.train(
                # data='ultralytics/cfg/dataset/DIOR.yaml',
                # data='ultralytics/cfg/dataset/TinyPerson.yaml',
                data='ultralytics/cfg/dataset/VisDrone-DET.yaml',
                # cache=False,
                imgsz=640, # 640 / 1280 for TinyPerson
                epochs=300, 
                batch=16,
                close_mosaic=0, 
                workers=2, 
                device=[0,1,2,3],
                optimizer='SGD',
                lr0=0.01, # 0.01 / 0.02 for TinyPerson 
                patience=0, # set 0 to close earlystop.
                # resume=True,
                amp=False, # close amp 
                project='work_dirs',
                name='ADR_yolo',
                exist_ok=False,
                seed=42
                )
