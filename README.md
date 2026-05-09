# ADR-YOLO
ADR-YOLO is an adaptive dual-refinement network for object detection in aerial images.

## Installation 
1. Create conda env
```bash
conda create -n adr python=3.10.13 -y
conda activate adr
```
2. Install pytorch and torchvision
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
```
3. Install ADR-YOLO from source
```bash
git clone https://github.com/cxylotus/ADR-YOLO.git
cd ADR-YOLO
pip install -e .
```
4. Install other dependencies
```bash
pip install thop opencv-python==4.9.0.80 numpy==1.26.4
```
## Acknowledgements
This project is based on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).

## Citation
If you find our work helpful for your research, please consider citing:

```bibtex
@ARTICLE{11479830,
  author={Chen, Xiangyu and Meng, Fangzhou and Hong, Aoping and Tong, Guanjun},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={ADR-YOLO: An Adaptive Dual-Refinement Network for Object Detection in Aerial Images}, 
  year={2026},
  volume={19},
  number={},
  pages={13862-13876},
  doi={10.1109/JSTARS.2026.3683156}}
```

