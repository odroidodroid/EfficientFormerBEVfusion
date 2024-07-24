## Version Info
torch : 1.13.0 \\
numpy : 1.23.4
mmcv : 1.4.0 (MMCV\_WITH\_OPS=1)
mmengine : 0.9.0
mmsegmentation : 0.12.0
mmdet : 2.20.0

## Run
### run test.py
	torchpack dist-run -np 1 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/.../convfuser.yaml checkpoints.pth --eval bbox

### run train.py
	torchpack dist-run -np 1 python tools/test.py configs/nuscenes/det/transfusion/secfpn/camera+lidar/.../convfuser.yaml --load\_from pretrained/lidar-only-det.pth

## Added
### ~2023.12
ResNet18
ResNet18-Depthwise
ResNet50-Depthwise
Distillation
### 2024.01~2024.06
IdentityFormer (Transfusion Head)
MetaFormer
InternImage
DeformPool
DSVT
### 2024.07~
LatentRendering
SimpleBEV
