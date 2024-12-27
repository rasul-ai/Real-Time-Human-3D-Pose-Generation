# Real-Time-Human-3D-Pose-Generation
I have created these 3 python files to capture live video from webcam and proccess at the same time. These code will open 2 window where you can see the original video frame with 2d pose and 3d pose seperately. Well these codes can not be run independently.

I have inspired from the official PyTorch implementation of the paper "[MotionAGFormer: Enhancing 3D Human Pose Estimation With a Transformer-GCNFormer Network](https://openaccess.thecvf.com/content/WACV2024/html/Mehraban_MotionAGFormer_Enhancing_3D_Human_Pose_Estimation_With_a_Transformer-GCNFormer_Network_WACV_2024_paper.html)" (WACV 2024) and used the [github repository](https://github.com/TaatiTeam/MotionAGFormer) for associative files.

To run these 3 codes, download [github repository](https://github.com/TaatiTeam/MotionAGFormer) repository and
```
1. put live.py and main.py in ./demo/ folder.
2. put the helper.py in ./demo/lib/hrnet/ folder.
3. setup environment using requirements.txt.
```

## Environment
For installation of the project dependencies, please run:
```
pip install -r requirements.txt
``` 

## Downloading Pretrained models
First, download the YOLOv3 and HRNet pretrained models [here](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) and put it in the './demo/lib/checkpoint' directory. Next, download the base model checkpoint from [here](https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view) and put it in the './checkpoint' directory.

Run the command below:
```
python demo/live.py
```