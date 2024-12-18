# YOLOv5 Object Detection

## 실험 목적
YOLOv5를 사용하여 YOLO의 한계점으로 언급된 작은 객체 탐지와 공간적 제약 문제를 실험했습니다.

## 🖥️ **실험환경**
* ![COLAB](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)

## 📝 **실험일지**
* [LINK](https://so-fast.tistory.com/entry/YOLO%EC%9D%98-%ED%95%9C%EA%B3%84%EC%A0%90-%EC%8B%A4%ED%97%98-%EB%B0%8F-%EB%B6%84%EC%84%9D-%EC%9E%91%EC%9D%80%EA%B0%9D%EC%B2%B4%ED%83%90%EC%A7%80%EC%99%80-%EA%B3%B5%EA%B0%84%EC%A0%81-%EC%A0%9C%EC%95%BD)
---

1.서론 (실험 배경 · 목적)


2.실험 목표와 한계점 정의


3.실험 과정

- 3.1 작은 객체 탐지 실험
  
   ![num1](https://github.com/ruru-kor/YOLO_Limitations_Test/blob/main/result/01.png)


- 3.2 공간적 제약 실험

   ![num2](https://github.com/ruru-kor/YOLO_Limitations_Test/blob/main/result/03.png)

- 3.3 해상도 변경 실험

   ![num3](https://github.com/ruru-kor/YOLO_Limitations_Test/blob/main/result/04.png)
  
- 3.4 YOLOv5로 개선 가능성 실험

   ![num3](https://github.com/ruru-kor/YOLO_Limitations_Test/blob/main/result/05.png)
  
4.결론 및 개선 방안


5.참고 코드 및 데이터셋

```
# PyTorch와 torchvision 설치 (필요한 경우)
!pip install torch torchvision

# YOLOv5 설치
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt

# torch.hub로 YOLOv5 모델을 로드
import torch
from google.colab import files

# YOLOv5 사전 학습된 모델 불러오기 ('yolov5s'는 소형 모델)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s', 'yolov5m', 'yolov5l' 등 선택 가능

# 이미지 업로드
uploaded = files.upload()

# 업로드한 이미지 이름 가져오기
image_path = list(uploaded.keys())[0]  # 업로드된 첫 번째 파일 경로
print(f"Uploaded image path: {image_path}")

# YOLOv5 탐지 실행 (torch.hub 방식)
results = model(image_path)

# 탐지 결과 출력 및 표시
results.print()  # 탐지된 클래스 정보 출력
results.show()   # 탐지 결과 이미지 표시

# 탐지 결과를 저장
results.save()  # 저장된 결과는 runs/detect/exp/ 폴더에 위치

# detect.py로 동일한 이미지를 탐지하고 결과 비교
!python detect.py --source "{image_path}" --weights "yolov5s.pt" --conf-thres 0.1 --iou_thres 0.6 --img-size 640
```

## ☑ 코드 실행 방법
1. Google Colab 환경에서 실행
2. 위 코드 복사 후 각 블록별로 실행
3. 이미지 업로드 후 결과 확인
   
   ┖이미지는 원하는걸로 해도 됩니다.

   

## 📌 주요 하이퍼파라미터
- conf_thres
- iou_thres
- img_size: 640
