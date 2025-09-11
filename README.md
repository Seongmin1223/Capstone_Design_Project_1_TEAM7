
<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">광시야각 카메라 기반 운전자·승객 모니터링 (Driver & Passenger Monitoring)</h3>

  <p align="center">
    엣지 디바이스 기반의 DMS(Driver Monitoring System) 프로토타입
    <br />
    <a href="https://img1.daumcdn.net/thumb/R1280x0/?fname=http://t1.daumcdn.net/brunch/service/user/cjBn/image/e1qRbB15Y382IJVAbDn_7pADI1w.jpg"><strong>문서 살펴보기 »</strong></a>
    <br /><br />
    <a href="https://img1.daumcdn.net/thumb/R1280x0/?fname=http://t1.daumcdn.net/brunch/service/user/cjBn/image/e1qRbB15Y382IJVAbDn_7pADI1w.jpg">데모 보기</a>
    &middot;
    <a href="https://img1.daumcdn.net/thumb/R1280x0/?fname=http://t1.daumcdn.net/brunch/service/user/cjBn/image/e1qRbB15Y382IJVAbDn_7pADI1w.jpg">버그 제보</a>
    &middot;
    <a href="https://img1.daumcdn.net/thumb/R1280x0/?fname=http://t1.daumcdn.net/brunch/service/user/cjBn/image/e1qRbB15Y382IJVAbDn_7pADI1w.jpg">기능 요청</a>
  </p>
</div>

<!-- CONTRIBUTORS -->
## 👨‍💻 Contributors

<div align="center">
  <a href="https://github.com/namjin1231" style="display:inline-block; margin: 10px;">
    <img src="https://avatars.githubusercontent.com/u/203584270?v=4" width="80px;" alt="namjin1231"/>
  </a>
  <a href="https://github.com/imsh1127" style="display:inline-block; margin: 10px;">
    <img src="https://avatars.githubusercontent.com/u/125844849?v=4" width="80px;" alt="imsh1127"/>
  </a>
  <a href="https://github.com/Sumin020726" style="display:inline-block; margin: 10px;">
    <img src="https://avatars.githubusercontent.com/u/162936275?v=4" width="80px;" alt="Sumin020726"/>
  </a>
  <a href="https://github.com/yoonhoc" style="display:inline-block; margin: 10px;">
    <img src="https://avatars.githubusercontent.com/u/144187814?v=4" width="80px;" alt="yoonhoc"/>
  </a>
  <a href="https://github.com/Seongmin1223" style="display:inline-block; margin: 10px;">
    <img src="https://avatars.githubusercontent.com/u/165881011?v=4" width="80px;" alt="Seongmin1223"/>
  </a>
</div>

<p align="center">엣지 디바이스 기반의 DMS(Driver Monitoring System) 프로토타입</p>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>목차</summary>
  <ol>
    <li><a href="#개요">개요</a></li>
    <li><a href="#주요-기능">주요 기능</a></li>
    <li><a href="#사용-기술-스택">사용 기술 스택</a></li>
    <li><a href="#구조">구조</a></li>
   
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## 개요

하나의 **광시야각(와이드) 카메라**를 통해 운전자와 조수석 승객을 동시에 모니터링하고, **OpenCV 기반 얼굴·눈·핸드 분석**과 **YOLO 기반 객체 감지**를 결합하여  

- 졸음  
- 휴대폰 사용  
- 안전벨트 미착용  
- 부적절한 자세  
- Hands-On-Detection(HOD)  

등을 실시간으로 판정하여 경고를 발생시키는 시스템입니다.  

**목표**: 차내 엣지 디바이스에서 실시간(on-device)으로 동작하면서, 프라이버시를 지키는 DMS 프로토타입 구현.  

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 주요 기능

- 광시야각 카메라 입력 (주야간 대응: IR 보조 가능)  
- 영상 전처리 (렌즈 보정, 노이즈 제거, 정규화)  
- OpenCV 기반 얼굴/랜드마크/눈(blink, EAR) 분석  
- Head pose(머리 각도) 및 gaze(시선) 추정  
- Hand / HOD (핸들 파지) 분석  
- YOLO 기반 객체 감지 (스마트폰, 안전벨트 등 위험 객체)  
- 시계열 융합(슬라이딩 윈도우 기반 규칙 + ML)으로 행동 분류  
- 경고 출력: 음성/버저/HUD/CAN 이벤트 로그  
- 엣지 최적화 (ONNX → TensorRT / OpenVINO 변환)  

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 사용 기술 스택

- **운영체제**: Linux  
- **라이브러리 / 프레임워크**: OpenCV, PyTorch, YOLOv5/YOLOv8  
- **하드웨어 권장**: NVIDIA Jetson 시리즈, Raspberry Pi + NCS2 (성능 제한 주의)  

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 구조

```mermaid
flowchart TD
    Camera["Camera Input"]
    Pre["Video Preprocessing"]
    Face["Face/Eye/Hand Analysis"]
    YOLO["YOLO Object Detection"]
    Fusion["Time Series Fusion"]
    Alert["Alert Output"]

    Camera --> Pre --> Face
    Pre --> YOLO
    Face --> Fusion
    YOLO --> Fusion
    Fusion --> Alert

