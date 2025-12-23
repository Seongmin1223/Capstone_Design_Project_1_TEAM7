
<a id="readme-top"></a>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h1 align="center">광시야각 통합 카메라를 이용한 운전자/조수석 승객 모니터링 개발 및 연구 </h1>
  <h2 align="center">In-Vehicle Smartphone Use Detection System Using Gaze Tracking and Objection Detection</h2>

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
**YOLO-based Object Detection + Gaze Tracking for Driver Monitoring**

하나의 **광시야각(와이드) 카메라**를 통해 운전자와 조수석 승객을 동시에 모니터링하고, **OpenCV 기반 얼굴·눈·핸드 분석**과 **YOLO 기반 객체 감지**를 결합하여 휴대폰 사용을 실시간으로 판정하여 경고를 발생시키는 시스템입니다.  

**목표**: 차내 엣지 디바이스에서 실시간(on-device)으로 동작하면서, 프라이버시를 지키는 DMS 프로토타입 구현

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 📖 Background & Motivation

운전 중 스마트폰 사용은 전방 주시율 저하, 반응 시간 증가, 인지 부하 상승을 유발하며
교통사고의 주요 원인으로 지속적으로 보고되고 있다.
기존의 운전자 모니터링 시스템(DMS) 연구들은 다음과 같은 한계를 가진다.

- 휴대폰 객체 탐지만 수행 → 실제 응시 여부 판단 불가
- 시선 방향만 분석 → 스마트폰 사용 맥락 파악 어려움
- 단일 신호 기반 판단 → 오탐지 및 판단 누락 발생

본 연구는 이러한 한계를 보완하기 위해
**객체 정보(휴대폰 위치)** 와 **행동 정보(시선 방향)** 를 결합한
통합 분석 구조를 제안한다.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🎯 Project Objectives

- 단일 카메라 기반 실시간 운전자 모니터링 시스템 설계

- YOLO 기반 스마트폰 객체 탐지

- OpenFace 기반 3D 시선 벡터 추정

- 시선–객체 교차 분석을 통한 스마트폰 응시 판정

- 실제 차량 환경에서의 적용 가능성 검증

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 주요 기능

- 광시야각 카메라 입력 (주야간 대응: IR 보조 가능)  
- 영상 전처리 (렌즈 보정, 노이즈 제거, 정규화)  
- OpenCV 기반 얼굴/랜드마크/눈(blink, EAR) 분석  
- Head pose(머리 각도) 및 gaze(시선) 추정  
- YOLO 기반 객체 감지 (스마트폰, 안전벨트 등 위험 객체)  
- 경고 출력: 음성/버저/HUD/CAN 이벤트 로그
<p align="right">(<a href="#readme-top">back to top</a>)</p>


## 사용 기술 스택

- **운영체제**: Linux  
- **라이브러리 / 프레임워크**: OpenCV, PyTorch, YOLOv5/YOLOv8  
- **하드웨어 권장**: NVIDIA Jetson 시리즈, TI사의 J722SXH01 Evaluation 보드

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 권장 사항
- **환경설정**: 해당 Git에는 용량 문제로 인하여 라이브러리 파일이 직접적으로 업로드되지 않았습니다. 밑에 있는 Google Drive 링크로 접속하여 라이브러리 파일들을 다운로드 받아주시고 사용해주시면 감사하겠습니다.

https://drive.google.com/file/d/1ooFo7wtsK4Vr0c1kjTJtsyJjEHSNztRR/view?usp=sharing

## 구조

<p align="center">
  <strong>광시야각 카메라 기반 DMS 전체 시스템 구조도</strong>
</p>

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/ff92ae9b-f3d6-42c4-9bbd-07761b3af22f"
    alt="DMS System Architecture"
    width="621"
  />
</p>



<p align="right">(<a href="#readme-top">back to top</a>)</p>

### 1️. Smartphone Object Detection

- YOLOv8n 모델 기반 스마트폰 탐지

- 손–휴대폰 겹침 문제를 고려한 데이터 구성

- 휴대폰 클래스 우선 후처리(NMS 조정)

- Bounding Box 결과를 시선 분석 단계로 전달

### 2️. Gaze Estimation

- OpenFace를 이용한 얼굴 랜드마크 및 Head Pose 추정

- 좌·우 눈 시선 벡터 계산 후 평균화

- 동공 기반 3D 중심점 보정 및 프레임 간 smoothing 적용

- 시선 벡터의 진동(jitter) 문제 완화

### 3️. Cross-Detection Logic

운전자가 휴대폰을 실제로 응시하고 있는지 판단하기 위해
다음 기준을 결합하여 판정한다.

- 시선 선분과 휴대폰 Bounding Box의 직접 교차 여부

- Bounding Box 확장 영역 통과 여부

- 시선 벡터와 휴대폰 중심 벡터 간 각도 기반 판정

-> 단순 소지가 아닌 실질적인 주의 분산 행동만을 위험 행동으로 분류


  <p align="right">(<a href="#readme-top">back to top</a>)</p>


## Dataset & Experimental Setup

- 실제 차량 내부에 카메라를 부착하여 데이터 직접 수집

- 다양한 시선 방향, 스마트폰 사용 자세, 조명 환경 포함

- CPU 환경(Intel i5-1335U, GPU 미사용)에서 실험 수행

- 평균 23–25 FPS 실시간 처리 성능 확보

- 스마트폰 응시 판단 정확도 92.3% 달성

  <p align="right">(<a href="#readme-top">back to top</a>)</p>


## Results & Analysis

- 광각 환경에서도 안정적인 휴대폰 객체 탐지 확인

- 시선 정보 결합을 통해 단순 객체 탐지 대비 오탐지 감소

- 실제 차량 환경을 가정한 시나리오에서 시스템 동작 검증

**Limitations**

- 눈·코 가림 상황에서는 시선 추정 성능 저하 발생

- 고정 카메라 위치 기반 학습을 통해 개선 가능
