🖼️ Computer Vision Final Project
Panorama Image Stitching (From Scratch)

2020105660 정주희
컴퓨터비전 기말 프로젝트 – 파노라마 이미지 생성

✅ 구현 내용 요약
항목	구현 여부
이미지 전처리 & 노이즈 제거	✅
Harris Corner Detector	✅
NCC 기반 Point Matching	✅
Homography 계산 (DLT + SVD)	✅
RANSAC (Outlier 제거)	✅
Image Stitching	✅
Group Adjustment	△
Tone Mapping	△

🧩 전체 파이프라인

입력 이미지 로딩 및 정렬

이미지 전처리 (Grayscale 변환, 정규화)

Harris Corner Detector를 이용한 코너 검출

코너 주변 패치 기반 NCC(Point Matching)

대응점을 이용한 Homography 추정 (DLT)

RANSAC을 이용한 이상치 제거

Homography 기반 Inverse Warping

Feather Blending을 이용한 이미지 스티칭

최종 파노라마 이미지 생성 및 저장


💻 실행 환경

OS: Windows

Language: Python 3.x

Libraries:

numpy

pillow

matplotlib (디버그 시각화용)

▶️ 실행 방법
git clone https://github.com/wjdwngml1001/panorama_project.git
cd panorama_project

python -m venv venv
venv\Scripts\activate

예시 실행 명령어
(venv) python main.py --input "sampleset1" --pattern "testimg*.jpg" --out "results/result1.jpg" --debug_matches_dir "results/matches_set1"

(venv) python main.py --input "sampleset3/sampleset3" --pattern "testimg*.PNG" --out "results/result3.jpg" --debug_matches_dir "results/matches_set3"


--input : 입력 이미지 폴더 경로

--pattern : 이미지 파일 이름 패턴

--out : 최종 파노라마 이미지 출력 경로

--debug_matches_dir : 중간 결과(코너, 매칭, RANSAC) 저장 경로

📁 결과 이미지

sampleset1, sampleset2, sampleset3에 대한
파노라마 결과 이미지 포함

각 단계별 중간 결과 (코너, 매칭, RANSAC inlier) 시각화 가능
