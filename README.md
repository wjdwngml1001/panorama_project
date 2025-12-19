# 🖼️ Computer Vision Final Project  
## Panorama Image Stitching
---

## ✅ 구현 내용 요약

| 항목 | 구현 여부 |
|---|---|
| 이미지 전처리 & 노이즈 제거 | ✅ |
| Harris Corner Detector | ✅ |
| NCC 기반 Point Matching | ✅ |
| Homography 계산 (DLT + SVD) | ✅ |
| RANSAC (Outlier 제거) | ✅ |
| Image Stitching | ✅ |
| Group Adjustment | △ |
| Tone Mapping | △ |

---

## 💻 실행 환경

- **OS**: Windows  
- **Language**: Python 3.x  
- **Libraries**:
  - numpy  
  - pillow  
  - matplotlib (디버그 시각화용)

---

## ▶️ 실행 방법

```bash
git clone https://github.com/wjdwngml1001/panorama_project.git
cd panorama_project

python -m venv venv
venv\Scripts\activate

예시 실행 명령어
```bash
(venv) python main.py --input "sampleset1" --pattern "testimg*.jpg" --out "results/result1.jpg" --debug_matches_dir "results/matches_set1"
```bash
(venv) python main.py --input "sampleset3/sampleset3" --pattern "testimg*.PNG" --out "results/result3.jpg" --debug_matches_dir "results/matches_set3"

###실행 옵션 설명

- --input : 입력 이미지 폴더 경로

- --pattern : 이미지 파일 이름 패턴

- --out : 최종 파노라마 이미지 출력 경로

- --debug_matches_dir : 중간 결과(코너, 매칭, RANSAC) 저장 경로

---

## 📁결과 이미지

sampleset1, sampleset2, sampleset3에 대한
파노라마 결과 이미지 포함

각 단계별 중간 결과(코너, 매칭, RANSAC inlier) 시각화 가능
