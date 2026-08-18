# EcoRoute (에코루트)

ESTsoft AI Challengers 최종 해커톤을 위한 친환경 경로 추천 데모입니다. Washtenaw County 도로망에서 여러 경로를 생성하고, 시간대별 교통 상태·도로 길이·경사도·차량 조건을 DNN에 입력해 경로별 에너지 소비량과 직접 배기관 탄소배출량을 비교합니다.

첫 화면은 **Ann Arbor**를 기본 지도로 열며, **Washtenaw County** 확장 지도도 선택할 수 있습니다.

## 실행

Python 3.14 환경에서 프로젝트 최상단의 실행 파일을 사용합니다.

```powershell
python -m pip install -r requirements.txt
python serve_demo.py
```

브라우저가 열리지 않으면 `http://127.0.0.1:8000`으로 접속합니다.
지도 화면은 Leaflet CDN과 OpenStreetMap 타일을 사용하므로 데모 실행 중 인터넷 연결이 필요합니다.

Ann Arbor에서 선택 가능한 도로 노드는 로컬 장소명 캐시를 사용해 가까운 영어 시설명, 주소, 도로명 순으로 표시합니다. Washtenaw County는 기존 노드 ID와 좌표를 표시합니다. 장소명 캐시는 아래 명령으로 다시 만들 수 있으며, 이 생성 단계에서만 OpenStreetMap 데이터를 내려받습니다.

```powershell
python scripts/build_ann_arbor_place_labels.py
```

첫 화면의 지역 버튼으로 **Ann Arbor**와 **Washtenaw County**를 전환할 수 있습니다. 기본 지역은 Ann Arbor입니다. 서버는 두 지역의 도로 그래프를 시작 시 한 번만 읽고, 최근 사용한 시간대별 교통 프로필을 메모리에 캐시합니다. 웹 요청은 지역 규모에 맞춘 경로 후보 설정을 사용하며, CLI 분석용 PNG·CSV는 생성하지 않아 반복 계산 시간을 줄였습니다.

출발시간 옆에서 운전 요일을 선택할 수 있으며, 이 값은 DNN 입력과 주간 기록에 함께 반영됩니다. 주간 리포트는 요일별 상대 CO2 절감률과 선택 경로·가장 빠른 경로의 누적 CO2eq, 경유 환산량, 예상 비용 및 총 에너지를 비교합니다. 같은 요일을 여러 번 기록하면 가장 최근 결과가 기존 값을 교체합니다. 기록은 브라우저 `sessionStorage`에만 저장되므로 새로고침에는 유지되지만 해당 탭이나 브라우저 창을 닫으면 삭제됩니다.

## 프론트엔드·백엔드 인계 구조

```text
EcoRoute/
├─ config/
│  └─ demo_runtime.json
├─ data/processed/
│  ├─ README.md
│  ├─ preprocessing_manifest.json
│  ├─ preprocessing_summary.json
│  ├─ maps/washtenaw_county/
│  │  ├─ washtenaw_county_drive_enriched.graphml
│  │  └─ metadata.json
│  └─ traffic/washtenaw_county/
│     └─ edge_hourly_profiles.csv.gz
├─ models/dnn/
│  └─ best_model.pt
├─ models/baseline/
│  └─ *.joblib
├─ results/
│  ├─ baseline/
│  │  ├─ figures/
│  │  ├─ learning_curve.csv
│  │  ├─ metrics.csv
│  │  ├─ split_summary.csv
│  │  └─ training_config.json
│  └─ dnn/
│     ├─ figures/
│     ├─ metrics.csv
│     ├─ split_summary.csv
│     ├─ training_config.json
│     └─ training_history.csv
├─ scripts/
├─ src/ecoroute/
├─ web/
│  ├─ index.html
│  ├─ styles.css
│  └─ app.js
├─ build_traffic.py
├─ predict_routes.py
├─ prepare_map.py
├─ preprocess.py
├─ route.py
├─ serve_demo.py
├─ train.py
├─ train_dnn.py
├─ requirements.txt
├─ .gitignore
└─ README.md
```

백엔드 실행에 필수인 세 가지 산출물은 다음과 같습니다.

- `models/dnn/best_model.pt`: 학습된 에너지 소비 예측 DNN 가중치
- `data/processed/maps/washtenaw_county/washtenaw_county_drive_enriched.graphml`: 도로·길이·제한속도·고도가 포함된 Washtenaw County 그래프
- `data/processed/traffic/washtenaw_county/edge_hourly_profiles.csv.gz`: 도로별 24시간 교통 프로필의 GitHub 업로드용 압축본

`src/ecoroute/`는 전처리, 지도 준비, 경로 탐색, 교통 프로필, DNN 추론, 탄소 환산 및 데모 API 전체 구현입니다. `web/`는 HTML/CSS/JavaScript 프론트엔드이며, `serve_demo.py`가 정적 파일과 API를 함께 제공합니다.

## GitHub에 올리지 않는 항목

`.gitignore`는 아래 로컬 자산을 제외합니다.

- `data/raw/`: eVED/VED 원본 데이터
- `data/cache/`: 도로 매칭 등 재생성 가능한 중간 캐시
- 학습용 세그먼트·trip profile·감사 결과(전처리 설명과 요약 JSON은 유지)
- 압축 전 대용량 교통 CSV
- 수십 MB짜리 행별 예측 CSV, 실행 시 재생성되는 경로 결과, 테스트 및 Python 캐시
- 개인 VS Code 설정과 가상환경

GitHub 웹의 **Add file**은 `.gitignore`를 자동으로 적용해 주지 않으므로, 수동 업로드할 때는 위 인계 구조에 표시된 파일과 폴더만 선택해야 합니다. 특히 `edge_hourly_profiles.csv`가 아니라 `edge_hourly_profiles.csv.gz`를 올립니다.

## 모델 출력 범위

현재 탄소 환산은 휘발유 에너지 33.7 kWh/US gal과 직접 배기관 배출 8.887 kg CO2/US gal을 사용합니다. 표시값은 모델과 환산 가정에 기반한 참고용 추정치이며 실제 배출량이나 절감 효과를 보장하지 않습니다. 연료 생산·정제·운송과 차량 제조 배출량은 포함하지 않습니다.

## 데이터 출처

- [Vehicle Energy Dataset (VED)](https://github.com/gsoh/VED)
- [Extended Vehicle Energy Dataset (eVED)](https://bitbucket.org/datarepo/eved-dataset/src/main/)
- [OpenStreetMap](https://www.openstreetmap.org/copyright) 도로 데이터
