# Processed data

이 폴더에는 `data/raw`에서 생성한 재현 가능한 전처리 결과만 저장한다.

## 생성 명령

프로젝트 루트에서:

```text
python preprocess.py --mode all --overwrite
```

빠른 샘플 검증:

```text
python preprocess.py --mode all --limit-files 1 --overwrite
```

## 출력

- `trip_profiles/`: Trip별 시간·거리·target 품질
- `segments_250m/`: ICE 250m 구간별 DNN 학습 테이블 (`segment_energy_kwh` target)
- `trip_profiles/ice_trip_profiles.csv`: 전체 Trip master table
- `segments_250m/ice_segments_250m.csv`: 전체 segment master training table
- `preprocessing_manifest.json`: 전처리 설정
- `preprocessing_summary.json`: 처리 결과 합계

원본 파일은 수정하지 않는다.
