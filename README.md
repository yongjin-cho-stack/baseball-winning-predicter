# MLB Win Predictor

MLB(메이저리그) 경기 데이터를 분석해 특정 날짜의 경기 승리팀을 예측하는 Random Forest 모델입니다.

## 프로젝트 목적

과거 MLB 팀 스탯(타율, ERA, OPS 등)을 학습해 미래 경기의 승리팀을 예측합니다.

## 사용 데이터

- **출처:** [MLB Stats API](https://statsapi.mlb.com) (`MLB-StatsAPI` 라이브러리)
- **기간:** 2023, 2025 정규시즌 (총 4,860경기)
- **주요 피처:**
  - 팀 타격: OBP, SLG, OPS, 안타, 홈런, 볼넷, 삼진, 득점, 2루타, 3루타, 잔루
  - 팀 투구: ERA, 삼진, 볼넷, 피안타, 자책점, 투구 수, 스트라이크, 이닝
  - 홈/어웨이 여부, 시즌 누적 승률
  - 최근 15경기 롤링 평균 적용

## 모델

- **알고리즘:** Random Forest Classifier (scikit-learn)
- **타겟:** 홈팀 승리 여부 (1: 홈팀 승, 0: 원정팀 승)
- **평가 지표:** Accuracy, F1-score

## 프로젝트 구조

```
MLB_WinPredictor/
├── data/
│   ├── raw/          # 원본 데이터
│   └── processed/    # 전처리된 데이터
├── notebooks/
│   └── EDA.ipynb     # 데이터 탐색 및 시각화
├── src/
│   ├── data_loader.py    # 데이터 수집
│   ├── preprocessing.py  # 전처리
│   ├── train.py          # 모델 학습
│   └── predict.py        # 예측
├── models/           # 학습된 모델 저장
├── requirements.txt
└── README.md
```

## 설치 방법

```bash
pip install -r requirements.txt
```

## 사용 방법

```bash
# 데이터 수집
python src/data_loader.py

# 모델 학습
python src/train.py

# 특정 날짜 경기 예측 (날짜를 직접 입력)
python src/predict.py
```

## 결과

| 시도 | 변경 내용 | Accuracy |
|------|-----------|----------|
| 1차 | 기본 스탯 14개, 롤링 10경기 | 49.9% |
| 2차 | 홈 어드밴티지, 시즌 누적 승률 추가, 롤링 15경기 | 49.9% |
| 3차 | 득점, 투구 수 등 미사용 컬럼 9개 추가 | 50.9% |
| 4차 | 2025 시즌 데이터 추가 (총 4,860경기) | 52.0% |

> 참고: MLB 예측 모델의 현실적 정확도는 55~62% 수준입니다.
