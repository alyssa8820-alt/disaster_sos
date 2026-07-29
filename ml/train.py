import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from ml.labels import AID_TYPE_LABEL_COLUMNS, LABEL_COLUMNS, LABEL_KO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.joblib")
TRAIN_CSV = os.path.join(BASE_DIR, "disaster_response_messages_training.csv")
VALIDATION_CSV = os.path.join(BASE_DIR, "disaster_response_messages_validation.csv")


def _load_dataset(path):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=["message"])
    # 'related' 컬럼에 0/1 외 2(비정상 값)가 섞여 있어 다중 라벨 이진 분류를 위해 1로 통일
    df[LABEL_COLUMNS] = df[LABEL_COLUMNS].fillna(0).astype(int).clip(upper=1)
    return df


def train_and_save():
    train_df = _load_dataset(TRAIN_CSV)
    val_df = _load_dataset(VALIDATION_CSV)

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")),
            ("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))),
        ]
    )

    pipeline.fit(train_df["message"], train_df[LABEL_COLUMNS])

    val_pred = pipeline.predict(val_df["message"])
    print(classification_report(val_df[LABEL_COLUMNS], val_pred, target_names=LABEL_COLUMNS, zero_division=0))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"모델 저장 완료: {MODEL_PATH}")
    return pipeline


_cached_pipeline = None


def _get_pipeline():
    global _cached_pipeline
    if _cached_pipeline is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("학습된 모델이 없습니다. 'python -m ml.train'을 먼저 실행하세요.")
        _cached_pipeline = joblib.load(MODEL_PATH)
    return _cached_pipeline


def predict_label_probs(text: str) -> dict:
    """36개 라벨 전체에 대한 확률을 반환한다. 규칙 기반 스코어링(core/scoring.py)은
    화면에 표시되지 않는 라벨의 확률도 함께 필요하므로 이 함수의 결과를 사용해야 한다."""
    pipeline = _get_pipeline()
    probs = pipeline.predict_proba([text])[0]
    return {col: float(p) for col, p in zip(LABEL_COLUMNS, probs)}


def top_labels_from_probs(label_probs: dict, top_n: int = 5, min_probability: float = 0.0):
    """label_probs(전체 라벨 확률)에서 화면에 "긴급 지원 유형"으로 보여줄 상위 top_n개를 추린다.
    related/request 등 메타 라벨(AID_TYPE_LABEL_COLUMNS에서 제외됨)은 후보에서 제외한다."""
    candidates = ((label, label_probs.get(label, 0.0)) for label in AID_TYPE_LABEL_COLUMNS)
    ranked = sorted(candidates, key=lambda x: x[1], reverse=True)
    ranked = [(label, p) for label, p in ranked if p >= min_probability][:top_n]
    return [{"label": label, "label_ko": LABEL_KO[label], "probability": p} for label, p in ranked]


def predict_top_labels(text: str, top_n: int = 5, min_probability: float = 0.0):
    return top_labels_from_probs(predict_label_probs(text), top_n=top_n, min_probability=min_probability)


if __name__ == "__main__":
    train_and_save()
