# PRD.md 4.3절 기준 규칙 기반 구조 요청 여부 / 긴급도 산출

RESCUE_REQUEST_LABELS = {"request", "aid_related", "search_and_rescue"}
RESCUE_REQUEST_THRESHOLD = 0.5

# 긴급도 그룹 및 가중치 (그룹별 확률은 그룹 내 최댓값을 사용)
LIFE_RISK_LABELS = ["death", "medical_help", "search_and_rescue"]
DISASTER_TYPE_LABELS = ["earthquake", "fire", "floods", "storm"]
INFRA_LABELS = ["buildings", "shelter", "infrastructure_related"]

LIFE_RISK_WEIGHT = 3
DISASTER_TYPE_WEIGHT = 2
INFRA_WEIGHT = 1

HIGH_LIFE_RISK_THRESHOLD = 0.7  # 생명 위협군 단독으로 이 값 이상이면 무조건 "높음"
HIGH_SCORE_THRESHOLD = 4.0  # 가중 합산 점수 기준 (이론상 최댓값 6.0)
MEDIUM_SCORE_THRESHOLD = 2.0


def compute_rescue_request(label_probs: dict) -> bool:
    """label_probs: {라벨 컬럼명: 확률(0~1)}. 반드시 36개 라벨 전체 확률(predict_label_probs 결과)을
    넘겨야 하며, 화면 표시용으로 필터링된 top-N 딕셔너리를 넘기면 안 된다."""
    return any(label_probs.get(label, 0.0) >= RESCUE_REQUEST_THRESHOLD for label in RESCUE_REQUEST_LABELS)


def _group_max(label_probs: dict, labels: list) -> float:
    return max((label_probs.get(label, 0.0) for label in labels), default=0.0)


def compute_urgency(label_probs: dict) -> str:
    """label_probs: compute_rescue_request와 동일하게 라벨 전체 확률을 받아야 한다."""
    life_score = _group_max(label_probs, LIFE_RISK_LABELS)
    disaster_score = _group_max(label_probs, DISASTER_TYPE_LABELS)
    infra_score = _group_max(label_probs, INFRA_LABELS)

    weighted_total = (
        life_score * LIFE_RISK_WEIGHT
        + disaster_score * DISASTER_TYPE_WEIGHT
        + infra_score * INFRA_WEIGHT
    )

    if life_score >= HIGH_LIFE_RISK_THRESHOLD or weighted_total >= HIGH_SCORE_THRESHOLD:
        return "높음"
    if weighted_total >= MEDIUM_SCORE_THRESHOLD:
        return "보통"
    return "낮음"
