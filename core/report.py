# PRD.md 4.4절 포맷에 맞춘 구조기관 전달 보고서 조립.
# 구조화된 필드(언어/지원유형/구조요청/긴급도)는 실제 계산값을 그대로 사용하고,
# LLM은 "상황 요약" 한 문장만 생성해 정확도(할루시네이션 방지)와 포맷 일관성을 보장한다.


def build_report(
    detected_language: str,
    original_text: str,
    top_label_names: list,
    rescue_request: bool,
    urgency: str,
    summary: str,
    location: str | None = None,
    people_count: str | None = None,
    damage_status: str | None = None,
) -> str:
    lines = [
        f"- 원문 언어: {detected_language}",
        f"- 재난 메시지: {original_text}",
        f"- 위치: {location or '확인 필요'}",
        f"- 인원: {people_count or '확인 필요'}",
        f"- 피해 상황: {damage_status or '확인 필요'}",
        f"- 주요 지원 유형: {', '.join(top_label_names) if top_label_names else '해당 없음'}",
        f"- 구조 요청 여부: {'예' if rescue_request else '아니오'}",
        f"- 긴급도: {urgency}",
        f"- 상황 요약: {summary}",
    ]
    return "\n".join(lines)
