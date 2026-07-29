import uuid

import streamlit as st

from core.db import upsert_report
from core.geocode import geocode_location
from core.guidelines import GUIDELINE_LABEL_KO, GUIDELINES_KO, validate_disaster_type
from core.report import build_report
from core.scoring import compute_rescue_request, compute_urgency
from llm.chains import analyze_message, detect_and_translate, translate_text
from ml.train import predict_label_probs, top_labels_from_probs

TOP_N = 5
DISPLAY_MIN_PROBABILITY = 0.15  # 화면에는 이 값 미만인 라벨은 표시하지 않음 (규칙 기반 스코어링에는 영향 없음)

MISSING_FIELD_KO = {"location": "위치", "people_count": "인원", "damage_status": "피해 상황"}

EXAMPLES = {
    "🇰🇷 한국어 예시": "서울 강남구의 한 건물이 무너져서 학생 5명이 갇혔어요. 의료 지원이 급하게 필요합니다.",
    "🇺🇸 영어 예시": "I'm a foreign exchange student in Mapo-gu, Seoul, there's a fire in our dorm and several people are injured, we need medical help urgently.",
    "🇫🇷 프랑스어 예시": (
        "Je suis a Haeundae, Busan, une inondation a touche notre quartier et nous avons "
        "besoin de nourriture et d'eau de toute urgence."
    ),
}

URGENCY_BADGE_COLOR = {"높음": "red", "보통": "orange", "낮음": "green"}

st.set_page_config(page_title="다국어 재난 신고 AI", page_icon="🌍", layout="centered")

st.title("🌍 다국어 재난 신고 AI")
st.caption("재난 메시지를 어떤 언어로 입력해도 자동으로 번역·분석해 구조기관 보고서를 생성합니다.")

if "message_input" not in st.session_state:
    st.session_state.message_input = ""
if "report_id" not in st.session_state:
    st.session_state.report_id = None
if "result" not in st.session_state:
    st.session_state.result = None


def _set_example(text: str) -> None:
    st.session_state.message_input = text


def run_pipeline(detected_language: str, original_text: str, translated_text: str, ml_input_text: str) -> dict:
    """②~⑥ 단계(ML 예측, 위치/인원/피해 추출, 규칙 기반 스코어링, 요약·보고서, 대응 지침 번역)를
    모두 실행하고 결과를 하나의 dict로 반환한다. 최초 분석과, 누락 정보 보완 후 재분석 모두
    이 함수를 통해 처리한다.

    translated_text: 화면·보고서·LLM 컨텍스트에 쓰이는 한국어 번역문 (서비스가 한국 내 신고만
    다루므로 사용자에게 보이는 번역은 전부 한국어로 통일한다).
    ml_input_text: ML 분류기 입력 전용 영어 번역문. 학습된 TF-IDF+로지스틱회귀 모델이 영어
    데이터셋으로 학습되어 있어 분류 단계에서만 내부적으로 사용하고 화면에는 노출하지 않는다.
    """

    try:
        label_probs = predict_label_probs(ml_input_text)
        model_error = None
    except FileNotFoundError as e:
        model_error = str(e)
        label_probs = {}

    top_labels = top_labels_from_probs(label_probs, top_n=TOP_N, min_probability=DISPLAY_MIN_PROBABILITY)
    rescue_request = compute_rescue_request(label_probs)
    urgency = compute_urgency(label_probs)

    context = {
        "detected_language": detected_language,
        "translated_text": translated_text,
        "top_labels": [item["label_ko"] for item in top_labels],
        "rescue_request": rescue_request,
        "urgency": urgency,
    }
    try:
        analysis = analyze_message(context)
        analysis_error = None
    except Exception as e:
        analysis_error = str(e)
        analysis = {
            "location": None,
            "people_count": None,
            "damage_status": None,
            "missing_fields": [],
            "followup_message": "",
            "summary": "자동 요약을 생성하지 못했습니다. 잠시 후 다시 시도해주세요.",
        }

    lat, lon = None, None
    if analysis.get("location"):
        coords = geocode_location(analysis["location"])
        if coords:
            lat, lon = coords

    disaster_type = validate_disaster_type(analysis.get("disaster_type"))
    guideline = None
    guideline_error = None
    if disaster_type:
        korean_guideline = GUIDELINES_KO[disaster_type]
        translated_guideline = korean_guideline
        if detected_language not in ("한국어", "알 수 없음"):
            try:
                translated_guideline = translate_text(korean_guideline, detected_language)
            except Exception as e:
                guideline_error = str(e)
        guideline = {
            "disaster_type_ko": GUIDELINE_LABEL_KO[disaster_type],
            "korean": korean_guideline,
            "translated": translated_guideline,
        }

    report = build_report(
        detected_language=detected_language,
        original_text=original_text,
        top_label_names=[item["label_ko"] for item in top_labels],
        rescue_request=rescue_request,
        urgency=urgency,
        summary=analysis["summary"],
        location=analysis.get("location"),
        people_count=analysis.get("people_count"),
        damage_status=analysis.get("damage_status"),
    )

    return {
        "detected_language": detected_language,
        "original_text": original_text,
        "translated_text": translated_text,
        "ml_input_text": ml_input_text,
        "model_error": model_error,
        "top_labels": top_labels,
        "rescue_request": rescue_request,
        "urgency": urgency,
        "location": analysis.get("location"),
        "people_count": analysis.get("people_count"),
        "damage_status": analysis.get("damage_status"),
        "missing_fields": analysis.get("missing_fields") or [],
        "followup_message": analysis.get("followup_message") or "",
        "analysis_error": analysis_error,
        "summary": analysis["summary"],
        "report": report,
        "lat": lat,
        "lon": lon,
        "guideline": guideline,
        "guideline_error": guideline_error,
    }


def save_current_result() -> None:
    result = st.session_state.result
    if not result:
        return
    upsert_report(
        st.session_state.report_id,
        {
            "detected_language": result["detected_language"],
            "original_text": result["original_text"],
            "translated_text": result["translated_text"],
            "location_text": result["location"],
            "lat": result["lat"],
            "lon": result["lon"],
            "people_count": result["people_count"],
            "damage_status": result["damage_status"],
            "top_labels": [item["label_ko"] for item in result["top_labels"]],
            "rescue_request": result["rescue_request"],
            "urgency": result["urgency"],
            "summary": result["summary"],
            "report": result["report"],
        },
    )


st.write("빠른 시연을 위한 예시 문장")
example_cols = st.columns(len(EXAMPLES))
for col, (label, text) in zip(example_cols, EXAMPLES.items()):
    col.button(label, on_click=_set_example, args=(text,), width="stretch")

message = st.text_area(
    "재난 메시지를 입력하세요",
    key="message_input",
    height=120,
    placeholder="예: 부산 해운대구에 물이 필요합니다, 대피소를 알려주세요.",
)
run = st.button("🔍 분석 실행", type="primary", width="stretch")

if run:
    if not message.strip():
        st.warning("메시지를 입력해주세요.")
        st.stop()

    with st.spinner("메시지를 분석하고 있습니다..."):
        try:
            translation = detect_and_translate(message)
            translation_error = None
        except Exception as e:
            translation_error = str(e)
            translation = {
                "detected_language": "알 수 없음",
                "language_code": "-",
                "original_text": message,
                "korean_translation": message,
                "english_translation": message,
            }

        result = run_pipeline(
            detected_language=translation.get("detected_language") or "알 수 없음",
            original_text=translation.get("original_text") or message,
            translated_text=translation.get("korean_translation") or message,
            ml_input_text=translation.get("english_translation") or message,
        )
        result["translation_error"] = translation_error
        result["language_code"] = translation.get("language_code", "-")

    st.session_state.report_id = str(uuid.uuid4())
    st.session_state.result = result
    save_current_result()

if st.session_state.result:
    result = st.session_state.result
    st.divider()

    # ① 언어 감지 및 번역
    with st.container(border=True):
        st.subheader("① 언어 감지 및 번역")
        if result.get("translation_error"):
            st.error(f"번역 중 오류가 발생해 원문을 그대로 사용했습니다: {result['translation_error']}")
        st.markdown(f"**감지 언어**: {result['detected_language']} ({result.get('language_code', '-')})")
        st.markdown("**원문**")
        st.write(result["original_text"])
        st.markdown("**한국어 번역문**")
        st.write(result["translated_text"])

    # ② 긴급 지원 유형 예측
    with st.container(border=True):
        st.subheader("② 긴급 지원 유형 예측")
        if result.get("model_error"):
            st.error(result["model_error"])
        elif not result["top_labels"]:
            st.info("표시할 만큼 확률이 높은 지원 유형이 없습니다.")
        else:
            for item in result["top_labels"]:
                pct = item["probability"]
                st.progress(pct, text=f"{item['label_ko']} — {pct * 100:.0f}%")

    # ③ 위치·인원·피해 상황 + 누락 정보 재요청
    with st.container(border=True):
        st.subheader("③ 위치·인원·피해 상황")
        if result.get("analysis_error"):
            st.error(f"정보 추출 중 오류가 발생했습니다: {result['analysis_error']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("위치", result["location"] or "확인 필요")
        col2.metric("인원", result["people_count"] or "확인 필요")
        col3.metric("피해 상황", result["damage_status"] or "확인 필요")

        missing = result.get("missing_fields") or []
        if missing:
            missing_ko = ", ".join(MISSING_FIELD_KO.get(f, f) for f in missing)
            st.warning(f"다음 정보가 누락되었습니다: {missing_ko}")
            if result.get("followup_message"):
                st.info(f"신고자에게 보낼 재요청 문구 ({result['detected_language']}): {result['followup_message']}")

            with st.form("followup_form"):
                additional_info = st.text_area("누락된 정보를 추가로 입력해주세요 (어떤 언어든 가능)", height=80)
                submitted = st.form_submit_button("정보 추가 제출")

            if submitted and additional_info.strip():
                with st.spinner("추가 정보를 반영해 다시 분석하고 있습니다..."):
                    try:
                        addl_translation = detect_and_translate(additional_info)
                        addl_korean = addl_translation.get("korean_translation") or additional_info
                        addl_english = addl_translation.get("english_translation") or additional_info
                        addl_original = addl_translation.get("original_text") or additional_info
                    except Exception as e:
                        st.error(f"추가 정보 번역 중 오류가 발생했습니다: {e}")
                        addl_korean = additional_info
                        addl_english = additional_info
                        addl_original = additional_info

                    merged_translated = f"{result['translated_text']} {addl_korean}".strip()
                    merged_ml_input = f"{result['ml_input_text']} {addl_english}".strip()
                    merged_original = f"{result['original_text']} / {addl_original}".strip()

                    new_result = run_pipeline(
                        detected_language=result["detected_language"],
                        original_text=merged_original,
                        translated_text=merged_translated,
                        ml_input_text=merged_ml_input,
                    )
                    new_result["translation_error"] = None
                    new_result["language_code"] = result.get("language_code", "-")
                    st.session_state.result = new_result
                    save_current_result()
                st.rerun()
        else:
            st.success("위치·인원·피해 상황 정보가 모두 확인되었습니다.")

    # ④ 구조 요청 여부 및 긴급도
    with st.container(border=True):
        st.subheader("④ 구조 요청 여부 및 긴급도")
        badge_col1, badge_col2 = st.columns(2)
        with badge_col1:
            st.markdown("**구조 요청**")
            st.badge(
                "예" if result["rescue_request"] else "아니오",
                color="red" if result["rescue_request"] else "green",
            )
        with badge_col2:
            st.markdown("**긴급도**")
            st.badge(result["urgency"], color=URGENCY_BADGE_COLOR.get(result["urgency"], "gray"))

    # ⑤ AI 요약 및 구조기관용 보고서
    st.subheader("⑤ AI 요약 및 구조기관용 보고서")
    with st.container(border=True):
        st.markdown("**📝 AI 요약**")
        st.code(result["summary"], language=None, wrap_lines=True)

    with st.container(border=True):
        st.markdown("**📋 구조기관 전달 보고서**")
        st.code(result["report"], language=None, wrap_lines=True)

    # ⑥ 재난 대응 지침 (다국어 번역)
    guideline = result.get("guideline")
    if guideline:
        st.subheader("⑥ 재난 대응 지침")
        with st.container(border=True):
            st.markdown(f"**유형**: {guideline['disaster_type_ko']}")
            if result.get("guideline_error"):
                st.error(f"지침 번역 중 오류가 발생해 한국어 원문을 표시합니다: {result['guideline_error']}")
            st.markdown(f"**{result['detected_language']} 번역**")
            st.code(guideline["translated"], language=None, wrap_lines=True)
            with st.expander("한국어 원문 보기"):
                st.write(guideline["korean"])

    st.caption("💡 관리자용 지도·통계 대시보드는 왼쪽 상단 메뉴의 '관리자 대시보드' 페이지에서 확인할 수 있습니다.")
