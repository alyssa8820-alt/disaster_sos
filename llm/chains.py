from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import get_llm

_MAX_ATTEMPTS = 2


def _invoke_json_chain(chain, payload):
    """JSON 파싱 실패 등 일시적 오류에 대비해 최대 _MAX_ATTEMPTS회까지 재시도한다."""
    last_error = None
    for _ in range(_MAX_ATTEMPTS):
        try:
            return chain.invoke(payload)
        except Exception as e:
            last_error = e
    raise last_error


def detect_and_translate(text: str) -> dict:
    """언어 감지 + 한국어 번역(화면·보고서 표시용) + 영어 번역(ML 분류기 입력 전용) 을
    LLM 1회 호출로 처리한다. 서비스가 한국 내 재난 신고만 다루므로 사용자·구조기관에게는
    korean_translation만 노출하고, english_translation은 화면에 보이지 않는 내부 값이다
    (학습된 ML 분류기가 영어 데이터셋으로 학습되어 있어 분류 입력만 영어가 필요하기 때문)."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You detect the language of the user's message and translate it. "
                "Respond ONLY as compact JSON with keys: "
                "detected_language (write the language name IN KOREAN, e.g. '프랑스어', '한국어', '영어'), "
                "language_code (ISO 639-1, e.g. 'fr'), "
                "korean_translation (translation of the message into Korean; if the message is "
                "already Korean, copy it as-is), "
                "english_translation (translation of the message into English; if the message is "
                "already English, copy it as-is).",
            ),
            ("human", "{text}"),
        ]
    )
    chain = prompt | llm | JsonOutputParser()
    result = _invoke_json_chain(chain, {"text": text})
    # 원문은 LLM이 다시 생성하게 하지 않고 사용자가 입력한 값을 그대로 사용해 100% 일치를 보장한다.
    result["original_text"] = text
    return result


def analyze_message(context: dict) -> dict:
    """번역문·ML 예측 결과를 바탕으로 (1) 위치/인원/피해 상황을 추출하고, (2) 누락된 항목을
    표시하며, (3) 누락 항목이 있으면 신고자의 원문 언어로 재요청 문구를 만들고, (4) 한국어
    상황 요약을 생성한다. 이 네 가지를 LLM 1회 호출로 함께 처리한다.

    context에는 translated_text(한국어 번역문), detected_language, top_labels, rescue_request,
    urgency가 포함되어야 한다.

    반환 키: location, people_count, damage_status, missing_fields(list), followup_message, summary
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a disaster response assistant. Based on the given structured JSON context "
                "(Korean translation of a disaster message as translated_text, the language the "
                "reporter used as detected_language, predicted aid types, rescue request flag, "
                "urgency level):\n"
                "1. Extract 'location' as a CLEAN geocodable place name within South Korea only "
                "(e.g. '서울 강남구', '부산 해운대구', '경기도 성남시 분당구') — an administrative "
                "region or well-known landmark name, WITHOUT descriptive words like building/facility "
                "types (dorm, school, hospital, etc.) or relative qualifiers (near, around). Use null "
                "if no place name is mentioned. Put any building/facility description into "
                "'damage_status' instead. Also extract 'people_count' (short phrase describing the "
                "number/group of people affected, or null if not mentioned) and 'damage_status' (short "
                "phrase describing the damage or situation, or null if not mentioned).\n"
                "2. Set 'missing_fields' to an array containing any of 'location', 'people_count', "
                "'damage_status' that are null.\n"
                "3. If missing_fields is non-empty, write ONE short polite sentence in the reporter's "
                "detected_language asking them to provide exactly those missing details, as "
                "'followup_message'. If nothing is missing, set 'followup_message' to an empty string.\n"
                "4. Write ONE concise Korean sentence summarizing who/what is affected and what help "
                "is needed, as 'summary'.\n"
                "5. Set 'disaster_type' to exactly one of: earthquake, fire, floods, storm, cold, "
                "other_weather — ONLY if the message CLEARLY and SPECIFICALLY describes that disaster "
                "(e.g. the message explicitly mentions a fire, an earthquake/shaking, a flood, a "
                "storm/typhoon, extreme cold, or other severe weather). If the message does not "
                "clearly describe one of these (e.g. it's about a building collapse with no stated "
                "cause, or a medical emergency alone), set 'disaster_type' to null. Do not guess — "
                "when in doubt, use null.\n"
                "Respond ONLY as compact JSON with keys: location, people_count, damage_status, "
                "missing_fields, followup_message, summary, disaster_type.",
            ),
            ("human", "{context}"),
        ]
    )
    chain = prompt | llm | JsonOutputParser()
    result = _invoke_json_chain(chain, {"context": context})
    result.setdefault("location", None)
    result.setdefault("people_count", None)
    result.setdefault("damage_status", None)
    result.setdefault("missing_fields", [])
    result.setdefault("followup_message", "")
    result.setdefault("disaster_type", None)
    result["summary"] = (result.get("summary") or "").strip()
    return result


def translate_text(text: str, target_language: str) -> str:
    """주어진 한국어 텍스트를 target_language로 번역하는 범용 유틸 (재난 대응 지침 번역에 사용)."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Translate the given Korean text into {target_language}, preserving the numbered "
                "list structure. Respond ONLY as compact JSON with a single key: translated_text.",
            ),
            ("human", "{text}"),
        ]
    )
    chain = prompt | llm | JsonOutputParser()
    result = _invoke_json_chain(chain, {"text": text, "target_language": target_language})
    return (result.get("translated_text") or "").strip()
