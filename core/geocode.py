# OpenStreetMap Nominatim 기반 무료 지오코딩 (지명 -> 위도/경도).
# 서비스가 한국 내 재난 신고만 다루므로 country_codes="kr"로 검색 범위를 대한민국으로 제한한다
# (동명 지명이 해외에 있어도 잘못 매칭되지 않도록 함).
# Nominatim 사용 정책상 초당 1회로 요청 속도를 제한하고, 같은 지명은 캐시해서 재요청을 줄인다.

import logging
from functools import lru_cache

from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

logger = logging.getLogger(__name__)

_geolocator = Nominatim(user_agent="disaster_sos_app")
# swallow_exceptions=False로 두고 우리가 직접 예외를 잡아 로그로 남긴다.
# (Streamlit Cloud 등 배포 환경에서 지오코딩이 조용히 실패할 때 원인 진단이 안 되는 문제 방지)
_rate_limited_geocode = RateLimiter(_geolocator.geocode, min_delay_seconds=1, max_retries=1, swallow_exceptions=False)

_KOREA_COUNTRY_CODE = "kr"

last_error: str | None = None


@lru_cache(maxsize=512)
def geocode_location(location_text: str):
    """location_text를 대한민국 내 (위도, 경도) 튜플로 변환한다. 대한민국 내에서 찾지 못하거나
    입력이 비어 있으면 None을 반환한다. 예외 발생 시에도 None을 반환하되, 원인은 로그와
    last_error 변수에 남겨 배포 환경에서도 진단할 수 있게 한다."""
    global last_error
    if not location_text or not location_text.strip():
        return None
    try:
        result = _rate_limited_geocode(location_text, timeout=10, country_codes=_KOREA_COUNTRY_CODE)
    except Exception as e:
        last_error = f"{type(e).__name__}: {e}"
        logger.warning("지오코딩 실패 (%s): %s", location_text, last_error)
        return None
    if result is None:
        return None
    return (result.latitude, result.longitude)
