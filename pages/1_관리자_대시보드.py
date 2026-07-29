import pydeck as pdk
import streamlit as st

from core.db import fetch_all_reports

URGENCY_COLOR = {
    "높음": [220, 38, 38],
    "보통": [245, 158, 11],
    "낮음": [22, 163, 74],
}
DEFAULT_COLOR = [107, 114, 128]

st.set_page_config(page_title="관리자 대시보드 | 다국어 재난 신고 AI", page_icon="🗺️", layout="wide")

st.title("🗺️ 관리자 실시간 대시보드")
st.caption("접수된 재난 신고를 한눈에 확인합니다. 새 신고를 반영하려면 새로고침 버튼을 눌러주세요.")

if st.button("🔄 새로고침"):
    st.rerun()

df = fetch_all_reports()

if df.empty:
    st.info("아직 접수된 신고가 없습니다. 메인 페이지에서 신고를 분석하면 여기에 표시됩니다.")
else:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 신고", len(df))
    col2.metric("구조 요청", int(df["rescue_request"].sum()))
    col3.metric("긴급도 높음", int((df["urgency"] == "높음").sum()))
    col4.metric("위치 확인됨", int(df["lat"].notna().sum()))

    st.subheader("지도")
    map_df = df.dropna(subset=["lat", "lon"]).copy()
    if map_df.empty:
        st.info("좌표를 확인할 수 있는 신고가 없습니다 (위치 정보 없음 또는 지오코딩 실패).")
    else:
        map_df["color"] = map_df["urgency"].apply(lambda u: URGENCY_COLOR.get(u, DEFAULT_COLOR))
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[lon, lat]",
            get_fill_color="color",
            get_radius=20000,
            pickable=True,
        )
        view_state = pdk.ViewState(
            latitude=float(map_df["lat"].mean()),
            longitude=float(map_df["lon"].mean()),
            zoom=6.5,  # 대한민국 전역이 한눈에 들어오는 정도의 기본 확대 수준
        )
        tooltip = {"text": "위치: {location_text}\n긴급도: {urgency}\n요약: {summary}"}
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))
        st.caption("🔴 긴급도 높음 · 🟠 보통 · 🟢 낮음")

    st.subheader("신고 목록")
    display_df = df[
        [
            "updated_at",
            "detected_language",
            "location_text",
            "people_count",
            "urgency",
            "rescue_request",
            "summary",
        ]
    ].rename(
        columns={
            "updated_at": "갱신 시각",
            "detected_language": "언어",
            "location_text": "위치",
            "people_count": "인원",
            "urgency": "긴급도",
            "rescue_request": "구조 요청",
            "summary": "요약",
        }
    )
    display_df["구조 요청"] = display_df["구조 요청"].map({1: "예", 0: "아니오"})
    st.dataframe(display_df, width="stretch", hide_index=True)

    with st.expander("전체 원본 데이터 보기"):
        st.dataframe(df, width="stretch", hide_index=True)
