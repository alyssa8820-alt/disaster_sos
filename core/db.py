# 관리자 대시보드(기능 3)를 위한 신고 데이터 저장소. 별도 DB 서버 없이 SQLite 파일 하나로 관리한다.
# 주의: Streamlit Community Cloud 등 컨테이너가 재시작되는 배포 환경에서는 이 파일도 함께
# 초기화될 수 있어 완전한 영구 저장소는 아니다 (데모/로컬 실행 기준으로는 충분).

import json
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "reports.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    detected_language TEXT,
    original_text TEXT,
    translated_text TEXT,
    location_text TEXT,
    lat REAL,
    lon REAL,
    people_count TEXT,
    damage_status TEXT,
    top_labels TEXT,
    rescue_request INTEGER,
    urgency TEXT,
    summary TEXT,
    report TEXT
)
"""


def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def upsert_report(report_id: str, record: dict) -> None:
    """report_id가 이미 존재하면 갱신(재요청으로 정보가 보완된 경우), 없으면 새로 삽입한다."""
    now = datetime.now().isoformat(timespec="seconds")
    conn = _get_conn()
    with conn:
        existing = conn.execute("SELECT created_at FROM reports WHERE id = ?", (report_id,)).fetchone()
        created_at = existing[0] if existing else now
        conn.execute(
            """
            INSERT INTO reports (
                id, created_at, updated_at, detected_language, original_text, translated_text,
                location_text, lat, lon, people_count, damage_status, top_labels,
                rescue_request, urgency, summary, report
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                updated_at=excluded.updated_at,
                detected_language=excluded.detected_language,
                original_text=excluded.original_text,
                translated_text=excluded.translated_text,
                location_text=excluded.location_text,
                lat=excluded.lat,
                lon=excluded.lon,
                people_count=excluded.people_count,
                damage_status=excluded.damage_status,
                top_labels=excluded.top_labels,
                rescue_request=excluded.rescue_request,
                urgency=excluded.urgency,
                summary=excluded.summary,
                report=excluded.report
            """,
            (
                report_id,
                created_at,
                now,
                record.get("detected_language"),
                record.get("original_text"),
                record.get("translated_text"),
                record.get("location_text"),
                record.get("lat"),
                record.get("lon"),
                record.get("people_count"),
                record.get("damage_status"),
                json.dumps(record.get("top_labels", []), ensure_ascii=False),
                1 if record.get("rescue_request") else 0,
                record.get("urgency"),
                record.get("summary"),
                record.get("report"),
            ),
        )
    conn.close()


def fetch_all_reports():
    """모든 신고를 최신순으로 pandas DataFrame으로 반환한다."""
    import pandas as pd

    conn = _get_conn()
    df = pd.read_sql_query("SELECT * FROM reports ORDER BY created_at DESC", conn)
    conn.close()
    if not df.empty:
        df["top_labels"] = df["top_labels"].apply(lambda s: json.loads(s) if s else [])
    return df
