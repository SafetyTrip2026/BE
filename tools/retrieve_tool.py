"""
disaster_guidelines pgvector 유사도 검색 (retrieve 노드용).
질문은 solar-embedding-1-large-query로, 문서는 이미 -passage로 임베딩되어 있음
(같은 벡터 공간이라 서로 비교 가능).

실행: python tools/retrieve_tool.py "해양오염사고 발생하면 어떻게 대피해야 하나요"
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
EMBEDDING_URL = "https://api.upstage.ai/v1/solar/embeddings"
QUERY_MODEL = "solar-embedding-1-large-query"


def embed_query(text: str) -> list:
    headers = {"Authorization": f"Bearer {UPSTAGE_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(EMBEDDING_URL, headers=headers,
                        json={"input": text, "model": QUERY_MODEL}, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def retrieve_guidelines(query: str, top_k: int = 5):
    """
    질문과 코사인 거리가 가까운 행동요령 top_k개를 반환.
    인덱스 없이 순차 스캔(현재 데이터 규모에서는 충분히 빠름).
    """
    query_embedding = embed_query(query)
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, content, safety_cate_nm1, safety_cate_nm2, safety_cate_nm3,
               source_dataset, embedding <=> %s::vector AS distance
        FROM disaster_guidelines
        ORDER BY distance ASC
        LIMIT %s
    """, (embedding_str, top_k))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "content": r[1],
            "cate_nm1": r[2],
            "cate_nm2": r[3],
            "cate_nm3": r[4],
            "source_dataset": r[5],
            "distance": float(r[6]),
        })
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('사용법: python tools/retrieve_tool.py "질문 내용"')
        sys.exit(1)

    query = sys.argv[1]
    results = retrieve_guidelines(query)

    print(f"\n=== 질문: {query} ===\n")
    for r in results:
        print(f"[distance={r['distance']:.4f}] {r['cate_nm1']} > {r['cate_nm2']} > {r['cate_nm3']}")
        print(f"  {r['content'][:150]}")
        print()