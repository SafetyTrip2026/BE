"""
FastAPI 앱. SSE로 그래프 판단 결과(meta) + 스트리밍 답변(token)을 순서대로 흘려보냄.

실행: uvicorn app.main:app --reload --port 8000
테스트: curl -N -X POST http://localhost:8000/chat/stream \
        -H "Content-Type: application/json" \
        -d '{"query": "8월 초에 부모님 모시고 부산 해운대 가는데 주의할 게 있을까?"}'
"""
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.graph.build_graph import build_graph
from app.llm_client import build_respond_prompt, stream_response

app = FastAPI(title="재난안전 여행 가이드 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 배포 시 실제 프론트엔드 도메인으로 제한
    allow_methods=["*"],
    allow_headers=["*"],
)

_graph = build_graph()


class ChatRequest(BaseModel):
    query: str


def sse_event(event: str, data: dict) -> str:
    """SSE 포맷 한 줄 생성"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def generate_sse_stream(user_query: str):
    # 1) 그래프 실행 (parse -> stats/retrieve -> gate, 여기까진 논스트리밍)
    result_state = _graph.invoke({"user_query": user_query})

    # 2) 파싱 실패 -> 재질문 요청
    # (reactive 질문은 region_sido 없어도 정상이라, prevention일 때만 지역 필수로 체크)
    intent = result_state.get("intent")
    is_unrecoverable = (
        result_state.get("parse_failed")
        or (intent == "prevention" and not result_state.get("region_sido"))
        or (intent == "reactive" and not result_state.get("disaster_type"))
        or intent not in ("prevention", "reactive")
    )
    if is_unrecoverable:
        yield sse_event("reask", {
            "message": "지역과 시기를 조금 더 구체적으로 말씀해 주시겠어요? "
                       "예: '8월 초에 부산 해운대 여행 가는데 주의할 점이 있을까요?'"
        })
        yield sse_event("done", {})
        return

    # 3) meta 이벤트: 통계/검색결과 요약 (프론트에서 차트 그릴 때 사용)
    stats_result = result_state.get("stats_result")
    meta = {
        "region_sido": result_state.get("region_sido"),
        "region_sigungu": result_state.get("region_sigungu"),
        "month": result_state.get("month"),
        "intent": result_state.get("intent"),
        "stats": {
            "scope_used": stats_result.scope_used if stats_result else None,
            "total_count": stats_result.total_count if stats_result else None,
            "breakdown": stats_result.breakdown if stats_result else [],
            "fallback_notice": stats_result.fallback_notice if stats_result else None,
        } if stats_result else None,
        "guideline_sources": [
            {
                "matched_disaster_type": g.get("matched_disaster_type"),
                "cate_nm2": g["cate_nm2"],
                "cate_nm3": g["cate_nm3"],
                "distance": g["distance"],
            }
            for g in (result_state.get("retrieved_guidelines") or [])
        ],
        "should_escalate": result_state.get("should_escalate", False),
    }
    yield sse_event("meta", meta)

    # 4) 에스컬레이션 분기
    if result_state.get("should_escalate"):
        from app.graph.nodes import escalate_node
        escalate_result = escalate_node(result_state)
        contact = escalate_result["escalate_contact"]
        yield sse_event("escalate", {
            "reason": result_state.get("escalate_reason"),
            "contact": contact,
            "message": (
                f"공식 매뉴얼에서 충분한 근거를 찾지 못했습니다. "
                f"아래 기관으로 문의해 주세요: {contact['agency']} ({contact['phone']})"
            ),
        })
        yield sse_event("done", {})
        return

    # 5) 정상 응답: LLM 스트리밍
    messages = build_respond_prompt(
        user_query=user_query,
        stats_result=stats_result,
        retrieved_guidelines=result_state.get("retrieved_guidelines") or [],
        has_vulnerable=result_state.get("has_vulnerable", False),
    )

    for token in stream_response(messages):
        yield sse_event("token", {"text": token})

    yield sse_event("done", {})


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        generate_sse_stream(req.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 등에서 SSE 버퍼링 방지
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}