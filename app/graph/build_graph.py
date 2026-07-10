"""
LangGraph 그래프 구성.
parse -> [조건분기: 실패시 종료] -> stats + retrieve(병렬 성격이지만 여기선 순차) -> gate

gate 이후(respond 스트리밍 / escalate)는 SSE 특성상 main.py에서 직접 처리.
(LangGraph 노드 자체를 스트리밍시키는 대신, 그래프는 "판단"까지만 하고
 스트리밍 생성은 FastAPI 레이어에서 함 - 스트리밍+그래프상태관리 동시최적화보다
 지금 단계에선 단순하고 명확한 구조를 우선함)
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from langgraph.graph import StateGraph, END

from app.graph.state import AgentState
from app.graph.nodes import (
    parse_node,
    route_after_parse,
    stats_node,
    retrieve_node,
    gate_node,
    escalate_node,
)


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("parse", parse_node)
    graph.add_node("stats", stats_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("gate", gate_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("parse")

    graph.add_conditional_edges(
        "parse",
        route_after_parse,
        {
            "parse_failed": END,
            "stats_and_retrieve": "stats",
        },
    )

    graph.add_edge("stats", "retrieve")
    graph.add_edge("retrieve", "gate")

    graph.add_conditional_edges(
        "gate",
        lambda state: "escalate" if state.get("should_escalate") else "respond_ready",
        {
            "escalate": "escalate",
            "respond_ready": END,
        },
    )

    graph.add_edge("escalate", END)

    return graph.compile()