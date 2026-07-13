"""
Solar Chat API 래퍼.
- parse_user_query: 자유 텍스트 -> 구조화 JSON (지역/시기/의도/재난유형/노약자 여부)
- stream_response_safe: 최종 답변을 토큰 단위로 스트리밍 생성 (SSE에서 그대로 사용)

Solar Chat Completions는 OpenAI SDK와 호환되는 인터페이스를 제공함
(base_url만 바꿔서 openai 패키지 그대로 사용).

LLM 3단 방어 (기획서 4번 섹션):
① 호출당 timeout 30초
② 일시 오류 시 지수 백오프 재시도 최대 2회
③ 최종 실패 시 답변 생성 대신 행동요령 원문 + 대응기관 안내로 안전하게 강등
   (③은 main.py에서 LLMUnavailableError/LLMStreamInterruptedError를 잡아서 처리)
"""
import os
import json
import re
import time
import logging

from openai import (
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    InternalServerError,
    AuthenticationError,
    PermissionDeniedError,
)
# Langfuse 드롭인 교체: OpenAI 클라이언트를 이걸로 쓰면 모든 chat.completions 호출
# (파싱 + 스트리밍 둘 다)이 자동으로 Langfuse에 트레이싱됨. 인터페이스는 원본과 동일.
from langfuse.openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("llm_client")

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
# 모델명은 .env로 오버라이드 가능 (Upstage 콘솔에서 실제 사용 가능한 모델명 확인 후 조정)
SOLAR_MODEL = os.getenv("SOLAR_MODEL", "solar-pro2")

# 3단 방어 설정값
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 2
BACKOFF_BASE_SECONDS = 1  # 1초 -> 2초로 지수 증가

# 재시도 대상으로 볼 예외 (일시적 오류 성격만 - 타임아웃/연결끊김/속도제한/서버내부오류)
RETRYABLE_EXCEPTIONS = (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)

# 재시도해도 절대 성공 못하는 영구적 오류 (인증/권한) - 재시도는 스킵하되
# 여전히 LLMUnavailableError로 변환해서 강등 처리로 이어지게 함
FATAL_EXCEPTIONS = (AuthenticationError, PermissionDeniedError)

_client = None


class LLMUnavailableError(Exception):
    """재시도를 전부 소진했는데 토큰을 하나도 못 받은 경우 (완전 실패)"""
    pass


class LLMStreamInterruptedError(Exception):
    """일부 토큰은 정상 전달됐는데 중간에 스트림이 끊긴 경우 (부분 실패)"""
    pass


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not UPSTAGE_API_KEY:
            raise RuntimeError("UPSTAGE_API_KEY가 .env에 설정되지 않았습니다.")
        _client = OpenAI(
            api_key=UPSTAGE_API_KEY,
            base_url="https://api.upstage.ai/v1",
            timeout=REQUEST_TIMEOUT_SECONDS,  # ① 호출당 timeout 30초
        )
    return _client


def _call_with_backoff(fn, *args, **kwargs):
    """
    ② 일시 오류 시 지수 백오프 재시도 (최대 MAX_RETRIES회).
    RETRYABLE_EXCEPTIONS만 재시도하고, FATAL_EXCEPTIONS(인증/권한 오류)는
    재시도해도 성공할 수 없으므로 즉시 전파. 그 외 예외도 즉시 전파.
    재시도를 다 소진하면 마지막 예외를 그대로 던짐 (호출부에서 처리).
    """
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except FATAL_EXCEPTIONS as e:
            logger.error(f"Solar API 인증/권한 오류 (재시도 무의미, 즉시 실패): {e}")
            raise
        except RETRYABLE_EXCEPTIONS as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE_SECONDS * (2 ** attempt)
                logger.warning(f"Solar API 호출 실패(시도 {attempt + 1}/{MAX_RETRIES + 1}): {e}. {wait}초 후 재시도.")
                time.sleep(wait)
            else:
                logger.error(f"Solar API 호출 최종 실패 ({MAX_RETRIES + 1}회 시도 소진): {e}")
    raise last_exc


PARSE_SYSTEM_PROMPT = """당신은 여행 안전 챗봇의 질문 분석기입니다.
사용자의 자연어 질문에서 다음 정보를 추출해서 JSON으로만 응답하세요. 설명이나 다른 텍스트는 절대 포함하지 마세요.

추출할 필드:
- region_sido: 시/도 (예: "부산광역시", "경기도"). 모르면 null
- region_sigungu: 시/군/구 (예: "해운대구"). 모르면 null
- month: 1~12 사이 정수. 언급 안 되면 null
- intent: "prevention"(여행 계획/예방형 질문, 통계 필요) 또는 "reactive"(이미 재난문자를 받았거나 특정 재난 대응법을 묻는 질문) 중 하나
- disaster_type: reactive일 때 언급된 구체적 재난유형(예: "호우", "폭염"). prevention이면 null
- has_vulnerable: 노약자/어린이/임산부 등 동반 여부 (boolean)

예시 입력: "8월 초에 부모님 모시고 부산 해운대 가는데 주의할 게 있을까?"
예시 출력: {"region_sido": "부산광역시", "region_sigungu": "해운대구", "month": 8, "intent": "prevention", "disaster_type": null, "has_vulnerable": true}
"""


def _strip_code_fence(text: str) -> str:
    """```json ... ``` 형태로 감싸져 오는 경우 제거"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_user_query(query: str) -> dict:
    """
    사용자 질문을 구조화 JSON으로 변환.
    - JSON 형식이 안 맞으면(모델이 지침을 안 따른 경우) 최대 1번 더 명시적으로 재요청
    - API 자체가 일시적으로 실패하면(타임아웃/연결오류 등) _call_with_backoff가 재시도
    - 그래도 최종 실패하면 LLMUnavailableError 발생 (파싱 실패와는 다른 상황이라 구분)
    """
    client = get_client()
    messages = [
        {"role": "system", "content": PARSE_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    for attempt in range(2):
        try:
            resp = _call_with_backoff(
                client.chat.completions.create,
                model=SOLAR_MODEL,
                messages=messages,
                temperature=0,
                name="parse_user_query",  # Langfuse 대시보드에서 이 이름으로 구분됨
            )
        except (RETRYABLE_EXCEPTIONS + FATAL_EXCEPTIONS) as e:
            raise LLMUnavailableError(f"parse 단계에서 Solar API 호출 실패: {e}") from e

        raw = resp.choices[0].message.content
        try:
            cleaned = _strip_code_fence(raw)
            parsed = json.loads(cleaned)
            return parsed
        except (json.JSONDecodeError, TypeError):
            if attempt == 0:
                messages.append({"role": "assistant", "content": raw})
                messages.append({
                    "role": "user",
                    "content": "JSON 형식이 아닙니다. 오직 JSON 객체만 응답하세요. 다른 텍스트 없이.",
                })
                continue
            # 2번 다 실패하면 파싱 실패를 알리는 결과 반환 (호출부에서 재질문 처리)
            return {
                "region_sido": None, "region_sigungu": None, "month": None,
                "intent": None, "disaster_type": None, "has_vulnerable": False,
                "_parse_failed": True,
            }


RESPOND_SYSTEM_PROMPT = """당신은 재난안전 정보만 안내하는 여행 안전 가이드 에이전트입니다.
제공된 통계 데이터와 행동요령 컨텍스트에 없는 내용은 절대 생성하지 않습니다.

응답 구성 규칙:
- 통계에서 확인된 재난유형별로 섹션을 나눠서 답변하세요 (예: "1. 폭염 대비", "2. 호우 대비").
- 각 섹션은 해당 재난유형으로 태깅된 행동요령 컨텍스트만 사용하세요.
- 먼저 "통계상 이 지역/시기에 어떤 재난이 몇 % 발생했는지"를 간단히 언급한 뒤, 그 재난에 대한 구체적 대비/행동요령을 이어서 안내하세요.

톤앤매너:
- 침착하고 명확한 안전 안내자 톤. 위험 정보는 과장하거나 축소하지 않습니다.
- "~할 수 있습니다" 같은 모호한 표현 대신 "~하세요 / ~를 피하세요"의 명확한 행동 지시형 문장을 사용합니다.
- 공포를 조장하는 표현(치명적, 재앙 등)은 사용하지 않되, 생명과 직결된 경고는 단호하게 표현합니다.
- 노약자 동반 사용자에게는 배려하는 어조를 더합니다.
- 각 안전 지침에는 근거(통계 수치 또는 행동요령 카테고리)를 명시하세요.
"""


def build_respond_prompt(user_query: str, stats_result, retrieved_guidelines: list, has_vulnerable: bool) -> list:
    """respond 노드에서 쓸 messages 리스트 구성 (stats + RAG 컨텍스트 주입)"""
    context_parts = []

    if stats_result is not None:
        stats_lines = [f"[통계] {stats_result.sido} {stats_result.sigungu or ''} {stats_result.month}월 재난 발생 빈도 (범위: {stats_result.scope_used}, 총 {stats_result.total_count}건)"]
        for item in stats_result.breakdown[:5]:
            stats_lines.append(f"  - {item['disaster_type']}: {item['count']}건 ({item['pct']}%)")
        if stats_result.fallback_notice:
            stats_lines.append(f"  [안내] {stats_result.fallback_notice}")
        context_parts.append("\n".join(stats_lines))

    if retrieved_guidelines:
        guide_lines = ["[행동요령 검색결과 - 재난유형별]"]
        grouped: dict = {}
        for g in retrieved_guidelines:
            key = g.get("matched_disaster_type") or "일반"
            grouped.setdefault(key, []).append(g)

        for dtype, items in grouped.items():
            guide_lines.append(f"  ● {dtype}:")
            for g in items:
                guide_lines.append(f"    - ({g['cate_nm2']} > {g['cate_nm3']}) {g['content']}")
        context_parts.append("\n".join(guide_lines))

    context_text = "\n\n".join(context_parts) if context_parts else "(제공된 컨텍스트 없음)"

    vulnerable_note = "\n동반자 중 노약자/어린이가 있으니 관련 주의사항을 더 챙겨서 안내하세요." if has_vulnerable else ""

    user_content = f"""사용자 질문: {user_query}

아래는 실제 공식 데이터에서 조회된 컨텍스트입니다. 이 안의 내용만 근거로 답변하세요.

{context_text}{vulnerable_note}
"""

    return [
        {"role": "system", "content": RESPOND_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _stream_once(messages: list):
    """실제 API 스트리밍 호출 1회 (재시도/방어 로직 없는 원본)"""
    client = get_client()
    stream = client.chat.completions.create(
        model=SOLAR_MODEL,
        messages=messages,
        stream=True,
        name="respond_stream",  # Langfuse 대시보드에서 이 이름으로 구분됨
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def stream_response_safe(messages: list):
    """
    SSE 엔드포인트에서 소비할 스트리밍 제너레이터. 3단 방어 적용:
    ① timeout 30초 (get_client에서 클라이언트 생성 시 설정됨)
    ② 재시도: 이번 시도에서 토큰을 하나도 못 받은 상태에서 실패하면
       -> 아직 사용자에게 아무것도 안 보낸 상태이므로 안전하게 재시도 가능
    ③ 최종 실패:
       - 토큰을 하나도 못 받고 재시도까지 소진 -> LLMUnavailableError
         (호출부가 행동요령 원문 + 대응기관 안내로 강등 처리)
       - 이미 일부 토큰은 보낸 뒤 중간에 실패 -> LLMStreamInterruptedError
         (재시도하면 앞부분이 중복되므로 재시도하지 않고 중단 처리)
    """
    last_exc = None

    for attempt in range(MAX_RETRIES + 1):
        yielded_any = False
        try:
            for delta in _stream_once(messages):
                yielded_any = True
                yield delta
            return  # 정상 완료
        except FATAL_EXCEPTIONS as e:
            # 인증/권한 오류는 재시도해도 성공 못하므로 즉시 강등 처리로 넘김
            logger.error(f"인증/권한 오류로 즉시 실패 처리: {e}")
            if yielded_any:
                raise LLMStreamInterruptedError(str(e)) from e
            raise LLMUnavailableError(str(e)) from e
        except RETRYABLE_EXCEPTIONS as e:
            last_exc = e
            if yielded_any:
                # 이미 일부 응답이 나간 상태 -> 재시도하면 중복/모순 발생, 중단 처리로 넘김
                logger.error(f"스트리밍 중간에 끊김 (이미 일부 토큰 전송됨): {e}")
                raise LLMStreamInterruptedError(str(e)) from e

            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE_SECONDS * (2 ** attempt)
                logger.warning(f"스트리밍 시작 전 실패(시도 {attempt + 1}/{MAX_RETRIES + 1}): {e}. {wait}초 후 재시도.")
                time.sleep(wait)
            else:
                logger.error(f"스트리밍 최종 실패 ({MAX_RETRIES + 1}회 시도 소진): {e}")

    raise LLMUnavailableError(str(last_exc))


def build_degraded_fallback_text(retrieved_guidelines: list) -> str:
    """
    ③ LLM 최종 실패 시 사용. LLM 가공 없이 검색된 행동요령 원문을 그대로 나열.
    (LLM이 없어도 이미 pgvector 검색은 완료된 상태이므로, 이 원문 자체는 신뢰 가능한 공식 데이터)
    """
    if not retrieved_guidelines:
        return "현재 상세 답변 생성이 어렵습니다. 아래 공식 연락처로 문의해 주세요."

    lines = ["※ 현재 AI 응답 생성에 일시적 문제가 있어, 검색된 공식 행동요령 원문을 그대로 안내합니다.\n"]
    grouped: dict = {}
    for g in retrieved_guidelines:
        key = g.get("matched_disaster_type") or g.get("cate_nm2") or "일반"
        grouped.setdefault(key, []).append(g)

    for dtype, items in grouped.items():
        lines.append(f"\n[{dtype}]")
        for g in items:
            lines.append(f"- ({g['cate_nm3']}) {g['content']}")

    return "\n".join(lines)