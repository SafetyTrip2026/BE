"""
SSE 스트림 데모 클라이언트.
curl로 보면 토큰이 쪼개져서 지저분해 보이지만, 실제 클라이언트는
token 이벤트의 text를 계속 이어붙이기만 하면 깔끔한 문장이 완성됨.
이 스크립트는 그 "이어붙이기"를 실제로 구현해서 최종 결과가 어떻게 보이는지
확인하기 위한 용도 (프론트엔드가 참고할 로직 예시이기도 함).

실행: python scripts/demo_client.py "8월 초에 부모님 모시고 부산 해운대 가는데 주의할 게 있을까?"
"""
import sys
import json
import requests


def run(query: str, base_url: str = "http://localhost:8000"):
    print(f"질문: {query}\n")
    print("-" * 60)

    resp = requests.post(
        f"{base_url}/chat/stream",
        json={"query": query},
        stream=True,
    )

    current_event = None
    full_answer = ""

    for raw_line in resp.iter_lines(decode_unicode=True):
        if raw_line is None or raw_line == "":
            continue

        if raw_line.startswith("event:"):
            current_event = raw_line[len("event:"):].strip()
            continue

        if raw_line.startswith("data:"):
            payload = raw_line[len("data:"):].strip()
            data = json.loads(payload)

            if current_event == "meta":
                print("[통계/검색 요약]")
                print(json.dumps(data, ensure_ascii=False, indent=2))
                print("-" * 60)
                print("[답변]\n")

            elif current_event == "token":
                # 여기가 핵심: 토큰을 그냥 계속 이어붙여서 바로 출력
                text = data["text"]
                full_answer += text
                print(text, end="", flush=True)

            elif current_event == "reask":
                print(f"[재질문 필요] {data['message']}")

            elif current_event == "escalate":
                print(f"\n[에스컬레이션] {data['message']}")
                print(f"  담당기관: {data['contact']['agency']} ({data['contact']['phone']})")

            elif current_event == "done":
                print("\n" + "-" * 60)
                print("[스트리밍 종료]")

    print("\n\n=== 최종 조립된 전체 답변 (이렇게 프론트에 렌더링하면 됨) ===\n")
    print(full_answer)


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "8월 초에 부모님 모시고 부산 해운대 가는데 주의할 게 있을까?"
    run(query)