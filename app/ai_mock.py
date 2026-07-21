"""AI 피드백 / 튜터챗 생성기.

GEMINI_API_KEY가 설정되어 있으면 실제 Gemini 호출을 먼저 시도하고,
키가 없거나 호출이 실패하면(네트워크 오류, 쿼터 초과 등) 규칙 기반 mock으로 자동 폴백한다.
호출부인 routers/analyses.py, routers/ai.py는 그대로 두고 이 파일 안에서만 교체된다.
"""

import json
import logging
import re

from . import ai_client, models

logger = logging.getLogger(__name__)

STEP_LABEL = {
    "feeling": "Feeling",
    "beat": "Beat",
    "melody": "Melody",
    "structure": "Structure",
    "feedback": "Feedback",
}

FEEDBACK_SYSTEM_PROMPT = """당신은 작곡 입문자를 가르치는 다정하지만 전문적인 음악 프로듀서 튜터입니다.
학생이 레퍼런스 곡을 5단계(Feeling-Beat-Melody-Structure-Feedback)로 직접 스스로 분석한 내용을 검토하고,
구체적이고 실전적인 피드백을 줍니다. 정답을 대신 만들어주지 말고, 학생이 이미 관찰한 내용을 인정하고
심화시키는 방향으로 코칭하세요.

응답 형식 규칙:
- 반드시 한국어로, 간단한 HTML 태그(<strong>, <br>, <em>)만 사용해서 작성하세요. 마크다운이나 목록 태그는 쓰지 마세요.
- 3~5개 문단을 <br><br>로 구분하세요.
- 학생이 작성한 관찰/적용 계획 내용을 직접 인용하며 칭찬하고, 한 가지씩 더 발전시킬 방향을 제안하세요.
- 학생의 난이도(level)에 맞게 설명 깊이를 조절하세요 (beginner는 쉽게, advanced는 전문 용어 사용).
- 너무 길게 쓰지 말고 실전 조언 위주로 마무리하세요."""

CHAT_SYSTEM_PROMPT = """당신은 작곡과 프로듀싱 전문 음악 튜터입니다. 처음 배우는 입문자부터 중급자까지 누구나
이해할 수 있도록 용어를 설명합니다.

설명 방식:
- 전문 용어를 쉬운 말로 먼저 설명하고, 이후 정확한 개념을 덧붙이세요
- 실제 음악 예시(곡명, 아티스트)를 들어 설명하면 더 좋아요
- 한국어로 답변하되, 영어 용어는 함께 병기하세요
- 핵심만 명확하게, 너무 길지 않게 (3~5문단 이내)
- 입문자가 바로 이해할 수 있도록 친근하게

다룰 수 있는 주제: 곡 구조(Intro/Verse/Chorus/Bridge 등), 화성/코드(메이저/마이너, maj7/m7/dom7, 텐션),
리듬/그루브(BPM, 스윙, 그루브), 사운드/믹싱(리버브, 딜레이, 컴프레션, 사이드체인, 808), 장르 특징, DAW/프로듀싱 용어.

응답은 HTML 태그를 사용해서: <p>단락</p>, <strong>강조</strong>, <em>코드/영어</em>.
JSON이나 마크다운 코드블록 사용 금지."""


def _mock_feedback(session: "models.AnalysisSession") -> str:
    song = f"{session.artist} - {session.title}".strip(" -") or "이번 곡"
    parts: list[str] = []
    parts.append(f"<strong>{song} 언박싱, 수고하셨어요! 🎉</strong>")

    if session.note_beat.strip():
        parts.append(
            f"Beat 단계에서 남기신 관찰(\"{session.note_beat.strip()[:80]}\")을 보면, "
            "리듬 패턴을 귀로 정확히 짚어내고 계세요. 그 패턴이 어느 섹션에서 반복/변형되는지도 함께 체크해보면 좋아요."
        )
    if session.note_melody.strip():
        parts.append(
            f"멜로디 관찰(\"{session.note_melody.strip()[:80]}\")도 좋은 포인트예요. "
            "코드 진행과 멜로디 라인이 만나는 지점을 표시해두면 나중에 직접 작곡할 때 참고하기 훨씬 쉬워질 거예요."
        )
    if session.note_structure.strip():
        parts.append(
            f"Structure 노트(\"{session.note_structure.strip()[:80]}\")에서 섹션별 악기 레이어링 변화를 "
            "잘 짚어주셨어요. 그리드에서 표시한 레이어 온오프 타이밍이 실제로 긴장·이완을 만드는 핵심 장치예요."
        )

    if session.note_apply.strip():
        parts.append(
            f"<strong>적용 계획도 구체적이네요:</strong> \"{session.note_apply.strip()[:120]}\"<br>"
            "이 아이디어를 다음 곡의 같은 섹션에 그대로 적용해보고, 원곡과 어떻게 다르게 들리는지 비교해보세요."
        )
    else:
        parts.append(
            "다음 단계로, 오늘 관찰한 내용 중 하나를 골라 '내 다음 곡에는 이렇게 적용해보겠다'는 "
            "한 문장으로 정리해보면 언박싱이 훨씬 실전적으로 남아요."
        )

    parts.append(
        f"현재 난이도는 <em>{session.level}</em> 모드였어요. "
        "다음 곡에서는 한 단계 더 세밀한 관찰에 도전해보세요!"
    )
    return "<br><br>".join(parts)


_KEYWORD_REPLIES = [
    (("bpm", "템포", "빠르기"), "BPM은 분당 비트 수(Beats Per Minute)예요. 숫자가 높을수록 곡이 빠르게 느껴지고, 보통 발라드는 70~90, 댄스/팝은 110~130 정도예요."),
    (("브릿지", "bridge"), "브릿지는 벌스-코러스 반복 구조에 변화를 주기 위해 중간에 삽입하는 색다른 섹션이에요. 보통 코드 진행이나 리듬이 바뀌고, 다음 코러스를 더 극적으로 만들어주는 역할을 해요."),
    (("드롭", "drop"), "드롭은 긴장(빌드업) 뒤에 에너지가 확 터지는 구간이에요. EDM/팝에서 자주 쓰이고, 드럼과 베이스가 한꺼번에 들어오면서 카타르시스를 주죠."),
    (("808",), "808은 로랜드 TR-808 드럼머신에서 유래한 이름으로, 지금은 보통 '길게 늘어지는 서브베이스 킥'을 가리키는 말로 쓰여요. 힙합/R&B에서 저음을 채우는 핵심 요소예요."),
    (("maj7", "메이저7", "메이저seventh"), "maj7(메이저 세븐스) 코드는 메이저 코드에 장7도 음을 더한 코드예요. 기본 메이저보다 몽환적이고 재즈/R&B 느낌이 나요. 예: Cmaj7 = C-E-G-B."),
    (("min7", "마이너7", "m7"), "m7(마이너 세븐스) 코드는 마이너 코드에 단7도 음을 더한 거예요. 메이저 계열보다 부드럽고 여운이 남는 느낌을 줘요."),
    (("리버브", "reverb"), "리버브는 소리가 공간에서 반사되며 남는 잔향을 인공적으로 만드는 이펙트예요. 넓은 공간감을 줘서 소리를 '멀리, 크게' 들리게 해요."),
    (("딜레이", "delay"), "딜레이는 소리를 일정 시간 뒤에 반복 재생하는 이펙트예요. 리버브가 '공간감'이라면 딜레이는 '메아리'에 가까운 반복감을 줘요."),
    (("사이드체인", "sidechain"), "사이드체인은 한 트랙(보통 킥)이 울릴 때 다른 트랙(보통 베이스/패드)의 볼륨을 순간적으로 낮추는 테크닉이에요. 킥이 더 선명하게 들리고 곡에 펌핑감을 줘요."),
    (("보컬", "vocal"), "보컬 레이어링은 리드 보컬 아래/위에 하모니나 더블링을 쌓는 걸 말해요. 코러스에서 두꺼워지는 보컬은 대부분 이 레이어링 때문이에요."),
    (("코러스", "chorus", "후렴"), "코러스(후렴)는 곡에서 가장 기억에 남는 핵심 멜로디·메시지가 나오는 구간이에요. 보통 가장 넓은 음역대와 두꺼운 편곡으로 에너지를 최고조로 끌어올려요."),
    (("벌스", "verse"), "벌스는 이야기를 전개하는 구간이에요. 코러스보다 편곡을 절제해서, 코러스가 나왔을 때의 대비 효과를 살려주는 역할을 해요."),
    (("프리코러스", "pre-chorus", "프리 코러스"), "프리코러스는 벌스에서 코러스로 넘어가기 직전, 긴장을 쌓아 올리는 구간이에요. 코드가 불안정해지거나 리듬이 촘촘해지는 경우가 많아요."),
]


def _mock_chat_reply(message: str, step: str | None = None) -> str:
    lowered = message.lower()
    for keywords, reply in _KEYWORD_REPLIES:
        if any(k in lowered for k in keywords):
            return f"<p>{reply}</p>"

    step_label = STEP_LABEL.get(step or "", "")
    hint = f" 지금은 <strong>{step_label}</strong> 단계를 보고 계시니, 이 단계와 관련된 용어를 물어보셔도 좋아요." if step_label else ""
    return (
        "<p>아직 제가 바로 답변드릴 수 있는 캔드 답변 목록에는 없는 질문이에요. "
        "BPM, 브릿지, 드롭, 808, maj7/min7 코드, 리버브/딜레이, 사이드체인, 코러스/벌스/프리코러스 같은 "
        f"음악 용어를 물어보시면 자세히 설명해드릴게요.{hint}</p>"
    )


def _session_summary_prompt(session: "models.AnalysisSession") -> str:
    song = f"{session.artist} - {session.title}".strip(" -") or "제목 미상"
    lines = [
        f"곡: {song}", f"장르: {session.genre}", f"난이도: {session.level}", "",
        f"[Beat 관찰] {session.note_beat.strip() or '(작성 안 함)'}",
        f"[Melody 관찰] {session.note_melody.strip() or '(작성 안 함)'}",
        f"[Structure 관찰] {session.note_structure.strip() or '(작성 안 함)'}",
        f"[내 창작 적용 계획] {session.note_apply.strip() or '(작성 안 함)'}",
    ]
    return "\n".join(lines) + "\n\n위 내용을 바탕으로 피드백을 작성해주세요."


def generate_feedback(session: "models.AnalysisSession") -> str:
    if ai_client.is_configured():
        try:
            return ai_client.chat_once(FEEDBACK_SYSTEM_PROMPT, _session_summary_prompt(session))
        except Exception as e:  # noqa: BLE001
            logger.warning("Gemini feedback call failed, falling back to mock: %s", e)
    return _mock_feedback(session)


def generate_chat_reply(message: str, step: str | None = None) -> str:
    if ai_client.is_configured():
        try:
            prompt = message if not step else f"(현재 {STEP_LABEL.get(step, step)} 단계를 보는 중) {message}"
            return ai_client.chat_once(CHAT_SYSTEM_PROMPT, prompt)
        except Exception as e:  # noqa: BLE001
            logger.warning("Gemini chat call failed, falling back to mock: %s", e)
    return _mock_chat_reply(message, step)


REFERENCE_ANALYSIS_SYSTEM_PROMPT = """당신은 음악 프로덕션 전문 분석가입니다. 주어진 곡 정보(곡명, 아티스트, 작곡가, 길이)를 바탕으로,
당신이 실제로 알고 있는 지식을 활용해 곡 구조를 최대한 정확하게 분석하세요. 만약 정확히 아는 곡이 아니라면,
해당 장르/아티스트 스타일의 전형적인 구조와 길이 비율을 근거로 합리적인 추정치를 만드세요.

반드시 아래 JSON 스키마와 정확히 동일한 형식으로만 응답하세요. 다른 설명 텍스트, 마크다운 코드블록(```) 없이 순수 JSON 객체만 출력하세요.

{
  "sections": [
    { "label": "Intro", "start": 0, "end": 15 },
    { "label": "Verse 1", "start": 15, "end": 45 }
  ],
  "tracks": [
    { "name": "Drums", "segs": [ { "s": 0, "e": 15, "label": "설명 텍스트", "ghost": false } ] },
    { "name": "Bass", "segs": [ ... ] },
    { "name": "Chords", "segs": [ ... ] },
    { "name": "Vocal", "segs": [ ... ] },
    { "name": "FX", "segs": [ ... ] }
  ],
  "harmony": {
    "top": [ { "s": 0, "e": 15, "label": "코드 진행 텍스트" } ],
    "bottom": [ { "s": 45, "e": 75, "label": "코드 진행 텍스트" } ],
    "freq": { "섹션라벨": [0.5, 0.6, 0.4] },
    "noteText": "이 곡의 구조적 특징을 2~3문장으로 설명"
  },
  "quiz": [
    { "s": 0, "e": 15, "section": "Intro", "chordName": "Dm7", "root": "D", "quality": "min" }
  ]
}

규칙:
- sections는 곡 전체를 처음부터 끝까지 빈틈없이 순서대로 덮어야 해요 (한 섹션의 end가 다음 섹션의 start와 같아야 함).
- 섹션 라벨은 Intro, Verse 1, Verse 2, Pre-chorus, Chorus, Bridge, Outro, Hook, Drop 등 실제 곡 구조에 맞게 사용하세요.
- tracks의 각 세그먼트 label은 한국어로, 그 구간에서 해당 악기가 어떻게 연주되는지 짧고 구체적으로 설명하세요.
- 해당 구간에 그 트랙이 완전히 빠지면 ghost: true로 표시하세요.
- Chords 트랙의 label에는 실제 코드명(가능한 경우)이나 화성적 특징을 적으세요.
- harmony.top은 벌스류 코드 진행, harmony.bottom은 코러스류 코드 진행을 나타내요.
- harmony.freq는 섹션 라벨을 key로, [Low, Mid, High] 대역 에너지값(0~1)을 value로 하세요. sections의 모든 라벨이 freq에 포함되어야 해요.
- 맨 뒤 sections의 마지막 end 값이 곡 전체 길이(초)를 넘지 않아야 해요.
- quiz는 이 곡의 실제(또는 추정) 코드 진행 중 대표적인 코드 변화 구간을 6~10개 뽑아서, 귀로 코드를 맞히는 이어카피 퀴즈용 데이터로 제공하세요. 반드시 이 곡에 맞게 매번 다르게 생성하세요.
- quiz[].root는 C, C#, D, D#, E, F, F#, G, G#, A, A#, B 중 하나(샤프 표기)여야 해요.
- quiz[].quality는 'maj'(메이저/maj7), 'min'(마이너/m7), 'dom'(도미넌트 7th), 'dim'(디미니시드/기타) 중 하나여야 해요."""


def generate_reference_structure(title: str, artist: str, composer: str, duration: int) -> dict:
    """풀스코어 모드에서 유튜브 링크로 곡을 추가할 때, 실제 Gemini에게 곡 구조를 추론시킨다.
    GEMINI_API_KEY가 없거나 호출/파싱이 실패하면 예외를 그대로 던진다 — 호출부(라우터)가
    이를 502로 변환하면, 프론트엔드는 자체 mock 휴리스틱으로 폴백한다."""
    user_prompt = (
        f"곡명: {title}\n아티스트: {artist}\n작곡가: {composer or '정보 없음'}\n곡 길이: {duration}초\n\n"
        "이 곡의 구조를 분석해서 JSON으로 응답해주세요."
    )
    raw = ai_client.chat_once(REFERENCE_ANALYSIS_SYSTEM_PROMPT, user_prompt, max_tokens=4000)
    raw = raw.strip()
    raw = re.sub(r"^```json\s*", "", raw, flags=re.I)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"```\s*$", "", raw)
    parsed = json.loads(raw)

    sections = parsed.get("sections") or []
    if sections and sections[-1].get("end", 0) < duration:
        sections[-1]["end"] = duration

    harmony = parsed.get("harmony") or {}
    return {
        "data": {"TOTAL": duration, "sections": sections, "tracks": parsed.get("tracks") or []},
        "harmony": {
            "top": harmony.get("top") or [],
            "bottom": harmony.get("bottom") or [],
            "freq": harmony.get("freq") or {},
            "noteText": harmony.get("noteText") or "AI가 분석한 곡 구조예요.",
        },
        "quiz": parsed.get("quiz") or [],
    }
