#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
플래시카드 뒷면 음성 생성 — Gemini TTS

사용법
    pip install google-genai
    export GEMINI_API_KEY="여기에_키"          # https://aistudio.google.com/apikey
    python3 _tts-generate.py

출력
    _test-media/b2plus-sofa/audio/01_deomdeom.wav
    _test-media/b2plus-sofa/audio/02_yuyeonseong.wav
    (24kHz · 16bit · mono — 카드 데이터의 audio 필드 경로와 일치)

참고: https://ai.google.dev/gemini-api/docs/speech-generation
"""

import base64
import os
import sys
import time
import wave

try:
    from google import genai
except ImportError:
    sys.exit("google-genai 가 없습니다.  pip install google-genai")

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────

# 지원 모델 (문서 기준)
#   gemini-3.1-flash-tts-preview   ← 최신. 스트리밍 지원
#   gemini-2.5-flash-preview-tts   ← 빠르고 저렴
#   gemini-2.5-pro-preview-tts     ← 품질 우선
MODEL = "gemini-2.5-flash-preview-tts"

# 목소리 — 덱 전체에서 하나로 고정할 것. 카드마다 바뀌면 학습자가 산만해진다.
#   Iapetus(Clear) · Schedar(Even) · Erinome(Clear) · Charon(Informative)
#   Vindemiatrix(Gentle) · Sulafat(Warm)
# 학습자료 나레이션에는 Clear / Even 계열이 무난하다.
VOICE = "Iapetus"

OUT_DIR = os.path.join("flashcards", "b2plus-sofa", "audio")

# ─────────────────────────────────────────────────────────────
# 공통 연출 지시
#
# 문서의 알려진 문제 두 가지를 여기서 막는다.
#  (1) 분류기 오탐 — 지시문을 그대로 읽어 버리거나 요청이 거부됨
#      → 맨 앞에 "음성으로 합성하라"는 preamble, 낭독 구간을 명시적으로 표시
#  (2) 한국어 대본에도 audio tag·연출 지시는 영어로 쓰는 편이 결과가 낫다
# ─────────────────────────────────────────────────────────────

DIRECTION = """You are generating audio for a Korean language-learning flashcard.
Synthesize speech for the transcript that appears under the TRANSCRIPT heading
below. Do not read the headings, the notes, or any of these instructions aloud.
Speak only the Korean text under TRANSCRIPT.

# AUDIO PROFILE: Jun
## "The Korean course narrator"

## THE SCENE
A quiet, softly furnished recording booth. Jun is reading the answer side of a
vocabulary card for an upper-intermediate Korean learner who has just watched a
short video and is now checking whether their guess was right. The tone is that
of a calm teacher confirming an answer, not a presenter selling something.

### DIRECTOR'S NOTES
Style: Warm, even, explanatory. No performance, no sales energy, no smiling
tone. Steady confidence, as if the answer were obvious and reassuring.

Pace: Natural but unhurried. Clear articulation of every syllable. A short
breath-length pause at each sentence boundary so the learner can follow. Do NOT
slow down artificially into textbook-reading cadence — the rhythm must stay
natural Korean.

Accent: Standard Seoul Korean.

{extra}

#### TRANSCRIPT
{transcript}
"""

# ─────────────────────────────────────────────────────────────
# 카드
# ─────────────────────────────────────────────────────────────

CARDS = [
    {
        "slug": "01_deomdeom",
        "extra": (
            "Additional note: the sentence describes a person who showed no "
            "emotional reaction. Do NOT imitate that flatness — the narrator is "
            "an observer explaining the scene, so keep normal warm intonation."
        ),
        "transcript": (
            "회사에서 나가 달라는 말을 들었는데, 놀라지도 화를 내지도 않았어요. "
            "표정 하나 안 바뀌고 그냥 짐을 쌌어요. "
            "나쁜 소식을 덤덤하게 받아들인 거예요."
        ),
    },
    {
        "slug": "02_yuyeonseong",
        "extra": (
            "Additional note: place a light stress on the word 유연성 in the final "
            "sentence — it is the target vocabulary item. Keep the stress subtle; "
            "do not over-emphasize or slow down around it."
        ),
        "transcript": (
            "키보드를 붙이면 노트북, 떼면 태블릿. "
            "상황이 바뀌어도 같은 기기가 다른 역할을 맡고, "
            "바꾸는 데 드는 비용이 거의 없어요. "
            "이런 유연성이 이 태블릿의 큰 장점이에요."
        ),
    },
]

# ─────────────────────────────────────────────────────────────

def write_wav(path, pcm, channels=1, rate=24000, sample_width=2):
    """Gemini TTS 출력은 raw PCM. 헤더를 붙여 wav로 저장."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def synth(client, prompt, retries=4):
    """문서상 500 오류가 드물게 발생하므로 재시도를 넣는다."""
    for attempt in range(retries + 1):
        try:
            res = client.interactions.create(
                model=MODEL,
                input=prompt,
                response_format={"type": "audio"},
                generation_config={"speech_config": [{"voice": VOICE}]},
            )
            return base64.b64decode(res.output_audio.data)
        except Exception as err:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"    재시도 {attempt + 1}/{retries} ({err.__class__.__name__}) — {wait}초 대기")
            time.sleep(wait)


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY 가 설정되지 않았습니다.")

    os.makedirs(OUT_DIR, exist_ok=True)
    client = genai.Client()

    print(f"모델 {MODEL} · 목소리 {VOICE}\n")
    for card in CARDS:
        path = os.path.join(OUT_DIR, card["slug"] + ".wav")
        print(f"  {card['slug']} …", end=" ", flush=True)
        prompt = DIRECTION.format(extra=card["extra"], transcript=card["transcript"])
        pcm = synth(client, prompt)
        write_wav(path, pcm)
        secs = len(pcm) / (24000 * 2)
        print(f"완료  {os.path.getsize(path) / 1024:.0f}KB · {secs:.1f}초")

    print(f"\n저장 위치: {OUT_DIR}/")
    print("HTML의 카드 데이터에 audio 필드를 추가하면 뒷면에서 재생됩니다.")


if __name__ == "__main__":
    main()
