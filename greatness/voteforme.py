import streamlit as st

st.set_page_config(layout="centered")

# 색상 및 스타일 지정
PRIMARY_COLOR = "#0b2b45"
TEXT_COLOR = PRIMARY_COLOR
BG_COLOR = "#ffffff"

# 후보 점수 초기화
if "score" not in st.session_state:
    st.session_state.score = {"이재명": 0, "김문수": 0, "이준석": 0}
    st.session_state.q_index = 0
    st.session_state.answers = []
    st.session_state.finished = False

# 질문 및 선택지
QUESTIONS = [
    {
        "q": "집 문제 어떻게 해결해야 한다고 생각해?",
        "comment": "내 방 월세 진짜 너무 비싸..",
        "options": [
            ("나라가 임대주택을 많이 지어야 해. 월세 걱정 없는 사회!", "이재명"),
            ("사람들이 집 살 수 있게 규제를 풀어주면 자연스럽게 해결돼", "김문수"),
            ("청년 기숙사나 월세 지원 같은 게 현실적이지 않을까?", "이준석"),
        ],
    },
    {
        "q": "기본소득이 생긴다면 어때?",
        "comment": "모든 국민에게 매달 일정 금액 지급된다면",
        "options": [
            ("좋지! 누구든 기회를 가질 수 있잖아.", "이재명"),
            ("무상지원보다 자립이 중요하지.", "김문수"),
            ("청년 창업에 집중하는 게 더 실질적일 듯.", "이준석"),
        ],
    },
    {
        "q": "의료 파업 문제, 어떻게 해결하는 게 맞을까?",
        "comment": "병원 진료 중단, 국민 피해 발생 시",
        "options": [
            ("공공의료 확충하고 충분히 논의해야 해.", "이재명"),
            ("일단 복귀시키고 나중에 논의하자.", "김문수"),
            ("보건부 따로 만들어서 효율적으로 운영하자.", "이준석"),
        ],
    },
    {
        "q": "정치인들 싸우는 모습, 어떻게 바뀌면 좋을까?",
        "comment": "정치가 신뢰받으려면",
        "options": [
            ("검찰·의사 등 권력기관도 개혁해야.", "이재명"),
            ("국론 분열 일으키는 쪽부터 정리해야.", "김문수"),
            ("국회 구조부터 뜯어고치자.", "이준석"),
        ],
    },
    {
        "q": "외교·안보에서 가장 중요한 건?",
        "comment": "불안정한 국제 정세 속에서",
        "options": [
            ("북한과 대화하고 균형 외교가 중요해.", "이재명"),
            ("군사력 강화로 안보를 지켜야지.", "김문수"),
            ("현실적으로 이득 따져서 판단해야.", "이준석"),
        ],
    },
    {
        "q": "경제 성장, 어떤 방식이 더 효과적일까?",
        "comment": "경기 침체 속에서 정부 역할은?",
        "options": [
            ("복지를 통해 기회를 제공해야 해.", "이재명"),
            ("기업 자율성이 더 중요하지.", "김문수"),
            ("청년 창업을 적극 지원하자.", "이준석"),
        ],
    },
    {
        "q": "같이 일하고 싶은 정치인은 어떤 스타일?",
        "comment": "팀워크와 철학 중시한다면",
        "options": [
            ("국민 삶을 책임지는 리더.", "이재명"),
            ("국가 원칙을 지키는 강직한 인물.", "김문수"),
            ("공정한 경쟁을 추구하는 실용주의자.", "이준석"),
        ],
    },
]

# -----------------------------
# 헤더
# -----------------------------
st.markdown(f"""
<div style="background-color:{PRIMARY_COLOR}; padding: 20px; border-radius: 6px;">
  <h1 style="color:white; text-align:center;">President 101</h1>
  <p style="color:white; text-align:center; font-size:18px;">당신의 대통령에게 투표하세요!</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# 퀴즈 진행 중
# -----------------------------
if not st.session_state.finished:
    q = QUESTIONS[st.session_state.q_index]

    # 질문 박스
    st.markdown(f"""
    <div style="border: 1px solid {TEXT_COLOR}; padding: 25px; border-radius: 8px;">
      <p style="font-size:16px; color:{TEXT_COLOR};">{q['comment']}</p>
      <h2 style="margin-top: 0; color:{TEXT_COLOR};">{q['q']}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 선택지
    col1, col2, col3 = st.columns(3)
    for i, col in enumerate([col1, col2, col3]):
        label, cand = q['options'][i]
        if col.button(f"{label}"):
            st.session_state.score[cand] += 1
            st.session_state.q_index += 1
            if st.session_state.q_index >= len(QUESTIONS):
                st.session_state.finished = True
            st.experimental_rerun()

# -----------------------------
# 결과 페이지
# -----------------------------
else:
    winner = max(st.session_state.score, key=st.session_state.score.get)

    st.markdown("<br><hr>")
    st.header(f"🎉 당신은 '{winner}' 후보와 가장 유사합니다!")

    result_text = {
        "이재명": "- 복지 확대\n- 기본소득\n- 공공주택·평화 외교",
        "김문수": "- 자유시장 중심\n- 강경 안보\n- 전통 보수 가치",
        "이준석": "- 청년 중심 정책\n- 실용주의\n- 구조 개혁",
    }

    st.markdown(f"""
    <div style="background-color:#f0f8ff; padding: 20px; border-radius: 10px;">
        <p style="font-size:18px;">{result_text[winner].replace('\\n', '<br>')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>")
    if st.button("🔄 다시 해보기"):
        st.session_state.score = {"이재명": 0, "김문수": 0, "이준석": 0}
        st.session_state.q_index = 0
        st.session_state.answers = []
        st.session_state.finished = False
        st.experimental_rerun()
