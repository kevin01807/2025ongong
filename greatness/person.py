# app.py
# 고등학생용 ‘대통령 성향 테스트’ Streamlit 웹앱
import streamlit as st

st.set_page_config(page_title="대통령 성향 테스트", page_icon="🗳️", layout="centered")

# =============================
# 0. 꾸미기용 기본 이미지 URL
# =============================
VOTE_ICON = "https://cdn-icons-png.flaticon.com/512/1042/1042391.png"  # 투표 상자 아이콘
CAND_ICON = {
    "이재명": "https://cdn-icons-png.flaticon.com/512/4140/4140047.png",  # 사람 아이콘 예시
    "김문수": "https://cdn-icons-png.flaticon.com/512/4140/4140048.png",
    "이준석": "https://cdn-icons-png.flaticon.com/512/4140/4140051.png",
}

# -----------------------------
# 1. 질문·결과 데이터
# -----------------------------
QUESTIONS = [
    {
        "q": "모든 국민에게 매달 일정 금액을 주는 ‘기본소득’이 생긴다면 어때?",
        "opts": [
            ("너무 좋지! 기회가 생기잖아.", "이재명"),
            ("나라가 다 챙겨주는 건 무리야.", "김문수"),
            ("청년 창업·대출 지원이 더 현실적!", "이준석"),
        ],
    },
    {
        "q": "집값·월세가 너무 비싸… 해결 방법은 뭐가 좋을까?",
        "opts": [
            ("공공임대주택을 왕창 지어야 해.", "이재명"),
            ("규제 풀어서 시장이 알아서 해결!", "김문수"),
            ("청년 기숙사·월세 지원이 현실적.", "이준석"),
        ],
    },
    {
        "q": "의사 파업 등 의료 문제, 어찌 해결할까?",
        "opts": [
            ("공공의료 늘리고 충분히 논의!", "이재명"),
            ("빨리 복귀시켜서 정상화부터!", "김문수"),
            ("보건부 따로 만들어 효율적으로!", "이준석"),
        ],
    },
    {
        "q": "정치인들 싸우는 거 보면 답답해… 어떻게 바뀌면 좋겠어?",
        "opts": [
            ("검·의 권력도 투명하게 바꿔야!", "이재명"),
            ("나라 어지럽히는 세력부터 정리!", "김문수"),
            ("국회 구조 자체를 뜯어고치자!", "이준석"),
        ],
    },
    {
        "q": "외교·안보에서 가장 중요한 건?",
        "opts": [
            ("대화와 균형 외교가 좋아.", "이재명"),
            ("강한 군대가 나라를 지킨다!", "김문수"),
            ("상황 따라 실용적으로 가야지.", "이준석"),
        ],
    },
    {
        "q": "경제 살리려면 뭐가 최우선?",
        "opts": [
            ("복지를 늘려 기회를 보장!", "이재명"),
            ("기업이 활발해야 일자리도!", "김문수"),
            ("청년 창업이 자유로워야!", "이준석"),
        ],
    },
    {
        "q": "같이 일할 정치인을 뽑는다면?",
        "opts": [
            ("사람 사는 세상 만들 사람", "이재명"),
            ("나라 근간을 바로 세울 사람", "김문수"),
            ("공정을 중시하는 젊은 리더", "이준석"),
        ],
    },
]

RESULTS = {
    "이재명": {
        "title": "당신은 ‘이재명’과 가장 비슷해요!",
        "desc": "- 복지 확대\n- 기본소득\n- 공공주택·평화 외교",
        "emoji": "🌱",
    },
    "김문수": {
        "title": "당신은 ‘김문수’와 가장 비슷해요!",
        "desc": "- 자유시장·작은 정부\n- 강경 안보\n- 전통적 가치",
        "emoji": "🛡️",
    },
    "이준석": {
        "title": "당신은 ‘이준석’과 가장 비슷해요!",
        "desc": "- 청년·기술 중심\n- 실용주의\n- 구조 개혁",
        "emoji": "🚀",
    },
}

# -----------------------------
# 2. 세션 초기화
# -----------------------------
if "submitted" not in st.session_state:
    st.session_state.submitted = False
    st.session_state.answers = {}
    st.session_state.score = {}

# -----------------------------
# 3. 헤더 + 아이콘
# -----------------------------
st.image(VOTE_ICON, width=80)
st.title("🗳️ 대통령 성향 테스트")
st.markdown("7개의 질문에 답하면, **당신과 가장 성향이 비슷한 대통령**을 찾아드립니다!")
st.markdown("---")

# -----------------------------
# 4. 질문 폼
# -----------------------------
with st.form("quiz_form"):
    for idx, item in enumerate(QUESTIONS, start=1):
        st.markdown(f"### Q{idx}. {item['q']}")
        choice = st.radio(
            label="",
            options=[opt[0] for opt in item["opts"]],
            key=f"q{idx}",
            index=0,
        )
        st.session_state.answers[f"q{idx}"] = choice
    submitted = st.form_submit_button("결과 보기")

# -----------------------------
# 5. 결과 계산
# -----------------------------
if submitted and not st.session_state.submitted:
    score = {"이재명": 0, "김문수": 0, "이준석": 0}
    for idx, item in enumerate(QUESTIONS, start=1):
        sel_text = st.session_state.answers[f"q{idx}"]
        for opt_text, cand in item["opts"]:
            if sel_text == opt_text:
                score[cand] += 1
                break
    winner = max(score, key=score.get)
    st.session_state.submitted = True
    st.session_state.winner = winner
    st.session_state.score = score

# -----------------------------
# 6. 다시 시작 콜백
# -----------------------------
def reset_quiz():
    """세션 변수 초기화 후 페이지 리로드"""
    for k in list(st.session_state.keys()):
        if k.startswith("q"):
            del st.session_state[k]
    st.session_state.submitted = False
    st.session_state.answers = {}
    st.session_state.score = {}
    st.rerun()

# -----------------------------
# 7. 결과 페이지
# -----------------------------
if st.session_state.get("submitted"):
    result = RESULTS[st.session_state.winner]
    score = st.session_state.score

    st.markdown("---")
    # 후보 아이콘 + 제목
    st.image(CAND_ICON[st.session_state.winner], width=120)
    st.header(f"{result['emoji']} {result['title']}")
    st.markdown(result["desc"])
    st.markdown(
        f"**득표(?) 결과**  \n"
        f"- 이재명: {score['이재명']}  \n"
        f"- 김문수: {score['김문수']}  \n"
        f"- 이준석: {score['이준석']}"
    )
    st.button("🔄 다시 해보기", on_click=reset_quiz)
