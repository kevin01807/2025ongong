# voteforme.py
import streamlit as st

st.set_page_config(page_title="President 101", page_icon="🗳️", layout="centered")

# -----------------------------
# 질문 & 결과 데이터
# -----------------------------
QUESTIONS = [
    {
        "q": "내 방 월세 진짜 너무 비싸..\n집 문제 어떻게 해결해야 한다고 생각해?",
        "opts": [
            ("나라가 임대주택을 많이 지어야 해. 월세 걱정 없는 사회!", "이재명"),
            ("사람들이 집 살 수 있게 규제를 풀어주면 자연스럽게 해결돼", "김문수"),
            ("청년 기숙사나 월세 지원 같은 게 현실적이지 않을까?", "이준석"),
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
        "q": "정치인들 싸움 보면 답답해… 정치 어떻게 바뀌면 좋겠어?",
        "opts": [
            ("검·의 권력도 투명하게 바꿔야!", "이재명"),
            ("나라 어지럽히는 세력부터 정리!", "김문수"),
            ("국회 구조 자체를 뜯어고치자!", "이준석"),
        ],
    },
    {
        "q": "모든 국민에게 매달 일정 금액을 주는 ‘기본소득’이 생긴다면 어때?",
        "opts": [
            ("너무 좋지! 기회가 생기잖아.", "이재명"),
            ("나라가 다 챙겨주는 건 무리야.", "김문수"),
            ("청년 창업·대출 지원이 더 현실적!", "이준석"),
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
# 세션 초기화
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = 0
    st.session_state.score = {"이재명": 0, "김문수": 0, "이준석": 0}
    st.session_state.finished = False

# -----------------------------
# 질문 출력 함수
# -----------------------------
def show_question(idx):
    qdata = QUESTIONS[idx]
    st.markdown("## 🗳️ President 101")
    st.markdown("#### 당신의 대통령에게 투표하세요!\n")
    st.markdown("---")
    st.markdown(f"### Q{idx + 1}. {qdata['q']}")

    col1, col2, col3 = st.columns(3)
    for i, (label, cand) in enumerate(qdata["opts"]):
        if [col1, col2, col3][i].button(label, key=f"q{idx}_opt{i}"):
            st.session_state.score[cand] += 1
            st.session_state.page += 1
            if st.session_state.page >= len(QUESTIONS):
                st.session_state.finished = True
            st.experimental_rerun()

# -----------------------------
# 결과 페이지
# -----------------------------
def show_result():
    score = st.session_state.score
    winner = max(score, key=score.get)
    result = RESULTS[winner]

    st.markdown("## 🎉 테스트 결과")
    st.markdown(f"### {result['emoji']} {result['title']}")
    st.markdown(result["desc"])
    st.markdown("---")
    st.markdown("#### 🧮 선택 비율")
    for cand, s in score.items():
        st.markdown(f"- {cand}: {s}")

    if st.button("🔄 다시 하기"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

# -----------------------------
# 실행
# -----------------------------
if not st.session_state.finished:
    show_question(st.session_state.page)
else:
    show_result()
