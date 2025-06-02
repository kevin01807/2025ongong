import streamlit as st

mbti_data = {
    "INTJ": {
        "설명": "독립적이고 전략적인 사고를 가진 사람. 장기적 목표를 잘 설정하고 추진함.",
        "특징": ["계획적", "분석적", "냉철함"],
        "추천 직업": ["전략기획자", "AI 연구원", "변호사"]
    },
    "INFP": {
        "설명": "이상주의적이고 감정이 풍부함. 가치관이 뚜렷하고 타인에게 공감 잘함.",
        "특징": ["공감능력", "창의성", "감수성"],
        "추천 직업": ["작가", "심리상담가", "디자이너"]
    },
    "ENTP": {
        "설명": "아이디어가 넘치고 새로운 걸 만드는 데 두려움이 없음. 토론을 즐김.",
        "특징": ["창의력", "즉흥성", "말빨"],
        "추천 직업": ["스타트업 창업자", "기획자", "마케터"]
    },
    "ISTJ": {
        "설명": "신중하고 책임감이 강함. 규칙과 체계를 중요시함.",
        "특징": ["성실함", "논리적", "현실적"],
        "추천 직업": ["회계사", "공무원", "엔지니어"]
    },
    # 필요하면 나머지 유형도 추가하면 됨
}

st.set_page_config(page_title="MBTI 분석기", page_icon="🧠")

st.title("🧬 MBTI 유형 분석기")
st.write("당신의 MBTI 유형을 선택하면 맞춤 분석을 보여줍니다.")

selected_mbti = st.selectbox("MBTI를 선택하세요", list(mbti_data.keys()))

data = mbti_data[selected_mbti]

st.header(f"🔍 {selected_mbti} 분석 결과")
st.subheader("🧠 설명")
st.write(data["설명"])

st.subheader("📌 성격 특징")
for t in data["특징"]:
    st.markdown(f"- {t}")

st.subheader("💼 추천 직업")
for job in data["추천 직업"]:
    st.markdown(f"- {job}")

st.markdown("---")
st.caption("MBTI는 참고용일 뿐, 진짜 중요한 건 너 자신을 아는 거임.")

