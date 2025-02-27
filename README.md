# 🧠 Streamlit Elo Dashboard

## 📌 프로젝트 개요
** "Streamlit Elo Dashboard" **는 LLM (Large Language Model)의 응답을 클러스터링하고,  
Elo Rating System을 활용하여 피드백 기반 점수 변화를 시각적으로 분석하는 웹 애플리케이션입니다.

이 대시보드는 다음과 같은 기능을 제공합니다:
- **📊 클러스터별 데이터 분포(t-SNE 시각화)**
- **📈 Elo 점수 변화 분석 및 시각화**
- **🔍 클러스터별 대표 질문 및 응답 확인**
- **📂 새로운 데이터 업로드 및 비교 분석**
- **⚠️ Elo 점수 하락 클러스터 분석 및 개선 필요성 파악**
- **📥 혼합된 데이터 다운로드 기능**

---

## 🚀 **데모**
Streamlit Cloud에서 앱을 실행하여 대시보드를 직접 확인할 수 있습니다.

🔗 **[Live Demo]([https://your-app-name.streamlit.app](https://wellcheckai-elo-rating-dashboard.streamlit.app/))**

---

## 📂 **설치 및 실행 방법**
### 1️⃣ **로컬 환경에서 실행**
```bash
# 1️⃣ 프로젝트 클론
git clone https://github.com/your-username/streamlit-elo-dashboard.git
cd streamlit-elo-dashboard

# 2️⃣ 가상환경 생성 (선택)
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows

# 3️⃣ 필수 패키지 설치
pip install -r requirements.txt

# 4️⃣ Streamlit 실행
streamlit run streamlit_app.py
