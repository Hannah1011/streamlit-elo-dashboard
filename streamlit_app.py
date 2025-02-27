import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.colors
from sklearn.manifold import TSNE
from ast import literal_eval
import random

# # 📂 기본 데이터 경로
# default_file_690 = "data/humanfeedback_690_elo_updated.csv"

# 📌 데이터 로드 함수
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df["embedding"] = df["embedding"].apply(literal_eval).apply(np.array)  # 임베딩 변환
    df["cluster"] = df["cluster"].astype(int)  # 클러스터 번호 정렬 문제 해결
    return df

# 📌 t-SNE 시각화 함수
def tsne_visualization(df, title):
    tsne = TSNE(n_components=2, perplexity=25, random_state=45, init="random")
    reduced_data = tsne.fit_transform(np.vstack(df["embedding"].values))
    df["tsne_x"], df["tsne_y"] = reduced_data[:, 0], reduced_data[:, 1]

    # unique_clusters = sorted(df["cluster"].unique())
    # color_palette = px.colors.qualitative.Plotly 
    
     # ✅ **12개 클러스터에 대한 고유 색상 설정**
    custom_colors = [
   "#E63946", "#F77F00", "#FFD700", "#8B4513", "#2A9D8F", "#00A8E8",
    "#1D3557", "#6A0DAD", "#A29BFE", "#FAD0EF", "#D62828", "#708090"
    ]

    # ✅ 클러스터 개수 확인
    unique_clusters = sorted(df["cluster"].unique())
    num_clusters = len(unique_clusters) 
    color_palette = custom_colors[:num_clusters]
    
    fig = px.scatter(df, x="tsne_x", y="tsne_y", color=df["cluster"].astype(str), 
                     hover_data={"query": True, "theme_of_cluster": True}, title=title,
                     color_discrete_sequence=color_palette)

    return fig
# 📌 Ink의 법칙을 활용한 막대그래프 (클러스터 Elo 점수)
def plot_elo_bar_chart(df, title):
    elo_avg = df.groupby("cluster")["cluster_elo"].mean().reset_index()

    # 📌 Ink의 법칙 적용
    fig = px.bar(elo_avg, x="cluster", y="cluster_elo", title=title, 
                 labels={"cluster": "클러스터", "cluster_elo": "Elo 점수"},
                 color="cluster_elo", color_continuous_scale="purples")

    fig.update_traces(marker=dict(line=dict(width=1, color='black')))
    return fig

# 📌 대표 질문 3개 가져오기 (랜덤)
def get_random_queries(df, cluster_id, top_n=3):
    queries = df[df["cluster"] == cluster_id]["query"].tolist()
    return random.sample(queries, min(top_n, len(queries))) if queries else ["(데이터 없음)"]

# ✅ **📌 Streamlit 대시보드 시작**
st.title("🧠 Human Feedback Elo Dashboard")

# 📌 Step 1: 690개 데이터 로드

# 기존 데이터 업로드 기능 추가
st.subheader("📂 기존 데이터 업로드")
uploaded_base_file = st.file_uploader("📥 기존 데이터 파일을 업로드하세요. (.csv)", type=["csv"])
if uploaded_base_file:
    df_690 = load_data(uploaded_base_file)
    # 📊 t-SNE 시각화
    st.subheader("🔍 클러스터 분포 (690개 데이터)")
    st.plotly_chart(tsne_visualization(df_690, "t-SNE Visualization of 690 Data"))

    # 📌 **Step 2: 클러스터 정보 표시**
    st.subheader("🔍 클러스터 정보")
    cluster_selected = st.selectbox("**📌 클러스터 선택**", sorted(df_690["cluster"].unique()))  # 정렬된 순서 적용

    # ✅ 클러스터 변경 시 대표 질문 즉시 업데이트
    if "last_selected_cluster" not in st.session_state or st.session_state.last_selected_cluster != cluster_selected:
        st.session_state.random_queries = df_690[df_690["cluster"] == cluster_selected][["query", "answer"]].sample(min(3, len(df_690[df_690["cluster"] == cluster_selected]))).reset_index(drop=True)
        st.session_state.last_selected_cluster = cluster_selected  # 현재 선택한 클러스터 저장

    # 📌 클러스터 테마 출력 (중앙 정렬 & 간격 조정)
    st.markdown(f"""
    <div style="
        border: 2px solid #6A0DAD; 
        padding: 8px 15px; 
        border-radius: 8px; 
        background-color: #F8F5FC; 
        text-align: center;
        display: flex; 
        flex-direction: column; 
        align-items: center; 
        justify-content: center;
    ">
        <p style="font-size: 16px; font-weight: bold; color:black; margin: 0;">
        ✔️ 클러스터 테마: <span style="color:#6A0DAD;">{df_690[df_690["cluster"] == cluster_selected]["theme_of_cluster"].iloc[0]}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 📌 대표 질문 & 응답 간격 추가
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    st.write("**📌 대표 질문 & 응답 (랜덤 3개)**")
    st.table(st.session_state.random_queries)

    # 🔄 "다시 뽑기" 버튼 추가 (같은 클러스터 내에서 랜덤 샘플링)
    if st.button("🔄 다시 뽑기"):
        st.session_state.random_queries = df_690[df_690["cluster"] == cluster_selected][["query", "answer"]].sample(min(3, len(df_690[df_690["cluster"] == cluster_selected]))).reset_index(drop=True)
        st.rerun()

    # 📊 **클러스터별 Elo 점수**
    st.subheader("🔍  클러스터별 Elo 점수")
    st.plotly_chart(plot_elo_bar_chart(df_690, "Elo 점수 분포 (690개 데이터)"))

    # 📂 **Step 3: 새로운 데이터 업로드 기능**
    st.subheader("📂 새로운 데이터 업로드")
    uploaded_file = st.file_uploader("📥 데이터 파일을 업로드하세요. (.csv)", type=["csv"])

    if uploaded_file:
        # 📌 221개 데이터 로드
        df_221 = load_data(uploaded_file)

        # 📊 t-SNE 시각화 (업로드 후)
        df_combined = pd.concat([df_690, df_221], ignore_index=True)
        st.subheader("🔍  클러스터 분포 (업로드 후 911개 데이터)")
        st.plotly_chart(tsne_visualization(df_combined, "t-SNE Visualization of 911 Data"))

        # 📊 Elo 점수 변화 시각화 개선
        st.subheader("📊 Elo 점수 변화")

        # Elo 점수 차이 계산
        df_221["elo_change"] = df_221["cluster_elo"] - df_221["elo_old"]
        elo_comparison = df_221.groupby("cluster")[["elo_old", "cluster_elo", "elo_change"]].mean().reset_index()

        # ✅ 네이밍 개선
        elo_comparison = elo_comparison.rename(columns={
            "elo_old": "기존 클러스터 Elo 점수",
            "cluster_elo": "업데이트된 클러스터 Elo 점수",
            "elo_change": "Elo 변화량"
        })

        # 📊 **Elo 변화 그래프 (점과 선)**
        fig = px.line(elo_comparison, x="cluster", y=["기존 클러스터 Elo 점수", "업데이트된 클러스터 Elo 점수"],
                    title="Elo 점수 변화 (업로드 후)", 
                    labels={"value": "Elo 점수", "variable": "Elo 유형"},
                    color_discrete_sequence=["#D8BFD8", "#6A0DAD"],  
                    markers=True)

        fig.update_traces(mode="lines+markers")  

        st.plotly_chart(fig)

        # 🔔 **Elo 점수 변화 클러스터 메시지**
        st.subheader("📊 Elo 점수 변동 현황")

        decreased_clusters = elo_comparison[elo_comparison["Elo 변화량"] < 0]
        increased_clusters = elo_comparison[elo_comparison["Elo 변화량"] > 0]

        if not decreased_clusters.empty:
            st.warning("📉 Elo 점수가 하락한 클러스터입니다. 개선이 필요할 수 있으니 검토해보세요!")
            for _, row in decreased_clusters.iterrows():
                st.write(f"- 클러스터 {int(row['cluster'])} ({df_221[df_221['cluster'] == row['cluster']]['theme_of_cluster'].iloc[0]}) → {row['Elo 변화량']:.2f} 감소")

        if not increased_clusters.empty:
            st.success("📈 Elo 점수가 상승한 클러스터입니다. 긍정적인 피드백을 받았으니 강점을 유지하면서 더 발전시켜보세요! ")
            for _, row in increased_clusters.iterrows():
                st.write(f"- 클러스터 {int(row['cluster'])} ({df_221[df_221['cluster'] == row['cluster']]['theme_of_cluster'].iloc[0]}) → {row['Elo 변화량']:.2f} 증가")

        # 📌 진짜 Elo 점수가 평균적으로 하락한 클러스터만 선택
        true_decreased_clusters = decreased_clusters["cluster"].unique()  # 하락한 클러스터 리스트
        
        if len(true_decreased_clusters) > 0:
            # 📌 하락한 클러스터 중 하나 선택
            cluster_decrease_selected = st.selectbox(
                "하락한 클러스터 선택", sorted(true_decreased_clusters)
            )
        
            # 📌 선택한 클러스터의 문제 응답 데이터 보기
            st.write(f"**📌 클러스터 {int(cluster_decrease_selected)} - Elo 점수 하락 사유 분석**")
        
            # **Elo 점수 하락 클러스터에서 query, answer, reason, reason_detail 만 보기**
            columns_to_show = ["query", "answer", "reason", "reason_detail"]
            filtered_decreased_data = df_combined[df_combined["cluster"] == cluster_decrease_selected][columns_to_show]
        
            # 📌 **데이터 표 출력**
            st.dataframe(filtered_decreased_data)
        
        else:
            st.success("✅ Elo 점수가 하락한 클러스터가 없습니다!")
        
        # 🔍 **클러스터별 데이터 필터링 (690 + 221 합친 데이터)**
        st.subheader("🔍 특정 클러스터 데이터 보기")
        cluster_filter = st.selectbox("클러스터 선택", sorted(df_combined["cluster"].unique()))
        st.dataframe(df_combined[df_combined["cluster"] == cluster_filter])
        
        # 📥 **혼합된 데이터 다운로드 버튼 추가**
        st.subheader("📥 혼합 데이터 다운로드 (690 + 221 합친 데이터)")

        # ✅ 저장할 컬럼만 선택
        columns_to_keep = ["coach_no", "name", "quality", "created_at", "query", "answer", 
                        "combined", "guide_index", "reason", "reason_detail", 
                        "cluster", "cluster_avg_quality", "theme_of_cluster"]

        df_combined_filtered = df_combined[columns_to_keep]  # 필요한 컬럼만 선택

        # ✅ 인코딩 문제 해결 → `utf-8-sig`로 변경
        csv_combined = df_combined_filtered.to_csv(index=False, encoding="utf-8-sig")

        # ✅ Streamlit의 `download_button` 사용
        st.download_button("📥 혼합 데이터 다운로드", csv_combined, "combined_elo_data.csv", "text/csv", key="download_csv")
