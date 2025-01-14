import japanize_matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from recommend_algo import TopicBasedRecommender

st.set_page_config(
    page_title="大学院授業推薦システム",
    layout="wide",  # 横幅いっぱい
    initial_sidebar_state="expanded",  # サイドバーを展開した状態で開始
)

with st.sidebar:
    st.title("問題発見と解決")
    st.write("このアプリは、大学院の授業を推薦するためのアプリです。")
    
st.title("大学院授業推薦システム")

# タブの作成
tab1, tab2, tab3 = st.tabs(["データ入力", "最適化実行", "結果の可視化"])

# タブ1: データ入力
with tab1:
    with st.form("upload_form"):
        # ファイルアップロード
        report = st.file_uploader(
            "成績表をアップロード",
            type=['csv', 'xlsx', 'xls'],  # 受け付けるファイルタイプ
            help="CSVファイルまたはExcelファイル（.xlsx, .xls）のみアップロード可能。最大200MB"
        )

        # アップロードされた場合の処理
        if report is not None:
            # ファイルタイプのチェック
            if report.name.endswith('.csv'):
                st.write("CSVファイルがアップロードされました。")
                report_df = pd.read_csv(report)

            elif report.name.endswith(('.xlsx', '.xls')):
                st.write("Excelファイルがアップロードされました。")
                report_df = pd.read_excel(report)

            else:
                st.error("対応していないファイル形式です。"
                         )
        submitted = st.form_submit_button("データアップロード")
        
        if submitted and report:
            st.session_state['report_df'] = report_df
            st.success("データアップロード完了！")
            st.write(report_df)

# タブ2: 最適化実行
with tab2:
    if 'report_df' in st.session_state:
        if st.button("大学院授業の推薦を実行"):
            st.write("#### 最適化結果")
            recommender = TopicBasedRecommender(st.session_state['report_df'], num_topics=30)
            number_of_recommendations_by_topic = recommender.decide_number_of_recommendations_by_topic()
            
            for topic, n_r in number_of_recommendations_by_topic:
                if n_r == 0:
                    continue
                temp, your_course = recommender.execute_recommendation(topic, n_r)

                # カード風表示
                st.markdown(
                    f"""
                    <div style="padding: 1.5rem; background-color: #f9f9f9; border-radius: 10px; margin-bottom: 1rem">
                        <h2 style="color: #007BFF; margin-top: 0; font-size: 30px;">トピック: {topic}</h2>
                        <h4 style="color: #007BFF; margin-top: 0; font-size: 20px;">
                            <strong>推定関心度：</strong>{recommender.user_profile_percent[topic]}％
                        </h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # シラバスデータフレーム
                st.dataframe(
                    temp,
                    column_config={
                        "シラバス": st.column_config.LinkColumn(
                            "シラバス",
                            display_text="クリックしてシラバスを表示",
                        )
                    },
                )

                # トピック重要ワードと専門用語を囲む
                st.markdown(
                    f"""
                    <div style="padding: 1rem; background-color: #eaf7ff; border-radius: 10px; margin-top: 1rem;">
                        <p><strong>トピック重要ワード：</strong></p>
                        <p>{"、".join(keyword for keyword in recommender.topic_keywords[topic][0])}</p>
                        <p><strong>トピック専門用語：</strong></p>
                        <p>{"、".join(keyword for keyword in recommender.topic_keywords[topic][1])}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                #st.write(recommender.df_grad)
                st.markdown(
                    f"""
                    <div style="padding: 1rem; background-color: #eaf7ff; border-radius: 10px; margin-top: 1rem;">
                        <p><strong>このトピックは以下のような授業に基づいています：</strong></p>
                        <p>{"、".join(keyword for keyword in your_course)}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # 区切り線
                st.markdown("<hr style='border: 1px solid #ddd;'>", unsafe_allow_html=True)

    else:
        st.error("データを先にアップロードしてください。")

# タブ3: 結果の可視化
with tab3:
    if 'report_df' in st.session_state and 'solution_df' in st.session_state:
        # solution_df と report_df をマージ
        merge_df = st.session_state['solution_df'].merge(st.session_state['report_df'], on='student_id')
        grouped = merge_df.groupby('car_id')

        for car_id, group in grouped:
            # 男女比データの準備
            gender_counts = group['gender'].value_counts()
            gender_labels = {0: "男性", 1: "女性"}
            gender_counts.index = [gender_labels[idx] for idx in gender_counts.index]

            # 学年比データの準備
            grade_counts = group['grade'].value_counts()
            grade_labels = {1: "１年生", 2: "２年生", 3: "３年生", 4: "４年生"}
            grade_counts.index = [grade_labels[idx] for idx in grade_counts.index]

            draw_pie_charts(gender_counts, grade_counts, car_id)
    else:
        st.error("データを先にアップロードして、最適化を実行してください。")