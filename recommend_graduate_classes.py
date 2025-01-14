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
    st.info(
        """
        [このtwinsのリンク](https://twins.tsukuba.ac.jp/campusweb/campusportal.do?page=main&tabId=si)から成績表をダウンロードできます。  
        一番下までスクロールし、「ダウンロード」をクリックしてください。  
        ファイル形式と文字コードは初期設定のままで構いません。  
        """
    )
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
                df = pd.read_csv(report)

            elif report.name.endswith(('.xlsx', '.xls')):
                st.write("Excelファイルがアップロードされました。")
                df = pd.read_excel(report)

            else:
                st.error("対応していないファイル形式です。"
                         )
        submitted = st.form_submit_button("データアップロード")
        
        if submitted and report:
            df = df[df['総合評価']!='履修中']
            st.write("修正が必要な場合は、以下のテーブルを編集してください。")
            report_df = st.data_editor(df)
            st.session_state['report_df'] = report_df
            st.success("データアップロード完了！")

# タブ2: 最適化実行
with tab2:
    if 'report_df' in st.session_state:
        if st.button("大学院授業の推薦を実行"):
            st.write("#### 最適化結果")
            recommender = TopicBasedRecommender(st.session_state['report_df'], num_topics=30)
            recommender.create_lda_model()
            topic_keywords = recommender.get_topic_keywords()
            recommender.assign_topic_to_courses()
            number_of_recommendations_by_topic = recommender.decide_number_of_recommendations_by_topic()
            
            st.session_state['recommender'] = recommender
            
            for topic, n_r in number_of_recommendations_by_topic:
                if n_r == 0:
                    continue
                temp, your_course = recommender.execute_recommendation(topic)

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
                        <p>{"、".join(keyword for keyword in topic_keywords[topic][0])}</p>
                        <p><strong>トピック専門用語：</strong></p>
                        <p>{"、".join(keyword for keyword in topic_keywords[topic][1])}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                #st.write(recommender.df_grad)
                st.markdown(
                    f"""
                    <div style="padding: 1rem; background-color: #eaf7ff; border-radius: 10px; margin-top: 1rem;">
                        <p><strong>このトピックはあなたが履修した以下の授業に基づいています：</strong></p>
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
    if 'report_df' in st.session_state and 'recommender' in st.session_state:
        if st.button("推薦システムを可視化"):    
            visualizer = st.session_state['recommender']
            
            st.title("トピック分布の可視化")
            st.write("")
            # グラフ1
            st.subheader("ユーザー選好のトピック分布")
            fig_user = visualizer.plot_topic_distribution_of_user_profile()
            st.plotly_chart(fig_user)

            # グラフ2
            st.subheader("主専攻ごとのトピック分布")
            fig_social = visualizer.plot_topic_distribution_of_social()
            st.plotly_chart(fig_social)

            # グラフ3
            st.subheader("学位プログラムごとのトピック分布")
            fig_grad = visualizer.plot_topic_distribution_of_grad()
            st.plotly_chart(fig_grad)
            
            topic_keywords =  visualizer.get_topic_keywords()
            
            st.write("## 各トピックの情報")
            for topic in range(visualizer.num_topics):
                #if n_r == 0:
                #    continue
                temp, your_course = visualizer.execute_recommendation(topic)

                # トピック重要ワードと専門用語を囲む
                st.markdown(
                    f"""
                    <div style="padding: 1rem; background-color: #eaf7ff; border-radius: 10px; margin-top: 1rem;">
                        <p style="font-size: 25px;"><strong>トピック: {topic}</strong></p>
                        <p><strong>トピック重要ワード：</strong></p>
                        <p>{"、".join(keyword for keyword in topic_keywords[topic][0])}</p>
                        <p><strong>トピック専門用語：</strong></p>
                        <p>{"、".join(keyword for keyword in topic_keywords[topic][1])}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.error("データを先にアップロードして、推薦システムを実行してください。")