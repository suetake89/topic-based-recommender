import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from recommend_algo import OptimizeClasses, TopicBasedRecommender

st.set_page_config(
    page_title="大学院授業推薦システム",
    layout="wide",  # 横幅いっぱい
    initial_sidebar_state="expanded",  # サイドバーを展開した状態で開始
)

with st.sidebar:
    st.title("問題発見と解決")
    st.write("")
    st.write("🔎システム情報工学研究群の授業を推薦するアプリです。")
    st.write("")
    st.write("🔎タブ1：成績表データの入力")
    st.write("")
    st.write("🔎タブ2：興味があると思われるトピックに基づいて授業を推薦")
    st.write("")
    st.write("🔎タブ3：選好とトピックの特徴分析")
    st.write("")
    st.write("🔎タブ4：選択必修科目のおすすめスケジュール")
    st.write("")
    st.write("🔎タブ5：キーワードから授業を検索")
    st.write("")
    st.write("🔎気になった授業はすぐシラバスへ飛べるので是非面白そうな授業を探してみてください！")
    st.write("")
    st.write("※授業は2024年度")
    
st.title("大学院授業推薦システム")

# タブの作成
tab1, tab2, tab3, tab4, tab5 = st.tabs(["1.データ入力", "2.推薦システム", "3.トピックの可視化", '4.おすすめスケジュール', "5.授業検索"])

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
    
    st.write("")
    st.write("もしくは、仮想データを使用してください。")
    with st.form("upload_form_2"):
        submitted = st.form_submit_button("仮想データを使用")
        if submitted:
            df = pd.read_csv('成績データ.csv')
            df = df[df['総合評価']!='履修中']
            st.write("修正が必要な場合は、以下のテーブルを編集してください。")
            report_df = st.data_editor(df)
            st.session_state['report_df'] = report_df
            st.success("データアップロード完了！")

# タブ2: 最適化実行
with tab2:
    if 'report_df' in st.session_state:
        #input_value = st.text_input("トピック数に入力してください:", "15")
        #num_topics = int(input_value) if input_value else 15
        if st.button("大学院授業の推薦を実行"):
            num_topics = 15
            st.write("#### 最適化結果")
            recommender = TopicBasedRecommender(st.session_state['report_df'], num_topics=num_topics)
            recommender.create_lda_model()
            keywords_list = recommender.get_keywords_list()
            topic_keywords = recommender.get_topic_keywords(keywords_list)
            recommender.assign_info_to_courses()
            recommender.assign_topic_to_courses()
            number_of_recommendations_by_topic = recommender.decide_number_of_recommendations_by_topic()
            
            st.session_state['recommender'] = recommender
            
            for topic_id, n_r in number_of_recommendations_by_topic:
                topic = recommender._number_to_char(topic_id)
                if n_r == 0:
                    continue
                temp, your_course = recommender.execute_recommendation(topic)

                # カード風表示
                st.markdown(
                    f"""
                    <div style="padding: 1.5rem; background-color: #DEEBF2; border-radius: 10px; margin-bottom: 1rem">
                        <h2 style="color: #3e42b6; margin-top: 0; font-size: 30px;">トピック: {topic}</h2>
                        <h4 style="color: #3e42b6; margin-top: 0; font-size: 20px;">
                            <strong>推定関心度：</strong>{recommender.user_profile_percent[topic_id]}％
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
                            display_text="シラバスを表示",
                        )
                    },
                )

                # トピック重要ワードと専門用語を囲む
                st.markdown(
                    f"""
                    <div style="padding: 1rem; background-color: #DEEBF2; color: #000000; border-radius: 10px; margin-top: 1rem;">
                        <p><strong>トピック重要ワード：</strong></p>
                        <p>{"、".join(keyword for keyword in topic_keywords[topic_id][0])}</p>
                        <p><strong>トピック専門用語：</strong></p>
                        <p>{"、".join(keyword for keyword in topic_keywords[topic_id][1])}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                #st.write(recommender.df_grad)
                st.markdown(
                    f"""
                    <div style="padding: 1rem; background-color: #DEEBF2; color: #000000; border-radius: 10px; margin-top: 1rem;">
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
        recommender = st.session_state['recommender']
        
        st.title("トピック分布の可視化")
        st.write("")
        # グラフ1
        st.subheader("ユーザー選好のトピック分布")
        fig_user = recommender.plot_topic_distribution_of_user_profile()
        st.plotly_chart(fig_user)

        # グラフ2
        st.subheader("主専攻ごとのトピック分布")
        fig_social = recommender.plot_topic_distribution_of_social()
        st.plotly_chart(fig_social)

        # グラフ3
        st.subheader("学位プログラムごとのトピック分布")
        fig_grad = recommender.plot_topic_distribution_of_grad()
        st.plotly_chart(fig_grad)
        
        keywords_list = recommender.get_keywords_list()
        topic_keywords =  recommender.get_topic_keywords(keywords_list)
        
        st.write("## 各トピックの授業一覧")
        topic_list = [recommender._number_to_char(topic_id) for topic_id in range(recommender.num_topics)]
        option = st.selectbox("トピックを選択してください。", topic_list)
        
        st.write("#### 学類の授業")
        temp = recommender.df_social_courses[recommender.df_social_courses['トピック']==option]
        temp = temp[['科目番号', '授業科目名', '主専攻', 'キーワード', 'シラバス']]
        # シラバスデータフレーム
        st.dataframe(
            temp,
            column_config={
                "シラバス": st.column_config.LinkColumn(
                    "シラバス",
                    display_text="シラバスを表示",
                )
            },
        )
        st.write("#### 院の授業")
        temp = recommender.df_grad_courses[recommender.df_grad_courses['トピック']==option]
        temp = temp[['科目番号', '授業科目名', '学位プログラム', 'キーワード', 'シラバス']]
        # シラバスデータフレーム
        st.dataframe(
            temp,
            column_config={
                "シラバス": st.column_config.LinkColumn(
                    "シラバス",
                    display_text="シラバスを表示",
                )
            },
        )
    
        st.write("## 各トピックの情報")
        for topic_id in range(recommender.num_topics):
            topic = recommender._number_to_char(topic_id)
            #if n_r == 0:
            #    continue
            temp, your_course = recommender.execute_recommendation(topic)

            # トピック重要ワードと専門用語を囲む
            st.markdown(
                f"""
                <div style="padding: 1rem; background-color: #DEEBF2; color: #000000; border-radius: 10px; margin-top: 1rem;">
                    <p style="font-size: 25px;"><strong>トピック: {topic}</strong></p>
                    <p><strong>トピック重要ワード：</strong></p>
                    <p>{"、".join(keyword for keyword in topic_keywords[topic_id][0])}</p>
                    <p><strong>トピック専門用語：</strong></p>
                    <p>{"、".join(keyword for keyword in topic_keywords[topic_id][1])}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.error("データを先にアップロードして、推薦システムを実行してください。")
        
with tab4:
    if 'report_df' in st.session_state and 'recommender' in st.session_state:
        st.markdown("")
        option = st.selectbox(
                "学位プログラムを選択してください。",
                ("社会工学学位プログラム", "サービス工学学位プログラム", "リスク・レジリエンス工学学位プログラム", "情報理工学位プログラム", "知能機能システム学位プログラム", "構造エネルギー工学学位プログラム"),
            )
        if option == "社会工学学位プログラム":
            recommender = st.session_state['recommender']
            df = recommender.df_grad_courses
            opt = OptimizeClasses(df)
            df_schedule = opt.optimize()
            st.markdown("")
            st.write("### 推薦された選択必修の例（２４単位以上）")
            st.write("")
            
            st.write("#### 専門基礎科目かつ社会工学　（６単位以上）")
            temp = df_schedule[(df_schedule['学位プログラム']=='社会工学関連科目') & (df_schedule['科目区分']==0)]
            temp = temp[['科目番号', '授業科目名', '時間割', '単位数', 'トピック', 'おすすめ度', 'シラバス']]
            st.dataframe(
                    temp,
                    column_config={
                        "シラバス": st.column_config.LinkColumn(
                            "シラバス",
                            display_text="シラバスを表示",
                        )
                    },
                )
            
            st.write("#### 専門基礎科目かつ社会工学以外　（２単位以上）")
            temp = df_schedule[(df_schedule['学位プログラム']!='社会工学関連科目') & (df_schedule['科目区分']==0)]
            temp = temp[['科目番号', '授業科目名', '時間割', '単位数', 'トピック', 'おすすめ度', 'シラバス']]
            st.dataframe(
                    temp,
                    column_config={
                        "シラバス": st.column_config.LinkColumn(
                            "シラバス",
                            display_text="シラバスを表示",
                        )
                    },
                )
            
            st.write("#### 専門科目かつ社会工学　（１０単位以上）")
            temp = df_schedule[(df_schedule['学位プログラム']=='社会工学関連科目') & (df_schedule['科目区分']!=0)]
            temp = temp[['科目番号', '授業科目名', '時間割', '単位数', 'トピック', 'おすすめ度', 'シラバス']]
            st.dataframe(
                    temp,
                    column_config={
                        "シラバス": st.column_config.LinkColumn(
                            "シラバス",
                            display_text="シラバスを表示",
                        )
                    },
                )
            st.write("#### 専門科目かつ社会工学以外　（０単位以上）")
            temp = df_schedule[(df_schedule['学位プログラム']!='社会工学関連科目') & (df_schedule['科目区分']!=0)]
            if not temp.empty:
                temp = temp[['科目番号', '授業科目名', '時間割', '単位数', 'トピック', 'おすすめ度', 'シラバス']]
                st.dataframe(
                        temp,
                        column_config={
                            "シラバス": st.column_config.LinkColumn(
                                "シラバス",
                                display_text="シラバスを表示",
                            )
                        },
                    )
            else:
                st.write("該当する科目はありません。")
            st.markdown("")
            st.info(
                    """
                    ※時間割が被らないように選択されています。
                    卒業要件を満たすかは各自でもご確認ください。損害の責任は負いかねます。
                    """
                )


        else:
            st.write("すみません。まだ対応していません。")
    else:
        st.error("データを先にアップロードして、推薦システムを実行してください。")

with tab5:
    st.title("システム情報工学研究群・KDB")
    st.write("")
    if 'recommender' in st.session_state:
        recommender = st.session_state['recommender']
    else:
        report_df = pd.read_csv('成績データ.csv')
        recommender = TopicBasedRecommender(report_df, num_topics=15)
    report_df = pd.read_csv('成績データ.csv')
    keywords_list = recommender.get_keywords_list()
    recommender.assign_info_to_courses()
    opt = OptimizeClasses(recommender.df_grad_courses)
    m0, m1, m2, m3, m4, m5 = st.columns((1, 1, 1, 1, 1, 1))
    proglam = m0.selectbox('学位プログラムを選択してください:', ['指定なし'] + opt.df['学位プログラム'].unique().tolist())
    season = m1.selectbox('学期を選択してください:', ['指定なし'] + ['春', '秋', '春季休業中', '秋C春季休業中', '通年'])
    module = m2.selectbox('モジュールを選択してください:', ['指定なし'] + ['A', 'B', 'C', '集中'])
    week = m3.selectbox('曜日を選択してください:', ['指定なし'] + ['月', '火', '水', '木', '金'])
    period = m4.selectbox('時限を選択してください:', ['指定なし'] + ['1', '2', '3', '4', '5', '6'])
    keyword = m5.selectbox('キーワードを入力してください:', ['指定なし'] + keywords_list)
    select_list = [proglam, season, module, week, period, keyword]
    temp = opt.df.copy()
    for i, select in enumerate(select_list):
        if select != '指定なし':
            if i == 0:
                temp = temp[temp['学位プログラム']==select]
            elif i in [1, 2, 3, 4]:
                temp = temp[temp['時間割'].str.contains(select, na=False)]
            else:
                if sum(temp.apply(lambda x: select in x['キーワード'], axis=1)) != 0:
                    temp = temp[temp.apply(lambda x: select in x['キーワード'], axis=1)]
                else:
                    temp = pd.DataFrame()
    
    if temp.empty:
        st.write("該当する科目はありません。")
    else:
        temp = temp[['科目番号', '授業科目名', '時間割', '単位数', '科目区分名', 'キーワード', 'シラバス']]
        st.dataframe(
                temp,
                column_config={
                    "シラバス": st.column_config.LinkColumn(
                        "シラバス",
                        display_text="シラバスを表示",
                    )
                },
            )
    st.info(
        """
        ※こちらでは主に選択必修となる、研究群共通科目群を表示します。
        [「大学院 履修方法・修了要件」はここから参照できます（2024年版）](https://www.tsukuba.ac.jp/education/g-courses-handbook/2024rishu.html)
        """
    )