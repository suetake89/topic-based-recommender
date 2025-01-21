import ast
import math
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pulp
from gensim import corpora, models
from gensim.models import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity


class TopicBasedRecommender():
    def __init__(self, df_grad, num_topics=15):
        self.df_grad = df_grad

        # ===== データ読み込み =====
        self.df_combined = pd.read_excel("キーワード_拡大版.xlsx")
        self.df_0 = pd.read_excel("大学院専門基礎科目_df.xlsx")
        self.df_1 = pd.read_excel("社会工学類授業_df.xlsx")
        self.df_2 = pd.read_excel("大学院専門科目_df.xlsx")
        self.num_topics = num_topics

    def assign_info_to_courses(self, ratio=1):

        # ===== 関数定義 =====
        classification_rules_digit = {
            '0': '共通',
            '1': '社会工学関連科目',
            '2': 'サービス工学関連科目',
            '3': 'リスク・レジリエンス工学関連科目',
            '4': '情報理工関連科目',
            '5': '知能機能システム関連科目',
            '6': '構造エネルギー工学関連科目',
            '7': 'エンパワーメント情報学関連科目'
        }

        classification_rules_alpha = {
            'A': '社会工学関連科目',
            'B': 'サービス工学関連科目',
            'C': 'リスク・レジリエンス工学関連科目',
            'D': '情報理工関連科目',
            'E': '知能機能システム関連科目',
            'F': '構造エネルギー工学関連科目'
        }

        def classify_graduate_course(course_number):
            """科目番号から関連科目を分類する関数"""
            fourth_char = course_number[3]
            fifth_char = course_number[4]

            if fourth_char.isdigit():
                return classification_rules_digit.get(fifth_char, '予備')

            elif fourth_char.isalpha() and fourth_char.isupper():
                return classification_rules_alpha.get(fourth_char, '予備科目')

            return '分類不明'

        def classify_graduate_basis(class_num):
            if class_num[3] in ['0', '1', '2', '3', '4']:
                return 0 # 専門基礎
            elif class_num[3] in ['5', '6', '7', '8', '9']:
                return 1 # 専門基礎
            else:
              return 'その他'

        def classify_social_course(course_number):
            """
            科目番号から関連科目を分類する関数（社会工学関連の分類）
            """
            if not course_number.startswith("FH"):
                return "社会工学でない授業"

            third_char = course_number[2]  # 科目番号の3文字目
            fourth_char = course_number[3]  # 科目番号の4文字目

            if third_char == '2':
                return "社会経済システム" if fourth_char in ['4', '6', '7'] else "理工学群共通"

            elif third_char == '3':
                return "経営工学" if fourth_char in ['2', '3', '4'] else "理工学群共通"

            elif third_char == '4':
                return "都市計画" if fourth_char in ['6', '7', '8'] else "理工学群共通"

            return "その他"

        def return_syllabus_link(class_str):
            return f'https://kdb.tsukuba.ac.jp/syllabi/2024/{class_str}/jpn'

        # ===== 前処理 =====
        self.df_combined = self.df_combined.rename(columns={'キーワード': '1倍キーワード'})  # 'キーワード' 列をリスト形式に変換
        #self.df_combined['キーワード'] = self.df_combined[f'{ratio}倍キーワード'].apply(ast.literal_eval)  # 'キーワード' 列をリスト形式に変換
        self.df_combined['キーワード'] = self.df_combined[f'拡張キーワード'].apply(ast.literal_eval)
        self.df_0 = self.df_0.iloc[6:]  # df_0 の最初5行を削除（英語の授業を除外）

        # '授業科目名' 列をリスト化
        grad_course_list = pd.concat([self.df_0['授業科目名'], self.df_2['授業科目名']]).unique().tolist()
        social_course_list = self.df_1['授業科目名'].unique().tolist()

        # 大学院専門科目・大学院専門基礎科目の授業を抽出
        self.df_grad_courses = self.df_combined[self.df_combined['授業科目名'].isin(grad_course_list)].copy()
        self.df_social_courses = self.df_combined[self.df_combined['授業科目名'].isin(social_course_list)].copy()

        self.df_grad.rename(columns={'科目名 ': '科目名'}, inplace=True)

        # 評価を数値にマッピング
        grade_mapping = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 0, 'P': 3, 'F': 0}
        self.df_grad['総合評価'] = self.df_grad['総合評価'].replace(grade_mapping).astype('int')
        #self.df_grad['総合評価'] = self.df_grad.apply(lambda x: x['総合評価'] * int(x['単位数']))

        # 成績データから必要な列を抽出し、先頭の不要な行を除外
        self.df_grad = self.df_grad[['科目名', '総合評価']].iloc[9:]

        # '授業科目名' 列をデータフレームに変換して LEFT JOIN で結合
        self.df_result = self.df_combined[['授業科目名']]
        self.df_result = self.df_result.merge(self.df_grad[['科目名', '総合評価']], how='left', left_on='授業科目名', right_on='科目名')
        self.df_result['総合評価'] = self.df_result['総合評価'].fillna(0).astype(int)  # NaN を 0 に置換し整数型に変換
        self.df_result.drop(columns='科目名', inplace=True)  # 不要な列を削除

        self.df_0['シラバス'] = self.df_0['科目番号'].apply(return_syllabus_link)
        self.df_1['シラバス'] = self.df_1['科目番号'].apply(return_syllabus_link)
        self.df_2['シラバス'] = self.df_2['科目番号'].apply(return_syllabus_link)

        self.df_0['科目区分'] = [0]*len(self.df_0)
        self.df_2['科目区分'] = [1]*len(self.df_2)
            
        grad_subject_num_dict = (
              self.df_0.set_index('授業科目名')['科目番号'].astype(str).to_dict() |
              self.df_2.set_index('授業科目名')['科目番号'].astype(str).to_dict()
          )
        social_subject_num_dict = self.df_1.set_index('授業科目名')['科目番号'].astype(str).to_dict()
        
        social_subject_overview = self.df_1.set_index('授業科目名')['授業概要'].astype(str).to_dict()

        grad_subject_schedule = (
              self.df_0.set_index('授業科目名')['時間割'].astype(str).to_dict() |
              self.df_2.set_index('授業科目名')['時間割'].astype(str).to_dict()
          )

        grad_subject_unit = (
              self.df_0.set_index('授業科目名')['単位数'].astype(str).to_dict() |
              self.df_2.set_index('授業科目名')['単位数'].astype(str).to_dict()
          )

        # '関連授業分類' カラムを追加して分類を適用
        self.df_grad_courses['科目番号'] = self.df_grad_courses['授業科目名'].map(grad_subject_num_dict)
        self.df_grad_courses['時間割'] = self.df_grad_courses['授業科目名'].map(grad_subject_schedule)
        
        self.df_grad_courses['単位数'] = self.df_grad_courses['授業科目名'].map(grad_subject_unit).apply(lambda x: int(x[0]))
        self.df_grad_courses['実施学期'] = self.df_grad_courses['時間割'].apply(lambda x: x.split(" ")[0])
        self.df_grad_courses['曜時限'] = self.df_grad_courses['時間割'].apply(lambda x: x.split(" ")[1])
        self.df_grad_courses['学位プログラム'] = self.df_grad_courses['科目番号'].apply(classify_graduate_course)
        self.df_grad_courses['科目区分'] = self.df_grad_courses['科目番号'].apply(classify_graduate_basis)
        self.df_grad_courses['科目区分名'] =self.df_grad_courses['科目区分'].apply(lambda x: '専門科目' if x==1 else '専門基礎科目')
        self.df_grad_courses['シラバス'] = self.df_grad_courses['科目番号'].apply(return_syllabus_link)

        self.df_social_courses['科目番号'] = self.df_social_courses['授業科目名'].map(social_subject_num_dict)
        self.df_social_courses['主専攻'] = self.df_social_courses['科目番号'].apply(classify_social_course)
        self.df_social_courses['授業概要'] = self.df_social_courses['授業科目名'].map(social_subject_overview)
        self.df_social_courses['シラバス'] = self.df_social_courses['科目番号'].apply(return_syllabus_link)
    
    def create_lda_model(self):

        # ===== LDA モデルの作成 =====
        texts = self.df_combined['キーワード']  # トークン化されたキーワードのリスト
        self.dictionary = corpora.Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.lda_model = models.LdaModel(corpus=corpus, id2word=self.dictionary, num_topics=self.num_topics, random_state=42, passes=10)

        # ユーザーの成績評価リストを取得
        user_ratings = self.df_result['総合評価'].tolist()

        # ===== ユーザープロファイル作成 =====
        topic_distributions = [self.lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
        self.user_profile = np.average(
            np.array([[prob for _, prob in doc] for doc in topic_distributions]),
            axis=0,
            weights=user_ratings
        )
        print("\nユーザーの関心トピック分布:", self.user_profile)
        self.user_profile_percent = (self.user_profile / self.user_profile.sum() * 100).astype(int)
        
    def get_keywords_list(self):
        # キーワード列のすべてのリストを結合
        all_keywords = []
        for keywords in self.df_combined['キーワード']:
            if isinstance(keywords, str):  # 文字列の場合のみ処理
                all_keywords.extend(eval(keywords))  # リストとして評価して追加
            elif isinstance(keywords, list):  # 既にリスト形式の場合
                all_keywords.extend(keywords)

        # キーワードの出現頻度をカウント
        keyword_counts = Counter(all_keywords)

        # 出現頻度順に並び替え
        sorted_keywords = keyword_counts.most_common()
        keywords_only = [keyword for keyword, count in sorted_keywords]
        return keywords_only

    def get_topic_keywords(self, keywords_only):
        # ===== トピックの重要キーワード =====
        # トピックごとの単語分布を取得
        def find_highest_topic_for_keyword(keyword, dictionary):
            highest_topic_id = None
            highest_probability = 0

            # 各トピックをループして確率を確認
            for topic_id, topic_terms in self.lda_model.show_topics(formatted=False, num_words=len(dictionary)):
                for term_id, probability in topic_terms:
                    if term_id == keyword and probability > highest_probability:
                        highest_topic_id = topic_id
                        highest_probability = probability

            return highest_topic_id, highest_probability

        topic_keywords = [[[], []] for i in range(self.num_topics)]
        for topic_id in range(self.num_topics):
            top_words = self.lda_model.show_topic(topic_id, topn=10)
            topic_keywords[topic_id][0] = [word for word, _ in top_words]

        for keyword in keywords_only:
            highest_topic_id, probability = find_highest_topic_for_keyword(keyword, self.dictionary)
            topic_keywords[highest_topic_id][1].append(keyword)

        return topic_keywords
    
    def _number_to_char(self, number):
            if number < 0:
                raise ValueError("The number must be non-negative.")

            result = []
            while number >= 0:
                result.append(chr(ord('A') + number % 26))
                number = number // 26 - 1

            return ''.join(reversed(result))
        
    def assign_topic_to_courses(self):
        
        def return_most_relevant_topic_idx(doc):
            corpus = self.dictionary.doc2bow(doc)
            grad_topic_distribution = self.lda_model.get_document_topics(corpus, minimum_probability=0)
            most_relevant_topic_idx = np.argmax([prob for _, prob in grad_topic_distribution], axis=0)
            return most_relevant_topic_idx

        def return_relevant_prob(doc):
            corpus = self.dictionary.doc2bow(doc)
            grad_topic_distribution = self.lda_model.get_document_topics(corpus, minimum_probability=0)
            relevant_prob = np.max([prob for _, prob in grad_topic_distribution], axis=0)
            return relevant_prob
        
        self.df_grad_courses['関連トピック'] = self.df_grad_courses['キーワード'].apply(return_most_relevant_topic_idx)
        self.df_grad_courses['トピック'] = self.df_grad_courses['関連トピック'].apply(self._number_to_char)
        self.df_grad_courses['トピック選好'] = self.df_grad_courses['関連トピック'].apply(lambda x: self.user_profile_percent[x])
        self.df_grad_courses['トピック比率'] = self.df_grad_courses['キーワード'].apply(return_relevant_prob)
        self.df_grad_courses['推薦スコア'] = self.df_grad_courses['トピック選好'] * self.df_grad_courses['トピック比率']
        self.df_grad_courses['おすすめ度'] = self.df_grad_courses['推薦スコア'].astype(int)
        self.df_social_courses['関連トピック'] = self.df_social_courses['キーワード'].apply(return_most_relevant_topic_idx)
        self.df_social_courses['トピック'] = self.df_social_courses['関連トピック'].apply(self._number_to_char)
        self.df_social_courses['トピック比率'] = self.df_social_courses['キーワード'].apply(return_relevant_prob)
        self.df_grad = self.df_grad.merge(self.df_social_courses[['授業科目名', 'トピック']], left_on='科目名', right_on='授業科目名', how='left')

    def decide_number_of_recommendations_by_topic(self, total_n_recommendations = 50):
        number_of_recommendations_by_topic = [math.floor(i*total_n_recommendations) for i in self.user_profile]
        # 値が高い順にソートし、インデックスを保持
        number_of_recommendations_by_topic = sorted(enumerate(number_of_recommendations_by_topic), key=lambda x: x[1], reverse=True)
        return number_of_recommendations_by_topic

    def execute_recommendation(self, topic):
        # ===== 大学院授業の推薦 =====
        your_course = self.df_grad[self.df_grad['トピック']==topic]['科目名'].tolist()
        temp = pd.concat([self.df_0, self.df_2]).merge(self.df_grad_courses[['授業科目名', 'キーワード', 'トピック', 'トピック比率']], on='授業科目名', how='left')
        temp = temp[temp['トピック']==topic]

        return temp[['科目番号', '授業科目名', '単位数', '標準履修年次', '時間割', '担当教員', '成績評価方法', 'シラバス']], your_course

    def plot_topic_distribution_of_grad(self):
        # 'トピック番号' と '関連授業' ごとに出現回数をカウント
        df_topic_counts = self.df_grad_courses.groupby(['トピック', '学位プログラム']).size().reset_index(name='出現回数')

        classification_rules_digit = {
            '0': '共通',
            '1': '社会工学関連科目',
            '2': 'サービス工学関連科目',
            '3': 'リスク・レジリエンス工学関連科目',
            '4': '情報理工関連科目',
            '5': '知能機能システム関連科目',
            '6': '構造エネルギー工学関連科目',
            '7': 'エンパワーメント情報学関連科目'
        }

        for topic_id in range(self.num_topics):
            topic = self._number_to_char(topic_id)
            for classify_course in classification_rules_digit.values():
                if df_topic_counts[(df_topic_counts['トピック'] == topic) & (df_topic_counts['学位プログラム'] == classify_course)].empty:
                    df_topic_counts = pd.concat([
                        df_topic_counts,
                        pd.DataFrame({'トピック': [topic], '学位プログラム': [classify_course], '出現回数': [0]})
                    ], ignore_index=True)
        df_topic_counts = df_topic_counts.sort_values(by=['トピック', '学位プログラム'])

        # ===== 可視化 =====
        fig = px.line(df_topic_counts,
                      x='トピック',
                      y='出現回数',
                      color='学位プログラム',
                      title='学位プログラムごとの各トピックの出現回数',
                      labels={'トピック': 'トピック', '出現回数': '出現回数', '学位プログラム': '学位プログラム'})

        # グラフを表示
        fig.update_layout(
            xaxis=dict(
                tick0=0,      # X軸の開始値
                dtick=1,      # X軸の間隔
                showgrid=True # グリッドを表示
            ),
            legend=dict(
                #orientation="h",        # 横並びのレジェンド
                yanchor="top",          # レジェンドの垂直方向のアンカー
                y=-0.2,                 # グラフ下部に配置（値を調整して位置を変更）
                xanchor="center",       # レジェンドの水平方向のアンカー
                #x=0.5                   # グラフの中央に配置
            ),
            title="学位プログラムごとの各トピックの出現回数",
        )

        return fig

    def plot_topic_distribution_of_social(self):
        # 'トピック番号' と '関連授業' ごとに出現回数をカウント
        df_topic_counts = self.df_social_courses.groupby(['トピック', '主専攻']).size().reset_index(name='出現回数')

        for topic_id in range(self.num_topics):
            topic = self._number_to_char(topic_id)
            for classify_course in ["理工学群共通", "社会経済システム", "経営工学", "都市計画", "その他"]:
                if df_topic_counts[(df_topic_counts['トピック'] == topic) & (df_topic_counts['主専攻'] == classify_course)].empty:
                    df_topic_counts = pd.concat([
                        df_topic_counts,
                        pd.DataFrame({'トピック': [topic], '主専攻': [classify_course], '出現回数': [0]})
                    ], ignore_index=True)
        df_topic_counts = df_topic_counts.sort_values(by=['トピック', '主専攻'])

        # ===== 可視化 =====
        fig = px.line(df_topic_counts,
                      x='トピック',
                      y='出現回数',
                      color='主専攻',
                      title='主専攻ごとの各トピックの出現回数',
                      labels={'トピック': 'トピック', '出現回数': '出現回数', '主専攻': '主専攻'})

        # グラフを表示
        fig.update_layout(
            xaxis=dict(
                tick0=0,      # X軸の開始値
                dtick=1,      # X軸の間隔
                showgrid=True # グリッドを表示
            ),
            title="主専攻ごとの各トピックの出現回数",
            legend=dict(
                #orientation="h",        # 横並びのレジェンド
                yanchor="top",          # レジェンドの垂直方向のアンカー
                y=-0.2,                 # グラフ下部に配置（値を調整して位置を変更）
                xanchor="center",       # レジェンドの水平方向のアンカー
                #x=0.5                   # グラフの中央に配置
            ),
        )

        return fig

    def plot_topic_distribution_of_user_profile(self):
        self.user_profile_percent

        # データ
        x_values = [self._number_to_char(topic_id) for topic_id in range(self.num_topics)]
        y_values = self.user_profile_percent

        # 棒グラフの作成
        fig = go.Figure(data=[
            go.Bar(x=x_values, y=y_values, name='Values')
        ])

        # レイアウト設定
        fig.update_layout(
            title="ユーザーのトピック選考",
            xaxis_title="トピック",
            yaxis_title="関心度（％）",
            template="plotly_white",
            xaxis=dict(
                tick0=0,      # X軸の開始値
                dtick=1,      # X軸の間隔
                showgrid=True # グリッドを表示
            ),
        )

        # グラフの表示
        return fig
    
    
class OptimizeClasses():
    def __init__(self, df):
        self.df = df.copy()
        self.df['season'] = [None] * len(self.df)
        self.df['module'] = [None] * len(self.df)
        self.df['week'] = [None] * len(self.df)
        self.df['period'] = [None] * len(self.df)

        for index, row in df.iterrows():
            season_module = row['実施学期']
            print(season_module)
            result = []
            for season in ['春', '秋']:
                if pd.isna(season_module):
                    continue
                if season in season_module:
                    result.append(season)
                self.df.loc[index, 'season'] = str(result)
            
            result = []
            for module in ['A', 'B', 'C']:
                if pd.isna(season_module):
                    continue
                if module in season_module:
                    result.append(module)
                self.df.loc[index, 'module'] = str(result)
            
            result = []
            week_period = row['曜時限']
            for week in ['月', '火', '水', '木', '金']:
                if pd.isna(week_period):
                    continue
                if week in week_period:
                    result.append(week)
                self.df.loc[index, 'week'] = str(result)
                
            result = []
            for period in range(1, 6):
                if pd.isna(week_period):
                    continue
                if str(period) in week_period:
                    result.append(period)
                self.df.loc[index, 'period'] = str(result)
    
    def optimize(self, requirements_dict):
        # 問題を宣言
        problem = pulp.LpProblem("recommendation", pulp.LpMaximize)

        # 変数を宣言
        x = pulp.LpVariable.dicts('x', self.df['授業科目名'], 0, 1, pulp.LpBinary)

        def converse_list(list_):
            if not list_:
                return []
            else:
                return ast.literal_eval(list_)

        dict_time_class = {}
        for index, row in self.df.iterrows():
            for season in converse_list(row['season']):
                for module in converse_list(row['module']):
                    for week in converse_list(row['week']):
                        for period in converse_list(row['period']):
                            if f'{season}_{module}_{week}_{period}' in dict_time_class.keys():
                                dict_time_class[f'{season}_{module}_{week}_{period}'] += x[row['授業科目名']]
                            else:
                                dict_time_class[f'{season}_{module}_{week}_{period}'] = x[row['授業科目名']]
        
        # 時間割のバッキングを阻止
        for season in ['春', '秋']:
            for module in ['A', 'B', 'C']:
                for week in ['月', '火', '水', '木', '金']:
                    for period in range(1, 6):
                        if f'{season}_{module}_{week}_{period}' in dict_time_class.keys():
                            problem += dict_time_class[f'{season}_{module}_{week}_{period}'] <= 1
                            
        # 卒業要件
        
        # 選択科目数
        problem += pulp.lpSum(row['単位数'] * x[row["授業科目名"]] for index, row in self.df.iterrows()) == requirements_dict['選択必修数']
        # 専門基礎科目かつ専攻
        problem += pulp.lpSum(row['単位数'] * x[row["授業科目名"]] for index, row in self.df.iterrows() if row["科目番号"][4]==requirements_dict['学位番号'] and row["科目区分"]==0) >= requirements_dict['専門基礎科目かつ専攻']
        # 専門基礎科目かつ専攻以外
        problem += pulp.lpSum(row['単位数'] * x[row["授業科目名"]] for index, row in self.df.iterrows() if row["科目番号"][4]!=requirements_dict['学位番号'] and row["科目区分"]==0) >= requirements_dict['専門基礎科目かつ専攻以外']
        # 専門科目かつ専攻
        problem += pulp.lpSum(row['単位数'] * x[row["授業科目名"]] for index, row in self.df.iterrows() if row["科目番号"][4]==requirements_dict['学位番号'] and row["科目区分"]==1) >= requirements_dict['専門科目かつ専攻']
        # 専門科目かつ専攻以外
        problem += pulp.lpSum(row['単位数'] * x[row["授業科目名"]] for index, row in self.df.iterrows() if row["科目番号"][4]!=requirements_dict['学位番号'] and row["科目区分"]==1) >= requirements_dict['専門科目かつ専攻以外']

        # 目的関数を宣言
        problem += pulp.lpSum(row['推薦スコア'] * x[row["授業科目名"]] for index, row in self.df.iterrows())

        # 問題を解く
        status = problem.solve()

        def xx(class_name):
            return x[class_name].value()

        print("Status:", pulp.LpStatus[status])
        print("Result:")
        
        if 'トピック' in self.df.columns:
            df_took_classes = self.df[self.df['授業科目名'].apply(xx) == 1][['科目番号', '授業科目名', '時間割', '単位数', '学位プログラム', "科目区分", '科目区分名', 'トピック', '推薦スコア', 'シラバス', 'おすすめ度']]
        else:
            df_took_classes = self.df[self.df['授業科目名'].apply(xx) == 1][['科目番号', '授業科目名', '時間割', '単位数', '学位プログラム', "科目区分", '科目区分名', 'シラバス']]

        
        return df_took_classes

        #print('専門基礎科目', '社会工学')       

if __name__ == '__main__':
    df_grad = pd.read_csv("成績データ.csv", encoding="utf-8")
    recommender = TopicBasedRecommender(df_grad, num_topics=30)
    recommender.assign_info_to_courses()
    recommender.create_lda_model()
    keywords_list = recommender.get_keywords_list()
    topic_keywords = recommender.get_topic_keywords(keywords_list)
    recommender.assign_topic_to_courses()
    number_of_recommendations_by_topic = recommender.decide_number_of_recommendations_by_topic()

    #for topic, n_r in number_of_recommendations_by_topic:
    for topic in range(recommender.num_topics):
        #if n_r == 0:
        #    continue
        temp, your_course = recommender.execute_recommendation(topic)
        print("-" * 50)
        print(f"トピック {topic}")
        print(f"推定関心度：{recommender.user_profile_percent[topic]}%")
        print("推薦授業: ", end='')
        print("、".join(class_ for class_ in temp['授業科目名'].tolist()))
        print("トピック重要ワード: ", end='')
        print("、".join(word for word in topic_keywords[topic][0]))  # リストをカンマ区切りで表示
        print("トピック専門用語: ", end='')
        print("、".join(keyword for keyword in topic_keywords[topic][1]))  # リストをカンマ区切りで表示
        print("このトピックはあなたが履修した以下の授業に基づいています: ", end='')
        print("、".join(class_ for class_ in your_course))
        print()

    df = recommender.df_grad_courses
    opt = OptimizeClasses(df)
    print(opt.optimize())
    #recommender.plot_topic_distribution_of_grad().show()
    #recommender.plot_topic_distribution_of_social().show()
    #recommender.plot_topic_distribution_of_user_profile().show()
    