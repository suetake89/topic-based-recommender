import ast
import math
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim import corpora, models
from gensim.models import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity


class TopicBasedRecommender():
    def __init__(self, df_grad, num_topics=100):
        self.df_grad = df_grad
        
        # ===== データ読み込み =====
        self.df_combined = pd.read_excel('大学院専門科目_df+社会工学類授業_df+大学院専門科目_df+キーワード完成版.xlsx')
        self.df_0 = pd.read_excel("大学院専門基礎科目_df.xlsx")
        self.df_1 = pd.read_excel("社会工学類授業_df.xlsx")
        self.df_2 = pd.read_excel("大学院専門科目_df.xlsx")
        # self.df_grad = pd.read_csv("成績データ.csv", encoding="utf-8")
        
        self.num_topics = num_topics

        # ===== 前処理 =====
        self.df_combined['キーワード'] = self.df_combined['キーワード'].apply(ast.literal_eval)  # 'キーワード' 列をリスト形式に変換
        self.df_0 = self.df_0.iloc[6:]  # df_0 の最初5行を削除（英語の授業を除外）

        # '授業科目名' 列をリスト化
        grad_course_list = pd.concat([self.df_0['授業科目名'], self.df_2['授業科目名']]).unique().tolist()
        social_course_list = self.df_1['授業科目名'].unique().tolist()

        # 大学院専門科目・大学院専門基礎科目の授業を抽出
        self.df_grad_courses = self.df_combined[self.df_combined['授業科目名'].isin(grad_course_list)]
        self.df_social_courses = self.df_combined[self.df_combined['授業科目名'].isin(social_course_list)]
        
        self.df_grad.rename(columns={'科目名 ': '科目名'}, inplace=True)

        # 評価を数値にマッピング
        grade_mapping = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'D': 0, 'P': 3, 'F': 0}
        self.df_grad['総合評価'] = self.df_grad['総合評価'].replace(grade_mapping)
        #self.df_grad['総合評価'] = self.df_grad.apply(lambda x: x['総合評価'] * int(x['単位数']))

        # 成績データから必要な列を抽出し、先頭の不要な行を除外
        self.df_grad = self.df_grad[['科目名', '総合評価']].iloc[9:]

        # '授業科目名' 列をデータフレームに変換して LEFT JOIN で結合
        df_class_name = self.df_combined[['授業科目名']]
        df_grad_filtered = self.df_grad[['科目名', '総合評価']]
        df_result = df_class_name.merge(df_grad_filtered, how='left', left_on='授業科目名', right_on='科目名')
        df_result['総合評価'] = df_result['総合評価'].fillna(0).astype(int)  # NaN を 0 に置換し整数型に変換
        df_result.drop(columns='科目名', inplace=True)  # 不要な列を削除

        def return_syllabus_link(class_str):
            return f'https://kdb.tsukuba.ac.jp/syllabi/2024/{class_str}/jpn'

        self.df_0['シラバス'] = self.df_0['科目番号'].apply(return_syllabus_link)
        self.df_1['シラバス'] = self.df_1['科目番号'].apply(return_syllabus_link)
        self.df_2['シラバス'] = self.df_2['科目番号'].apply(return_syllabus_link)

        # ユーザーの成績評価リストを取得
        user_ratings = df_result['総合評価'].tolist()

        # ===== LDA モデルの作成 =====
        texts = self.df_combined['キーワード']  # トークン化されたキーワードのリスト
        self.dictionary = corpora.Dictionary(texts)
        corpus = [self.dictionary.doc2bow(text) for text in texts]
        self.lda_model = models.LdaModel(corpus=corpus, id2word=self.dictionary, num_topics=self.num_topics, random_state=42, passes=10)

        # 各トピックの上位単語を表示
        print("===== 各トピックの上位単語 =====")
        for idx, topic in self.lda_model.print_topics(num_words=5):
            print(f"トピック {idx}: {topic}")

        # ===== ユーザープロファイル作成 =====
        topic_distributions = [self.lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
        self.user_profile = np.average(
            np.array([[prob for _, prob in doc] for doc in topic_distributions]),
            axis=0,
            weights=user_ratings
        )
        print("\nユーザーの関心トピック分布:", self.user_profile)
        self.user_profile_percent = (self.user_profile / self.user_profile.sum() * 100).astype(int)


        # ===== トピックの重要キーワード =====（ここから変更）

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

        # トピックごとの単語分布を取得
        def find_highest_topic_for_keyword(keyword, dictionary):
            highest_topic = None
            highest_probability = 0

            # 各トピックをループして確率を確認
            for topic_id, topic_terms in self.lda_model.show_topics(formatted=False, num_words=len(dictionary)):
                for term_id, probability in topic_terms:
                    if term_id == keyword and probability > highest_probability:
                        highest_topic = topic_id
                        highest_probability = probability

            return highest_topic, highest_probability

        self.topic_keywords = [[[], []] for i in range(self.num_topics)]
        for topic_id in range(self.num_topics):
            top_words = self.lda_model.show_topic(topic_id, topn=10)
            self.topic_keywords[topic_id][0] = [word for word, _ in top_words]

        for keyword in keywords_only:
            highest_topic, probability = find_highest_topic_for_keyword(keyword, self.dictionary)
            self.topic_keywords[highest_topic][1].append(keyword)

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
        self.df_grad_courses['トピック確度'] = self.df_grad_courses['キーワード'].apply(return_relevant_prob)
        self.df_social_courses['関連トピック'] = self.df_social_courses['キーワード'].apply(return_most_relevant_topic_idx)
        #print(self.df_grad.columns)
        #print(self.df_social_courses['授業科目名'].values_counts())
        self.df_grad = self.df_grad.merge(self.df_social_courses[['授業科目名', '関連トピック']], left_on='科目名', right_on='授業科目名', how='left')
        
        
        

    def decide_number_of_recommendations_by_topic(self, total_n_recommendations = 50):
        number_of_recommendations_by_topic = [math.floor(i*total_n_recommendations) for i in self.user_profile]
        # 値が高い順にソートし、インデックスを保持
        number_of_recommendations_by_topic = sorted(enumerate(number_of_recommendations_by_topic), key=lambda x: x[1], reverse=True)
        return number_of_recommendations_by_topic

    # ===== 大学院授業の推薦 =====
    def execute_recommendation(self, topic, n_r):
        #print(self.df_grad[self.df_grad==topic])
        your_course = self.df_grad[self.df_grad['関連トピック']==topic]['科目名'].tolist()
        temp = pd.concat([self.df_0, self.df_2]).merge(self.df_grad_courses[['授業科目名', 'キーワード', '関連トピック', 'トピック確度']], on='授業科目名', how='left')
        temp = temp[temp['関連トピック']==topic]
        indices = list(range(len(temp)))
        weights = np.array(temp['トピック確度'].tolist())
        #print(weights)
        weights = weights ** 5
        weights = weights / weights.sum()
        weights[-1] += 1.0 - weights.sum()
        #print(weights)
        #n_r = min(n_r, len(indices))
        n_r = len(indices)
        random_choices = np.random.choice(indices, size=n_r, replace=False, p=weights)
        
        return temp.iloc[random_choices][['科目番号', '授業科目名', '単位数', '標準履修年次', '時間割', '担当教員', '成績評価方法', 'シラバス']], your_course

""" 
if __name__ == '__main__':            
    df_grad = pd.read_csv("成績データ.csv", encoding="utf-8")
    recommender = TopicBasedRecommender(df_grad)
    number_of_recommendations_by_topic = recommender.decide_number_of_recommendations_by_topic()


    for topic, n_r in number_of_recommendations_by_topic:
        if n_r == 0:
            continue
        temp = recommender.execute_recommendation(topic, n_r)
        print()
        print(f"トピック {topic}")
        print(f"推薦数: {n_r}")
        print(temp)
        print("トピック重要ワード: ", end='')
        print(", ".join(word for word in recommender.topic_keywords[topic][0]))  # リストをカンマ区切りで表示
        print("トピック専門用語: ", end='')
        print(", ".join(keyword for keyword in recommender.topic_keywords[topic][1]))  # リストをカンマ区切りで表示
        print("-" * 50)
         """