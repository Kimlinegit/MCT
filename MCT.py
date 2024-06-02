# import pandas as pd
# import numpy as np
# from gensim import corpora
# from gensim.models import LdaModel
# from gensim.parsing.preprocessing import preprocess_string
# from sklearn.feature_extraction.text import CountVectorizer
# from math import exp, log

# # Hàm tính toán xác suất tương tác qua lại
# def reciprocity_probability(followers_i, followers_j, friends_i, friends_j):
#     epsilon = 1e-9  # Một giá trị nhỏ để tránh chia cho 0
#     J = (followers_i * friends_j + followers_j * friends_i) / (followers_i + followers_j + friends_i + friends_j + epsilon)
#     phi = -log(epsilon + J) * (epsilon + J)
#     return 1 / (1 + exp(phi))

# # Hàm tính toán mối quan hệ qua lại và không qua lại dựa trên ngưỡng τ
# def f_sim(user_ids, data, tau=0.5):
#     phi_r = []  # Danh sách mối quan hệ qua lại
#     phi_u = []  # Danh sách mối quan hệ không qua lại

#     # Vòng lặp chính để duyệt qua tất cả các cặp người dùng
#     for i in range(len(user_ids)):
#         for j in range(i + 1, len(user_ids)):
#             user_i = user_ids[i]
#             user_j = user_ids[j]

#             # Lấy thông tin về followers và friends của từng người dùng
#             followers_i = data.loc[data['user_id'] == user_i, 'followers'].values[0]
#             friends_i = data.loc[data['user_id'] == user_i, 'friends'].values[0]
#             followers_j = data.loc[data['user_id'] == user_j, 'followers'].values[0]
#             friends_j = data.loc[data['user_id'] == user_j, 'friends'].values[0]

#             # Tính toán xác suất tương tác qua lại
#             p_r = reciprocity_probability(followers_i, followers_j, friends_i, friends_j)

#             # Xác định mối quan hệ qua lại và không qua lại
#             if p_r >= tau:
#                 phi_r.append((user_i, user_j))
#             else:
#                 phi_u.append((user_i, user_j))

#     return phi_r, phi_u


# def text_sim(data, num_topics=5):
#     # Tiền xử lý văn bản: loại bỏ stop words, chuyển thành lowercase, loại bỏ ký tự đặc biệt, ...
#     def preprocess_text(text):
#         return preprocess_string(text)

#     # Lấy danh sách văn bản từ cột 'post_text' của DataFrame
#     documents = data['post_text'].tolist()

#     # Tiền xử lý các văn bản
#     preprocessed_documents = [preprocess_text(doc) for doc in documents]

#     # Tạo từ điển từ văn bản
#     dictionary = corpora.Dictionary(preprocessed_documents)

#     # Chuyển các văn bản thành vector đếm từ
#     corpus = [dictionary.doc2bow(doc) for doc in preprocessed_documents]

#     # Tạo mô hình LDA
#     lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

#     results = []

#     # Duyệt qua từng cụm
#     for cluster in range(num_topics):
#         # Lấy danh sách các văn bản trong cụm
#         cluster_documents = [documents[i] for i, doc in enumerate(lda_model[corpus]) if max(doc, key=lambda x: x[1])[0] == cluster]

#         # Biểu diễn nội dung văn bản bằng vectơ
#         vectorizer = CountVectorizer()
#         X = vectorizer.fit_transform(cluster_documents)
#         word_freq = np.ravel(X.sum(axis=0))
#         word_freq = sorted([(word, word_freq[idx]) for word, idx in vectorizer.vocabulary_.items()], key=lambda x: x[1], reverse=True)

#         # Tính trung bình vectơ
#         avg_vector = np.mean([lda_model.get_document_topics(dictionary.doc2bow(preprocess_text(doc))) for doc in cluster_documents], axis=0)

#         results.append({
#             'cluster': cluster + 1,
#             'top_words': word_freq[:10],
#             'average_topic_vector': avg_vector
#         })

#         # Cập nhật mô hình LDA
#         lda_model.update([dictionary.doc2bow(preprocess_text(doc)) for doc in cluster_documents])

#     return results




import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
from math import exp, log
from collections import defaultdict

# Đọc dữ liệu từ tập tin CSV
data = pd.read_csv("data.csv")

# Hàm tính toán xác suất tương tác qua lại
def reciprocity_probability(followers_i, followers_j, friends_i, friends_j):
    epsilon = 1e-9  # Một giá trị nhỏ để tránh chia cho 0
    J = (followers_i * friends_j + followers_j * friends_i) / (followers_i + followers_j + friends_i + friends_j + epsilon)
    phi = -log(epsilon + J) * (epsilon + J)
    return 1 / (1 + exp(phi))

# Hàm tính toán mối quan hệ qua lại và không qua lại dựa trên ngưỡng τ
def f_sim(user_ids, data, tau=0.5):
    phi_r = []  # Danh sách mối quan hệ qua lại
    phi_u = []  # Danh sách mối quan hệ không qua lại

    # Vòng lặp chính để duyệt qua tất cả các cặp người dùng
    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            user_i = user_ids[i]
            user_j = user_ids[j]

            # Lấy thông tin về followers và friends của từng người dùng
            followers_i = data.loc[data['user_id'] == user_i, 'followers'].values[0]
            friends_i = data.loc[data['user_id'] == user_i, 'friends'].values[0]
            followers_j = data.loc[data['user_id'] == user_j, 'followers'].values[0]
            friends_j = data.loc[data['user_id'] == user_j, 'friends'].values[0]

            # Tính toán xác suất tương tác qua lại
            p_r = reciprocity_probability(followers_i, followers_j, friends_i, friends_j)

            # Xác định mối quan hệ qua lại và không qua lại
            if p_r >= tau:
                phi_r.append((user_i, user_j))
            else:
                phi_u.append((user_i, user_j))

    return phi_r, phi_u

# Hàm phân cụm đa cấp MCT
def multilevel_clustering(data, tau=0.5, num_topics=5):
    # Giai đoạn cấu trúc
    user_ids = data['user_id'].unique()
    phi_r, phi_u = f_sim(user_ids, data, tau)
    # Dùng thuật toán phát hiện cộng đồng để xác định thành phần cấu trúc

    # Giai đoạn văn bản
    documents = data['post_text'].tolist()
    preprocessed_documents = [preprocess_string(doc) for doc in documents]
    dictionary = corpora.Dictionary(preprocessed_documents)
    corpus = [dictionary.doc2bow(doc) for doc in preprocessed_documents]
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

    # So sánh các chủ đề
    user_topics = defaultdict(list)
    for i, doc in enumerate(lda_model[corpus]):
        user_topics[data['user_id'][i]].extend(doc)

    # Nhóm các nút có chủ đề văn bản tương tự nhau vào cùng một cộng đồng
    communities = defaultdict(list)
    for user_id, topics in user_topics.items():
        dominant_topic = max(topics, key=lambda x: x[1])[0]
        communities[dominant_topic].append(user_id)

    # Đánh giá và cập nhật

    # Kết quả
    return communities

# Thực hiện thuật toán
communities = multilevel_clustering(data)

# In kết quả
for i, community in enumerate(communities.values(), start=1):
    print(f"Cộng đồng {i}: {community}")


# Tạo đồ thị
G = nx.Graph()

# Màu cho các cộng đồng
community_colors = ['skyblue', 'lightgreen', 'salmon', 'red', 'gold']

# Thêm các nút và cạnh vào đồ thị
for i, community in enumerate(communities.values()):
    G.add_nodes_from(community)
    for j in range(len(community)):
        for k in range(j + 1, len(community)):
            G.add_edge(community[j], community[k])

# Vẽ đồ thị
pos = nx.spring_layout(G)  # Sắp xếp nút
for i, community in enumerate(communities.values()):
    nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=community_colors[i], node_size=500, label=f'Community {i+1}')
nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', font_weight='bold')
plt.title('Community Detection')
plt.legend(loc='best')
plt.axis('off')
plt.show()
