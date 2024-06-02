
# import pandas as pd
# from gensim import corpora
# from gensim.models import LdaModel
# from gensim.parsing.preprocessing import preprocess_string
# from sklearn.feature_extraction.text import CountVectorizer
# import numpy as np

# # Đọc dữ liệu từ tập tin CSV
# data = pd.read_csv("data.csv")

# # Tiền xử lý văn bản: loại bỏ stop words, chuyển thành lowercase, loại bỏ ký tự đặc biệt, ...
# def preprocess_text(text):
#     return preprocess_string(text)

# # Lấy danh sách văn bản từ cột 'post_text' của DataFrame
# documents = data['post_text'].tolist()

# # Tiền xử lý các văn bản
# preprocessed_documents = [preprocess_text(doc) for doc in documents]

# # Tạo từ điển từ văn bản
# dictionary = corpora.Dictionary(preprocessed_documents)

# # Chuyển các văn bản thành vector đếm từ
# corpus = [dictionary.doc2bow(doc) for doc in preprocessed_documents]

# # Số lượng chủ đề trong mô hình LDA
# num_topics = 5

# # Tạo mô hình LDA
# lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary)

# # Duyệt qua từng cụm
# for cluster in range(num_topics):
#     # Lấy danh sách các văn bản trong cụm
#     cluster_documents = [documents[i] for i, doc in enumerate(lda_model[corpus]) if max(doc, key=lambda x: x[1])[0] == cluster]
    
#     # Biểu diễn nội dung văn bản bằng vectơ
#     vectorizer = CountVectorizer()
#     X = vectorizer.fit_transform(cluster_documents)
#     word_freq = np.ravel(X.sum(axis=0))
#     word_freq = sorted([(word, word_freq[idx]) for word, idx in vectorizer.vocabulary_.items()], key=lambda x: x[1], reverse=True)
#     print(f"Cluster {cluster+1} - Top words:")
#     print(word_freq[:10])  # In ra 10 từ phổ biến nhất trong cụm
#     print()

#     # Tính trung bình vectơ
#     all_topic_vectors = [lda_model.get_document_topics(dictionary.doc2bow(preprocess_text(doc))) for doc in cluster_documents]
#     max_topics = max(len(topic_vector) for topic_vector in all_topic_vectors)
#     # Điều chỉnh độ dài của mỗi vectơ chủ đề để chúng có cùng kích thước
#     adjusted_topic_vectors = [np.pad(topic_vector, ((0, max_topics - len(topic_vector)), (0, 0)), mode='constant', constant_values=0) for topic_vector in all_topic_vectors]
#     avg_vector = np.mean(adjusted_topic_vectors, axis=0)
#     print(f"Cluster {cluster+1} - Average topic vector:")
#     print(avg_vector)
#     print()

#     # Cập nhật mô hình LDA
#     lda_model.update([dictionary.doc2bow(preprocess_text(doc)) for doc in cluster_documents])


import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial.distance import jensenshannon
import numpy as np

# # Đọc dữ liệu từ file CSV
# data = pd.read_csv('data1.csv')

# Đọc dữ liệu từ file CSV
data = pd.read_csv('Sr_data_with_post_text.csv',low_memory=False) 

# Hiển thị dữ liệu
print(len(data), data.columns)

# Chuẩn bị dữ liệu
user_data = data[['user_id', 'post_text']].drop_duplicates()

print(user_data)

# Lấy nội dung văn bản và ID người dùng
texts = user_data['post_text']
user_ids = user_data['user_id']

# Tạo TfidfVectorizer để chuyển đổi văn bản thành vector TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)

# Số chủ đề
num_topics = 10

# Tạo mô hình LDA
lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
lda.fit(X)

# Biểu diễn văn bản theo chủ đề
lda_X = lda.transform(X)

def jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    m = 0.5 * (p + q)
    return 0.5 * (jensenshannon(p, m) + jensenshannon(q, m))

# Tính toán ma trận JSD giữa các văn bản
n_docs = lda_X.shape[0]
jsd_matrix = np.zeros((n_docs, n_docs))

for i in range(n_docs):
    for j in range(i + 1, n_docs):
        jsd_matrix[i, j] = jsd(lda_X[i], lda_X[j])
        jsd_matrix[j, i] = jsd_matrix[i, j]

# Ngưỡng tương đồng
threshold = 0.5

# Tạo tập hợp các cụm văn bản liên quan và không liên quan
T_r = []
T_u = []

for i in range(n_docs):
    for j in range(i + 1, n_docs):
        if jsd_matrix[i, j] <= threshold:
            T_r.append((i, j))
        else:
            T_u.append((i, j))
            

# Tạo ma trận quan hệ
M = np.zeros((n_docs, n_docs))

for i, j in T_r:
    M[i, j] = 1
    M[j, i] = 1


print("Tập hợp các nút có liên quan về mặt văn bản:", T_r)
print("\nTập hợp các nút không liên quan về mặt văn bản:", T_u)
print("\nMa trận quan hệ M:")
print(M)


