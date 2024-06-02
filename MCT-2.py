
# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation

# # Đọc dữ liệu từ file CSV
# data = pd.read_csv("data1.csv")

# # Khởi tạo các tập hợp và ma trận
# S = set(data['user_id'])
# R = set(zip(data['user_id'], data['followers'], data['friends']))
# v = set()

# # Bước 2: Xác định thành phần cấu trúc
# # Có thể triển khai thuật toán f-sim ở đây để tạo ra các thành phần cấu trúc

# # Bước 3: Xác định thành phần văn bản
# vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
# X = vectorizer.fit_transform(data['post_text'].values.astype('U'))

# # Khởi tạo mô hình LDA và fit với dữ liệu
# lda = LatentDirichletAllocation(n_components=5, random_state=42)
# X_topics = lda.fit_transform(X)

# # Tính toán ma trận ái lực văn bản M_xn

# # Bước 4: Khởi tạo các cụm
# # Lựa chọn ngẫu nhiên bốn nút làm hạt giống cho các cụm
# seed_nodes = np.random.choice(list(S), 4, replace=True)

# # Tính toán độ tương đồng giữa các cặp nút sử dụng thông tin cấu trúc và văn bản
# def compute_similarity(v_i, v_j):
#     # Tính toán độ tương đồng sử dụng thông tin cấu trúc và văn bản
#     return similarity

# # Bước 5: Gán các nút vào các cụm
# # Lặp lại cho đến khi tất cả các nút được gán vào các cụm
# while len(S) > 0:
#     for v_i in S:
#         # Tính toán độ tương đồng của v_i với mỗi cụm
#         similarities = [compute_similarity(v_i, v_j) for v_j in seed_nodes]
#         # Gán v_i vào cụm có độ tương đồng cao nhất
#         max_similarity_cluster = np.argmax(similarities)
#         v.add((v_i, max_similarity_cluster))
#         # Cập nhật vectơ trung bình của cụm sau khi gán các nút mới
#         # seed_nodes[max_similarity_cluster] = updated_mean_vector

#         # Xóa nút đã gán khỏi tập hợp S
#         S.remove(v_i)

# # Bước 7: Đầu ra
# print("Các microcosms:")
# print(v)



# import pandas as pd
# import numpy as np

# def f_sim(data, tau=0.5, epsilon=1e-5):
#     # Chuẩn bị dữ liệu
#     user_data = data[['user_id', 'followers', 'friends']].drop_duplicates()

#     # Hàm tính xác suất tương tác qua lại dựa trên Eq. 3
#     def compute_reciprocity_prob(row1, row2):
#         followers1, friends1 = row1['followers'], row1['friends']
#         followers2, friends2 = row2['followers'], row2['friends']

#         # Tính toán xác suất tương tác qua lại
#         J_followers = lambda x, y: abs(x * y) / abs(x + y + epsilon)
#         J_friends = lambda x, y: abs(x * y) / abs(x + y + epsilon)
#         phi = -np.log(epsilon + J_followers(followers1, followers2) * J_friends(friends1, friends2)) * (epsilon + J_followers(followers1, followers2) * J_friends(friends1, friends2))
#         prob = 1 / (1 + np.exp(phi))
#         return prob

#     # Khởi tạo các tập hợp rỗng
#     Sr = set()
#     Su = set()

#     # Lấy các cặp người dùng
#     user_ids = user_data['user_id'].unique()

#     # Vòng lặp chính
#     for i in range(len(user_ids)):
#         for j in range(i + 1, len(user_ids)):
#             user1 = user_ids[i]
#             user2 = user_ids[j]

#             # Lấy dữ liệu của từng người dùng
#             row1 = user_data[user_data['user_id'] == user1].iloc[0]
#             row2 = user_data[user_data['user_id'] == user2].iloc[0]

#             # Tính toán xác suất tương tác qua lại
#             p_R = compute_reciprocity_prob(row1, row2)

#             # Kiểm tra ngưỡng xác suất
#             if p_R >= tau:
#                 Sr.add((user1, user2))
#             else:
#                 Su.add((user1, user2))

#     # Trích xuất danh sách user_id từ tập hợp Sr
#     user_ids_in_Sr = set([user_id for user_pair in Sr for user_id in user_pair])

#     # Lọc dữ liệu để chỉ chứa các user_id có trong tập hợp Sr và chỉ chọn cột 'user_id' và 'post_text'
#     post_texts_Sr = data[data['user_id'].isin(user_ids_in_Sr)][['user_id', 'post_text']].drop_duplicates()

#     # Loại bỏ các dòng có giá trị trùng lặp trong cột 'user_id'
#     unique_posts_Sr = post_texts_Sr.drop_duplicates(subset=['user_id'])

#     # Tạo ma trận kề
#     n = len(user_ids)
#     adj_matrix = np.zeros((n, n))

#     # Lập chỉ số cho user_id
#     user_index = {user_id: index for index, user_id in enumerate(user_ids)}

#     for (user1, user2) in Sr:
#         i, j = user_index[user1], user_index[user2]
#         adj_matrix[i, j] = 1
#         adj_matrix[j, i] = 1


#     return adj_matrix

# # Đọc dữ liệu từ file CSV
# data = pd.read_csv('data.csv', low_memory=False)

# # Gọi hàm f_sim
# adj_matrix = f_sim(data)

# # Hiển thị kết quả
# print(adj_matrix)


# import pandas as pd

# # Đọc dữ liệu từ tệp CSV
# data = pd.read_csv('data.csv')
# print(data.head())

# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Tách dữ liệu văn bản và người dùng
# texts = data['post_text'].values
# users = data['user_id'].values

# # Vector hóa văn bản
# vectorizer = TfidfVectorizer()
# text_vectors = vectorizer.fit_transform(texts)

# # Tính ma trận tương đồng văn bản
# text_similarity_matrix = cosine_similarity(text_vectors)

# # Lấy các đặc trưng người dùng
# user_features = data[['followers', 'friends', 'favourites', 'statuses']].values

# # Chuẩn hóa dữ liệu
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# user_features_scaled = scaler.fit_transform(user_features)

# # Tính ma trận tương đồng cấu trúc
# structural_similarity_matrix = cosine_similarity(user_features_scaled)

# # Tổng hợp ma trận tương đồng (giả định trọng số 0.5 cho mỗi phần)
# combined_similarity_matrix = (text_similarity_matrix + structural_similarity_matrix) / 2

# from sklearn.cluster import AgglomerativeClustering

# # Số cụm tối đa
# max_clusters = 10

# # Khởi tạo cụm sử dụng Agglomerative Clustering
# clustering = AgglomerativeClustering(n_clusters=max_clusters, affinity='precomputed', linkage='average')
# clusters = clustering.fit_predict(1 - combined_similarity_matrix)

# # Lưu kết quả vào DataFrame
# data['cluster'] = clusters
# print(data[['post_id', 'cluster']])

# # Tính trung bình vector văn bản của các cụm
# cluster_centers = []
# for cluster in range(max_clusters):
#     cluster_texts = text_vectors[clusters == cluster]
#     cluster_centers.append(cluster_texts.mean(axis=0))

# # Gán các nút vào cụm gần nhất
# def assign_cluster(text_vector, cluster_centers):
#     similarities = [cosine_similarity(text_vector, center)[0, 0] for center in cluster_centers]
#     return np.argmax(similarities)

# data['assigned_cluster'] = data.apply(lambda row: assign_cluster(text_vectors[row.name], cluster_centers), axis=1)
# print(data[['post_id', 'assigned_cluster']])

# # Cập nhật trung bình các cụm
# def update_cluster_centers(data, text_vectors, clusters, max_clusters):
#     cluster_centers = []
#     for cluster in range(max_clusters):
#         cluster_texts = text_vectors[clusters == cluster]
#         cluster_centers.append(cluster_texts.mean(axis=0))
#     return cluster_centers

# # Lặp lại bước gán và cập nhật cho đến khi không thay đổi
# prev_clusters = None
# while prev_clusters is None or not np.array_equal(prev_clusters, clusters):
#     prev_clusters = clusters.copy()
#     cluster_centers = update_cluster_centers(data, text_vectors, clusters, max_clusters)
#     data['assigned_cluster'] = data.apply(lambda row: assign_cluster(text_vectors[row.name], cluster_centers), axis=1)
#     clusters = data['assigned_cluster'].values

# # Kết xuất kết quả cuối cùng
# data['final_cluster'] = clusters
# print(data[['post_id', 'final_cluster']])


# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from scipy.spatial.distance import jensenshannon
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# data = pd.read_csv('data.csv', low_memory=False)

# # Tách dữ liệu văn bản và người dùng
# texts = data['post_text'].values
# users = data['user_id'].values

# # Vector hóa văn bản
# vectorizer = TfidfVectorizer()
# text_vectors = vectorizer.fit_transform(texts)

# def f_sim(data, tau=0.5):
#     # Chuẩn bị dữ liệu
#     user_data = data[['user_id', 'followers', 'friends']].drop_duplicates()

#     # Hàm tính xác suất tương tác qua lại dựa trên Eq. 3
#     def compute_reciprocity_prob(row1, row2, epsilon=1e-5):
#         followers1, friends1 = row1['followers'], row1['friends']
#         followers2, friends2 = row2['followers'], row2['friends']

#         # Tính toán xác suất tương tác qua lại
#         J_followers = lambda x, y: abs(x * y) / abs(x + y + epsilon)
#         J_friends = lambda x, y: abs(x * y) / abs(x + y + epsilon)
#         phi = -np.log(epsilon + J_followers(followers1, followers2) * J_friends(friends1, friends2)) * (epsilon + J_followers(followers1, followers2) * J_friends(friends1, friends2))
#         prob = 1 / (1 + np.exp(phi))
#         return prob

#     # Khởi tạo các tập hợp rỗng
#     Sr = set()
#     Su = set()

#     # Lấy các cặp người dùng
#     user_ids = user_data['user_id'].unique()

#     # Vòng lặp chính
#     for i in range(len(user_ids)):
#         for j in range(i + 1, len(user_ids)):
#             user1 = user_ids[i]
#             user2 = user_ids[j]

#             # Lấy dữ liệu của từng người dùng
#             row1 = user_data[user_data['user_id'] == user1].iloc[0]
#             row2 = user_data[user_data['user_id'] == user2].iloc[0]

#             # Tính toán xác suất tương tác qua lại
#             p_R = compute_reciprocity_prob(row1, row2)

#             # Kiểm tra ngưỡng xác suất
#             if p_R >= tau:
#                 Sr.add((user1, user2))
#             else:
#                 Su.add((user1, user2))

#     # Tạo ma trận kề
#     n = len(user_ids)
#     adj_matrix = np.zeros((n, n))

#     # Lập chỉ số cho user_id
#     user_index = {user_id: index for index, user_id in enumerate(user_ids)}

#     for (user1, user2) in Sr:
#         i, j = user_index[user1], user_index[user2]
#         adj_matrix[i, j] = 1
#         adj_matrix[j, i] = 1

#     return adj_matrix


# def text_sim(data):
#     # Chuẩn bị dữ liệu
#     user_data = data[['user_id', 'post_text']].drop_duplicates()

#     # Lấy nội dung văn bản và ID người dùng
#     texts = user_data['post_text']
#     user_ids = user_data['user_id']

#     # Tạo TfidfVectorizer để chuyển đổi văn bản thành vector TF-IDF
#     vectorizer = TfidfVectorizer(stop_words='english')
#     X = vectorizer.fit_transform(texts)

#     # Số chủ đề
#     num_topics = 10

#     # Tạo mô hình LDA
#     lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
#     lda.fit(X)

#     # Biểu diễn văn bản theo chủ đề
#     lda_X = lda.transform(X)

#     def jsd(p, q):
#         p = np.asarray(p)
#         q = np.asarray(q)
#         m = 0.5 * (p + q)
#         return 0.5 * (jensenshannon(p, m) + jensenshannon(q, m))

#     # Tính toán ma trận JSD giữa các văn bản
#     # n_docs = lda_X.shape[0]
#     # jsd_matrix = np.zeros((n_docs, n_docs))
#     n_users = len(user_ids)
#     jsd_matrix = np.zeros((n_users, n_users))

#     for i in range(n_users):
#         for j in range(i + 1, n_users):
#             jsd_matrix[i, j] = jsd(lda_X[i], lda_X[j])
#             jsd_matrix[j, i] = jsd_matrix[i, j]

#     # Ngưỡng tương đồng
#     threshold = 0.5

#     # Tạo tập hợp các cụm văn bản liên quan và không liên quan
#     T_r = []
#     T_u = []

#     for i in range(n_users):
#         for j in range(i + 1, n_users):
#             if jsd_matrix[i, j] <= threshold:
#                 T_r.append((i, j))
#             else:
#                 T_u.append((i, j))

#     # Tạo ma trận quan hệ
#     M = np.zeros((n_users, n_users))

#     for i, j in T_r:
#         M[i, j] = 1
#         M[j, i] = 1

#     return M


# # Sử dụng hàm f_sim
# data = pd.read_csv('data.csv', low_memory=False)
# unique_user_data = data.drop_duplicates(subset=['user_id'])
# print(unique_user_data)
# structural_similarity_matrix = f_sim(unique_user_data, tau=0.5)
# print("\nMa trận kề (adjacency matrix):")
# print(structural_similarity_matrix.shape)


# # Sử dụng hàm text_sim
# # data = pd.read_csv('Sr_data_with_post_text.csv', low_memory=False)
# # print(data)
# text_similarity_matrix = text_sim(unique_user_data)
# print("\nMa trận quan hệ M:")
# print(text_similarity_matrix.shape)


# # Tổng hợp ma trận tương đồng (giả định trọng số 0.5 cho mỗi phần)
# combined_similarity_matrix = (text_similarity_matrix + structural_similarity_matrix) / 2

# from sklearn.cluster import AgglomerativeClustering

# # Số cụm tối đa
# max_clusters = 10

# # Khởi tạo cụm sử dụng Agglomerative Clustering
# # clustering = AgglomerativeClustering(n_clusters=max_clusters, affinity='precomputed', linkage='average')
# # clusters = clustering.fit_predict(1 - combined_similarity_matrix)

# # Tính toán ma trận khoảng cách giữa các mẫu
# distance_matrix = 1 - combined_similarity_matrix

# # Sử dụng ma trận khoảng cách trong AgglomerativeClustering
# clustering = AgglomerativeClustering(n_clusters=max_clusters, linkage='average')
# clusters = clustering.fit_predict(distance_matrix)

# # Lưu kết quả vào DataFrame
# data['cluster'] = clusters
# print(data[['post_id', 'cluster']])

# # Tính trung bình vector văn bản của các cụm
# cluster_centers = []
# for cluster in range(max_clusters):
#     cluster_texts = text_vectors[clusters == cluster]
#     cluster_centers.append(cluster_texts.mean(axis=0))

# # Gán các nút vào cụm gần nhất
# def assign_cluster(text_vector, cluster_centers):
#     similarities = [cosine_similarity(text_vector, center)[0, 0] for center in cluster_centers]
#     return np.argmax(similarities)

# data['assigned_cluster'] = data.apply(lambda row: assign_cluster(text_vectors[row.name], cluster_centers), axis=1)
# print(data[['post_id', 'assigned_cluster']])

# # Cập nhật trung bình các cụm
# def update_cluster_centers(data, text_vectors, clusters, max_clusters):
#     cluster_centers = []
#     for cluster in range(max_clusters):
#         cluster_texts = text_vectors[clusters == cluster]
#         cluster_centers.append(cluster_texts.mean(axis=0))
#     return cluster_centers

# # Lặp lại bước gán và cập nhật cho đến khi không thay đổi
# prev_clusters = None
# while prev_clusters is None or not np.array_equal(prev_clusters, clusters):
#     prev_clusters = clusters.copy()
#     cluster_centers = update_cluster_centers(data, text_vectors, clusters, max_clusters)
#     data['assigned_cluster'] = data.apply(lambda row: assign_cluster(text_vectors[row.name], cluster_centers), axis=1)
#     clusters = data['assigned_cluster'].values

# # Kết xuất kết quả cuối cùng
# data['final_cluster'] = clusters
# print(data[['post_id', 'final_cluster']])


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import AgglomerativeClustering

def f_sim(data, tau=0.5):
    user_data = data[['user_id', 'followers', 'friends']].drop_duplicates()

    def compute_reciprocity_prob(row1, row2, epsilon=1e-5):
        followers1, friends1 = row1['followers'], row1['friends']
        followers2, friends2 = row2['followers'], row2['friends']

        J_followers = lambda x, y: abs(x * y) / abs(x + y + epsilon)
        J_friends = lambda x, y: abs(x * y) / abs(x + y + epsilon)
        phi = -np.log(epsilon + J_followers(followers1, followers2) * J_friends(friends1, friends2)) * (
                    epsilon + J_followers(followers1, followers2) * J_friends(friends1, friends2))
        prob = 1 / (1 + np.exp(phi))
        return prob

    Sr = set()
    Su = set()
    user_ids = user_data['user_id'].unique()

    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            user1 = user_ids[i]
            user2 = user_ids[j]
            row1 = user_data[user_data['user_id'] == user1].iloc[0]
            row2 = user_data[user_data['user_id'] == user2].iloc[0]
            p_R = compute_reciprocity_prob(row1, row2)
            if p_R >= tau:
                Sr.add((user1, user2))
            else:
                Su.add((user1, user2))

    n = len(user_ids)
    adj_matrix = np.zeros((n, n))
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}

    for (user1, user2) in Sr:
        i, j = user_index[user1], user_index[user2]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1

    return adj_matrix

def text_sim(data):
    user_data = data[['user_id', 'post_text']].drop_duplicates()
    texts = user_data['post_text']
    user_ids = user_data['user_id']
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    num_topics = 10
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)
    lda_X = lda.transform(X)

    def jsd(p, q):
        p = np.asarray(p)
        q = np.asarray(q)
        m = 0.5 * (p + q)
        return 0.5 * (jensenshannon(p, m) + jensenshannon(q, m))

    n_users = len(user_ids)
    jsd_matrix = np.zeros((n_users, n_users))

    for i in range(n_users):
        for j in range(i + 1, n_users):
            jsd_matrix[i, j] = jsd(lda_X[i], lda_X[j])
            jsd_matrix[j, i] = jsd_matrix[i, j]

    threshold = 0.5
    T_r = []
    T_u = []

    for i in range(n_users):
        for j in range(i + 1, n_users):
            if jsd_matrix[i, j] <= threshold:
                T_r.append((i, j))
            else:
                T_u.append((i, j))

    M = np.zeros((n_users, n_users))

    for i, j in T_r:
        M[i, j] = 1
        M[j, i] = 1

    return M

def psi_sa_ta(data, text_threshold=0.5, struct_threshold=0.5):
    struct_similarity = f_sim(data)
    text_similarity = text_sim(data)
    psi_matrix = struct_similarity * struct_threshold + text_similarity * text_threshold
    return psi_matrix

def mct_2(data, tau=0.5, text_threshold=0.5, struct_threshold=0.5, max_clusters=10):
    psi_matrix = psi_sa_ta(data, text_threshold, struct_threshold)

    # Chuyển đổi ma trận tương đồng thành ma trận khoảng cách
    distance_matrix = 1 - psi_matrix

    agg_cluster = AgglomerativeClustering(n_clusters=None, linkage='complete', distance_threshold=0.5)

    clusters = agg_cluster.fit_predict(distance_matrix)

    return clusters

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data.csv')

unique_data = data.drop_duplicates(subset=['user_id'])

print(unique_data)

# Áp dụng thuật toán MCT-2
clusters = mct_2(unique_data)

# In kết quả
# print(clusters)

# # Tạo một dictionary để lưu trữ các user_id trong từng cluster
# cluster_users = {}

# # Lặp qua tất cả các cluster và lưu trữ user_id trong từng cluster
# for cluster_label in np.unique(clusters):
#     cluster_users[cluster_label] = unique_data.loc[clusters == cluster_label, 'user_id'].tolist()

# # In ra các cụm cluster cụ thể
# for cluster_label, users in cluster_users.items():
#     print(f"Cluster {cluster_label}: {users}")


community_df = pd.DataFrame({'user_id': unique_data['user_id'], 'cluster': clusters})
# Lưu DataFrame vào file CSV
community_df.to_csv('local_communities.csv', index=False)
print(community_df)