
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data.csv',low_memory=False) 

# Hiển thị dữ liệu
print(len(data), data.columns)

# Chuẩn bị dữ liệu
user_data = data[['user_id', 'followers', 'friends']].drop_duplicates()

print(user_data)

# Hàm tính xác suất tương tác qua lại dựa trên Eq. 3
def compute_reciprocity_prob(row1, row2, epsilon=1e-5):
    followers1, friends1 = row1['followers'], row1['friends']
    followers2, friends2 = row2['followers'], row2['friends']

    # Tính toán xác suất tương tác qua lại
    J_followers = lambda x, y: abs(x * y) / abs(x + y + epsilon)
    J_friends = lambda x, y: abs(x * y) / abs(x + y + epsilon)
    phi = -np.log(epsilon + J_followers(followers1, followers2) * J_friends(friends1, friends2)) * (epsilon + J_followers(followers1, followers2) * J_friends(friends1, friends2))
    prob = 1 / (1 + np.exp(phi))
    return prob

# Khởi tạo các tập hợp rỗng
Sr = set()
Su = set()

# Đặt ngưỡng xác suất
tau = 0.5

# Lấy các cặp người dùng
user_ids = user_data['user_id'].unique()

# Vòng lặp chính
for i in range(len(user_ids)):
    for j in range(i + 1, len(user_ids)):
        user1 = user_ids[i]
        user2 = user_ids[j]
        
        # Lấy dữ liệu của từng người dùng
        row1 = user_data[user_data['user_id'] == user1].iloc[0]
        row2 = user_data[user_data['user_id'] == user2].iloc[0]
        
        # Tính toán xác suất tương tác qua lại
        p_R = compute_reciprocity_prob(row1, row2)
        
        # Kiểm tra ngưỡng xác suất
        if p_R >= tau:
            Sr.add((user1, user2))
        else:
            Su.add((user1, user2))


# Trích xuất danh sách user_id từ tập hợp Sr
user_ids_in_Sr = set([user_id for user_pair in Sr for user_id in user_pair])

# Lọc dữ liệu để chỉ chứa các user_id có trong tập hợp Sr và chỉ chọn cột 'user_id' và 'post_text'
post_texts_Sr = data[data['user_id'].isin(user_ids_in_Sr)][['user_id', 'post_text']].drop_duplicates()

# Hiển thị kết quả
print(post_texts_Sr)

# Loại bỏ các dòng có giá trị trùng lặp trong cột 'user_id'
unique_posts_Sr = post_texts_Sr.drop_duplicates(subset=['user_id'])

# Hiển thị kết quả
print(unique_posts_Sr)

unique_posts_Sr.to_csv('Sr_data_with_post_text.csv')


# Tạo ma trận kề
n = len(user_ids)
adj_matrix = np.zeros((n, n))

# Lập chỉ số cho user_id
user_index = {user_id: index for index, user_id in enumerate(user_ids)}

for (user1, user2) in Sr:
    i, j = user_index[user1], user_index[user2]
    adj_matrix[i, j] = 1
    adj_matrix[j, i] = 1

# Kết quả
print("Tập hợp các cặp nút tương đồng cấu trúc:", Sr)
print("\nTập hợp các cặp nút không tương đồng cấu trúc:", Su)
print("\nMa trận kề (adjacency matrix):")
print(adj_matrix)
