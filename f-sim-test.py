
# import numpy as np
# import pandas as pd

# D = pd.read_csv('data1.csv',low_memory=False)
# print(len(D), D.columns)


# def get_abs_values(x):
#     if x >= 0.75 and x <= 1.25:
#         return 1
#     else:
#         return 0
   
# def jaccard_sim(l):
#     a = [i for i in l if i>0]
#     b = [i for i in l if i<1]
#     return (len(a)/(len(a)+len(b)))

# # binning communities to a high-level structure acording:
# comm_bands = np.linspace(0.75, 1.25) # create evely spaced-communities
# # account for outliers/very high values:
# comm_bands = np.append(comm_bands, [5,10,50,100,150,300,700,1000])

# import pandas as pd

# # Đọc dữ liệu từ tệp CSV
# data = pd.read_csv('data.csv')

# # Tính toán indegree và outdegree cho từng user
# # Indegree: Số lượng người theo dõi (followers)
# # Outdegree: Số lượng bạn bè (friends)

# # Giả sử chúng ta chỉ quan tâm đến các cột liên quan
# user_data = data[['user_id', 'followers', 'friends']]

# # Tính indegree (followers) và outdegree (friends) cho từng user
# indegree = user_data.groupby('user_id')['followers'].max().reset_index()
# outdegree = user_data.groupby('user_id')['friends'].max().reset_index()

# # Đổi tên các cột để dễ hiểu hơn
# indegree.columns = ['user_id', 'indegree']
# outdegree.columns = ['user_id', 'outdegree']

# # Gộp hai DataFrame lại để có đầy đủ thông tin
# user_degrees = pd.merge(indegree, outdegree, on='user_id')

# # Hiển thị kết quả
# print(user_degrees.head())


import pandas as pd
import numpy as np

# Đọc dữ liệu từ tệp CSV
data = pd.read_csv('data/data.csv',low_memory=False)

# Tính indegree và outdegree
# Indegree: số lượng followers của mỗi user_id
indegree = data.groupby('user_id')['followers'].sum().reset_index()
indegree.columns = ['user_id', 'indegree']

# Outdegree: số lượng friends của mỗi user_id
outdegree = data.groupby('user_id')['friends'].sum().reset_index()
outdegree.columns = ['user_id', 'outdegree']

# Kết hợp indegree và outdegree vào một DataFrame
user_degrees = pd.merge(indegree, outdegree, on='user_id')

# Hiển thị kết quả
print(user_degrees)

def get_abs_values(x): # this function applies to indegree and outdegree only ... 
    if x >= 0.75 and x <= 1.25:
        return 1
    else:
        return 0
# function to compute the Jaccard similarity:
def jaccard_sim(l):
    a = [i for i in l if i>0]
    b = [i for i in l if i<1]
    return (len(a)/(len(a)+len(b)))
# binning communities to a high-level structure acording:
comm_bands = np.linspace(0.75, 1.25) # create evely spaced-communities
# account for outliers/very high values:
comm_bands = np.append(comm_bands, [5,10,50,100,150,300,700,1000])


# Đoạn mã để trả về xác suất liên kết qua lại giữa cặp nút dựa trên sự tương tự về cấu trúc

# Khởi tạo danh sách chứa dữ liệu
SP = []
               
# a loop for the anchor node is needed such that after full iteration, another node is selected as the anchor node:
for r in range(len(user_degrees)-2):
    va_uid,va_ind,va_out = user_degrees.iloc[r].user_id, int(user_degrees.iloc[r].indegree), int(user_degrees.iloc[r].outdegree)
    netsize = (va_ind+va_out)
    try:
        for i in range(r+1, len(user_degrees)+1): # keep increasing the node for comparison until the end ... 

                vi_uid, vi_ind, vi_out = user_degrees.iloc[i].user_id, int(user_degrees.iloc[i].indegree), int(user_degrees.iloc[i].outdegree)
        
                # compute the ratios of features components, see text for details:
                Ind_Vai = va_ind/vi_ind
                Out_Vai = va_out/vi_out 
                # extract subset of features to compute the final similarity based on Jaccard index:
                AbsInd_Vai = get_abs_values(Ind_Vai)
                AbsOut_Vai = get_abs_values(Out_Vai)
                # a high-level reciprocal communities
                recip_comm = (Ind_Vai + Out_Vai)/2 
                # compute the J-sim and assign to the dataframe:
                JSim_Vai = jaccard_sim([AbsInd_Vai,AbsOut_Vai])
                
                # CONSTANT ERROR AND PROBABLE RECIPROCITY: expressed as a function of the similaity value:
                Error_Vai = round((1/(1+np.exp(1+np.log(JSim_Vai + 0.3)*(0.3)))),3)
                
                # comput phi i.e. the related terms ... 
                SimAndError_Vai = round(-np.log(Error_Vai + JSim_Vai) * (Error_Vai + JSim_Vai),3)
                Prob_Tie =  1/(1+np.exp(SimAndError_Vai))
                #Denote pairs as dyads (1) or otherwise (0) based on a threshold, 0.45 in this instance
                if Prob_Tie > 0.45:
                    Dyads = 1
                else:
                    Dyads = 0
                netsize = va_ind+va_out
                
                SP.append([va_uid, vi_uid, JSim_Vai, Prob_Tie, Dyads, netsize, recip_comm, Ind_Vai, Out_Vai, AbsInd_Vai, AbsOut_Vai])
    except:
        continue

# convert output to a dataframe object:
dk = pd.DataFrame(SP, columns=['Va_ID', 'Vi_ID', 'JSim_Vai', 'Prob_Tie', 'Dyads', 'NetSize', 'RecipComm', 'Ind_Vai', 'Out_Vai', 'AbsInd_Vai', 'AbsOut_Vai'])
print(dk)

# assign the length of the network size for binning:
s=set(dk.NetSize.values)
n_bins = len(s)
print(n_bins)
# reduce the bins ... 
# n_bins = n_bins-1000 # previous reduction
#create the network bins:
net_bins = np.linspace(dk['NetSize'].min(), dk['NetSize'].max(), n_bins, dtype=np.int64)
net_bin_index = np.digitize(dk['NetSize'], net_bins) # get the index of each item in the unbinned data
dk['NetBinIndex'] = net_bin_index # add the column containing the time bin of each data item
#associate each network bin to its corresponding network size:

# this code include the actual bin for each time:
net_bins_list = net_bins.tolist() # convert the net_bins to list from np.ndarray to support appending
net_bands = [] # stores binned posting times as periods
for index in dk.NetBinIndex:
    net_bands.append(net_bins_list[index])
    net_bins_list.append(net_bins_list[index])#replace item back to the list to avoid exhausting the list b4 end
dk['NetBand'] = net_bands

# Bin users according to high-level communities
dk['CommBand']=dk.RecipComm.apply(lambda x: round(min(comm_bands, key=lambda v: abs(v-x)),2))

#convert to dataframe and store for further analysis ...
#structurally-related nodes:
sr  = dk[dk.Dyads>0]
# structurally-unrelated nodes:
su = dk[dk.Dyads<1]
# sizes 
print(len(sr),len(su))
print("\ncác nút liên quan đến cấu trúc:\n", sr)
print("\ncác nút không liên quan đến cấu trúc:\n", su)

# Lưu files:
# Liên quan + không liên quan về cấu trúc
dk.to_csv('data/mct_structurally_related_nodes03.csv')
# Liên quan về cấu trúc:
sr.to_csv('data/mct_structurally_similar_nodes03.csv')
# Không liên quan về cấu trúc:
su.to_csv('data/mct_structurally_dissimilar_nodes03.csv')

# Load khung dữ liệu của các nút liên quan và không liên quan để phân tích quang phổ 
dsr = pd.read_csv('data/mct_structurally_related_nodes03.csv')
# drop duplicates trong dữ liệu:
df = dsr.drop_duplicates(subset='Va_ID')
# len(df),len(set(df.Va_ID)), len(set(df.Vi_ID))

print(df)

# Tạo khung dữ liệu dựa trên chiều dài của người dùng để cho phép hình thành Ma trận điều chỉnh. 
# Tạo danh sách các cột và chỉ mục từ các tập hợp
columns = list(set.union(set(df.Va_ID), set(df.Vi_ID)))
index = list(set.union(set(df.Va_ID), set(df.Vi_ID)))

# Tạo DataFrame với các giá trị ban đầu là 0
sr_amt = pd.DataFrame(np.zeros(shape=(len(index), len(columns))), columns=columns, index=index)

# Cập nhật các giá trị tương ứng trong DataFrame
for v1, v2, r in zip(df.Va_ID, df.Vi_ID, df.Dyads):
    if r > 0:
        sr_amt.at[v1, v2] = r
    else:
        sr_amt.at[v1, v2] = 0

# Lấy các hàng và cột với giá trị Dyads lớn hơn 0
k = sr_amt[sr_amt.values > 0]

# Hiển thị kết quả
print(k)
print("\n", sr_amt)

k.to_csv('data/k.csv')
# Liên quan về cấu trúc:
sr_amt.to_csv('data/sr_amt.csv')

