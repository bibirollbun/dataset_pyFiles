import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors



path = "/kaggle/input/matchverse-2025-ai-matrimonial-challenge/"
users = pd.read_csv(path+"users.csv")
interactions = pd.read_csv(path+"interactions.csv")
test_users = pd.read_csv(path+"recommendations.csv")




def age_group_category(age):
    if age < 30:
        return "Young"
    elif 30 <= age <= 45:
        return "Middle_Aged"
    else:
        return "Senior"

users["Age_Marital"] = users.apply(lambda x: f"{age_group_category(x['Age'])}_{x['Marital_Status']}", axis=1)
users["Age_Marital"].value_counts()


users["UP_Bihar"] = users["State"].apply(lambda x: 1 if x in ["Uttar Pradesh", "Bihar"] else 0)
users["UP_Delhi"] = users["State"].apply(lambda x: 1 if x in ["Uttar Pradesh", "Delhi"] else 0)
users["MH_KA"] = users["State"].apply(lambda x: 1 if x in ["Maharashtra", "Karnataka"] else 0)
users["MH_Delhi"] = users["State"].apply(lambda x: 1 if x in ["Delhi", "Maharashtara"] else 0)


users.head()


categorical_features = ['Marital_Status', 'Caste', 'State', 'Age_Marital']
additional_features = users[['UP_Bihar', 'UP_Delhi', 'MH_KA','MH_Delhi']].values

encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded_data = encoder.fit_transform(users[categorical_features])

scaler = StandardScaler()
users['Age'] = scaler.fit_transform(users[['Age']])
users["Age"].head()



X = np.hstack((users[['Age']].values, encoded_data, additional_features))
knn = NearestNeighbors(n_neighbors=5000, metric='manhattan') 
knn.fit(X)
interaction_map = interactions.groupby("Member_ID")["Target_ID"].apply(set).to_dict()



def recommend_profiles(user_id, top_n=100):
    if user_id not in users['Member_ID'].values:
        return ""  

    user_index = users[users['Member_ID'] == user_id].index[0]
    distances, indices = knn.kneighbors([X[user_index]])

    user_gender = users.loc[user_index, 'Gender']
    opposite_gender = "Male" if user_gender == "Female" else "Female"
    user_sect = users.loc[user_index, 'Sect']

    recommended_profiles = []
    interacted_profiles = interaction_map.get(user_id, set())  

    same_sect_profiles = []
    other_sect_profiles = []

    for i in indices[0]:
        target_id = users.loc[i, 'Member_ID']
        target_sect = users.loc[i, 'Sect']
        
        if users.loc[i, 'Gender'] == opposite_gender and target_id not in interacted_profiles:
            if target_sect == user_sect:
                same_sect_profiles.append(target_id)
            else:
                other_sect_profiles.append(target_id)
            
            if len(same_sect_profiles) >= top_n:
                break
    recommended_profiles = same_sect_profiles[:top_n]
    if len(recommended_profiles) < top_n:
        remaining_slots = top_n - len(recommended_profiles)
        recommended_profiles.extend(other_sect_profiles[:remaining_slots])
    return ",".join(map(str, recommended_profiles))



test_users["top_100_profiles"] = test_users["Member_ID"].apply(recommend_profiles)

test_users["ID"] = test_users["Member_ID"]

test_users[["ID", "Member_ID", "top_100_profiles"]].to_csv("submission.csv", index=False)

print("Recommendations generated")



df = pd.read_csv("/kaggle/working/submission.csv")
df.head()

