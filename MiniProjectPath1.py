import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score


'''
 The following is the starting code for path1 for data reading to make your first step easier.
 'dataset_1' is the clean data for path1.
'''

with open('behavior-performance.txt','r') as f:
    raw_data = [x.strip().split('\t') for x in f.readlines()]
df = pandas.DataFrame.from_records(raw_data[1:],columns=raw_data[0])
df['VidID']       = pandas.to_numeric(df['VidID'])
df['fracSpent']   = pandas.to_numeric(df['fracSpent'])
df['fracComp']    = pandas.to_numeric(df['fracComp'])
df['fracPlayed']  = pandas.to_numeric(df['fracPlayed'])
df['fracPaused']  = pandas.to_numeric(df['fracPaused'])
df['numPauses']   = pandas.to_numeric(df['numPauses'])
df['avgPBR']      = pandas.to_numeric(df['avgPBR'])
df['stdPBR']      = pandas.to_numeric(df['stdPBR'])
df['numRWs']      = pandas.to_numeric(df['numRWs'])
df['numFFs']      = pandas.to_numeric(df['numFFs'])
df['s']           = pandas.to_numeric(df['s'])
dataset_1 = df
#print(dataset_1[15620:25350].to_string()) #This line will print out the first 35 rows of your data


# Step 1 - Filtering out studnets that have less than 5 watched videos
stu_id_count = dataset_1['userID'].value_counts()
stu_5 = stu_id_count[stu_id_count >= 5].index
cleaned_data = dataset_1[dataset_1['userID'].isin(stu_5)]

#print(cleaned_data)

# Step 1 - Creating clusters, standardizing all features, and usign kmeans clustering

# Selecting features
K = cleaned_data[['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']]

# Standardizind Features
scaler = StandardScaler()
K_scaled = scaler.fit_transform(K)

# Kmeans Clustering

kmeans = KMeans(n_clusters=3) # Choosing 3 clusters
cleaned_data['cluster'] = kmeans.fit_predict(K_scaled)

#print(cleaned_data[['userID', 'cluster']])
#print(cleaned_data.head())


# Running a linear regression model with variables x, y

X = cleaned_data[['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']]
Y = cleaned_data['s'] #average

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_prediction = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_prediction)

print("Mean Squared Error: ", mse)

# Using Logistic Regession to find a correlation between student behaviour and performance on a specific question

vid_accuracies = {}

for vid_id in dataset_1['VidID'].unique(): # accessing each unique video and making a precidtion
    specific_video_data = dataset_1[dataset_1['VidID'] == vid_id]

    X = specific_video_data[['fracSpent', 'fracComp', 'fracPaused', 'numPauses', 'avgPBR', 'numRWs', 'numFFs']]
    y = specific_video_data['s']  
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    
    y_pred = model.predict(X_test)
    

    accuracy = accuracy_score(y_test, y_pred)
    vid_accuracies[vid_id] = accuracy




for vid_id, accuracy in vid_accuracies.items():
    print(f"Video ID {vid_id}: Accuracy = {accuracy}")

all_acc = sum(vid_accuracies.values()) 
avg = all_acc/len(vid_accuracies)
print("Average accuracy", avg)
