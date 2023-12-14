import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from datetime import datetime
from sklearn.metrics import silhouette_score
import json
from toolz import groupby


def calDistanceTocenter(data, distances):
    # print(data.index.tolist())
    center = np.mean(data, axis=0)
    distance = np.sqrt(np.sum(np.square(data - center), axis=1))
    distances = pd.concat([distances, pd.DataFrame(distance)], axis=0)

    return distances

#group intents by intent
def group_intents(file_path=r'evaluate.json', limit=50):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())
    	# Filter groups with more than 2 examples
    grouped_data = groupby('intent', data)
    return grouped_data




def main_clustering( intent_data=[], file_path=r'evaluate.json'):
    result = intent_data

    data = []
    data = [item['bert_output'] for item in result]
    data = pd.DataFrame(data)
    print(file_path)
 

    agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=12,
                                             metric='euclidean', linkage='ward')
    labels = agg_clustering.fit_predict(data)

    for i, label in enumerate(labels):
        result[i]['cluster_label'] = datetime.now().strftime('%Y-%m-%d') + '-' + str(label)
        del result[i]['bert_output']
 
 
    
 
    # result['Agglomerative Clustering'] = [datetime.now().strftime('%Y-%m-%d') + '-' + str(i) for i in labels]

    # from sklearn.metrics import silhouette_score
    # silhouette_avg = silhouette_score(data, labels)
    # print("聚类的轮廓系数:", silhouette_avg)

    file_name = file_path.split('/')[-1].split('.')[0]

    # print("Labels:", len(labels),  labels)
    unique_labels = np.unique(labels)
    # print("unique labels:",   unique_labels)
    print(f"unique / total: {len(unique_labels)} / {len(labels)} = { len(unique_labels) / len(labels)}", )
    global total_clusters, total_intents
    total_clusters = 0
    total_intents = 0

    total_clusters = total_clusters + len(unique_labels)
    total_intents += 1
    
    distances = pd.DataFrame()

    for label in unique_labels:
        cluster_samples = data[labels == label]
        distances = calDistanceTocenter(cluster_samples, distances)
     
    distances = distances.sort_index()
    for i in range(len(distances)):
        value = distances.iloc[i, 0]
        result[i]['distance'] = value

    return result


def main(evaluate_data):

    data_dir = r'../cluster/'
    cluster_num = 100
    # data_path = data_dir + '_evaluate.json'
    # data = group_intents(data_path)
    # print((data))

    data = evaluate_data
    data = groupby('intent', data)
    data_clusters = []
    for key, value in data.items():
        if len(value) < 2:
            continue
        cluster = main_clustering(intent_data=value, file_path=key + '.json')
        data_clusters.extend(cluster)
    
    data_clusters_path = os.path.join(data_dir, 'intents_clusters.json')
    with open(data_clusters_path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data_clusters, ensure_ascii=False))

    result_csv=pd.DataFrame(data_clusters)
    data_clusters_csv_path = os.path.join(data_dir, 'intents_clusters.csv')
    result_csv.to_csv(data_clusters_csv_path, index=False)

    return data_clusters
