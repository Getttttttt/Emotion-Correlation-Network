import numpy as np
import networkx as nx
import csv
from pymongo import MongoClient
from scipy.stats import pearsonr
import random

def initialize_database():
    """ Initialize MongoDB client and return collection for shortest paths. """
    client = MongoClient('mongodb://localhost:27017')
    db = client['emotion_network_db']
    collection = db['shortest_paths']
    collection.drop()  # Clear existing data
    return collection

def store_shortest_path_lengths(graph, collection):
    """ Store shortest path lengths of all pairs in MongoDB. """
    for source, path_lengths in nx.all_pairs_shortest_path_length(graph):
        for target, length in path_lengths.items():
            collection.insert_one({'source': source, 'target': target, 'length': length})

def get_path_length(source, target, collection):
    """ Retrieve the shortest path length between two nodes from MongoDB. """
    result = collection.find_one({'source': source, 'target': target})
    return result['length'] if result else None

def bootstrap_correlation(data, bootstrap_samples=10000):
    """ Calculate the mean and standard deviation of bootstrapped correlations. """
    bootstrap_correlations = [pearsonr(*zip(*[random.choice(data) for _ in range(len(data))]))[0] for _ in range(bootstrap_samples)]
    return np.mean(bootstrap_correlations), np.std(bootstrap_correlations)

def emotion_correlation(graph, distance_h, collection):
    """ Calculate emotion correlations for pairs at a specific graph distance. """
    emotions = {emotion: [] for emotion in ['anger', 'disgust', 'joy', 'sadness']}
    for i in graph.nodes():
        for j in graph.nodes():
            if i < j:
                path_length = get_path_length(i, j, collection)
                if path_length == distance_h:
                    for index, emotion in enumerate(emotions):
                        emotions[emotion].append([graph.nodes[i]['emotions'][index], graph.nodes[j]['emotions'][index]])
    
    return {emotion: bootstrap_correlation(data) for emotion, data in emotions.items()}

def load_graph():
    """ Load graph data from a file and initialize nodes with emotions. """
    with open('./Data/weibograph.txt', 'r') as file:
        # Read and parse each line, splitting by tabs and processing emotions embedded in brackets.
        data = []
        for line in file:
            parts = line.strip().split('\t')  # Splitting by tab
            user1, user2, weight = parts[0], parts[1], int(parts[2])
            emotions = list(map(int, parts[3].strip('[]').split(',')))  # Remove brackets and split by comma
            data.append((user1, user2, weight, emotions))
    

    graph = nx.Graph()
    for user1, user2, weight, emotions in data:
        graph.add_edge(user1, user2, weight=weight)
        if 'emotions' not in graph.nodes[user1]:
            graph.nodes[user1]['emotions'] = emotions
        if 'emotions' not in graph.nodes[user2]:
            graph.nodes[user2]['emotions'] = emotions

    return graph

def main():
    """ Main function to process data and store results. """
    graph = load_graph()
    collection = initialize_database()
    store_shortest_path_lengths(graph, collection)
    
    results = {}
    for h in range(1, 7):  # Example distances from 1 to 6
        results[h] = emotion_correlation(graph, h, collection)
    
    with open('./Output/Result.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        headers = ['h'] + [f'{emotion} Correlation, {emotion} Error' for emotion in ['Anger', 'Disgust', 'Joy', 'Sadness']]
        writer.writerow(headers)
        for h, correlations in results.items():
            row = [h] + [value for stats in correlations.values() for value in stats]
            writer.writerow(row)

if __name__ == "__main__":
    main()
