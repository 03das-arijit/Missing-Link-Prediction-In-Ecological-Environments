import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import networkx as nx
import numpy as np
import seaborn as sns
!pip install --upgrade matplotlib
import networkx as nx
import matplotlib.pyplot as plt

#extracting data from Csv
data = pd.read_csv('SA02601_v6_new.csv')
#number of times each unique value appears in the column
data['PLTSP_CODE'].value_counts()
#number of times each unique value appears in the column
data['VISSP_CODE'].value_counts()
# will return a count of missing values in each column of your DataFrame.
data.isnull().sum()
# Dropping rows with null values
data = data.dropna()
#will return a tuple with two values: the number of rows and the number of columns in your DataFrame.
data.shape
#number of times each unique value appears in the column
data['VISSP_CODE'].value_counts()
#number of times each unique value appears in the column
data['PLTSP_CODE'].value_counts()


# Create a directed graph
G = nx.DiGraph()
# Add edges to the graph
for index, row in data.iterrows():
    G.add_edge(row['VISSP_CODE'], row['PLTSP_CODE'])
# Draw the graph
# The following line has been changed to use plt.gca()
fig, ax = plt.subplots()
nx.draw(G, with_labels=True, ax=ax)
plt.show()


#nx.degree_centrality() calculates the degree centrality of each node in the graph.
G = nx.from_pandas_edgelist(data[['PLTSP_CODE', 'VISSP_CODE']], source='PLTSP_CODE', target='VISSP_CODE', create_using=nx.DiGraph())
# Calculate degree centrality
degree_centrality = nx.degree_centrality(G)
print("Degree Centrality: ", degree_centrality)
# Find the maximum value of degree centrality
max_value_dc = max(degree_centrality.values())
print("Max value of degree centrality: ", max_value_dc)


# Calculate betweenness centrality
betweenness_centrality = nx.betweenness_centrality(G)
print("Betweenness Centrality: ", betweenness_centrality)
# Find the maximum value of betweenness centrality
max_value_bc = max(betweenness_centrality.values())
print("Max value of betweenness centrality: ", max_value_bc)


# Remove self-loops from the graph
G.remove_edges_from(nx.selfloop_edges(G))
# Calculate k-core centrality
k_core = list(nx.core_number(G).values())
print("k-core centrality: ", k_core)
# Convert k-core centrality to a NumPy array
k_core_array = np.array(k_core)
# Find the node(s) with the maximum k-core centrality
max_value_kc = np.max(k_core_array)
print(f"Maximum value in k_core: {max_value_kc}")


G = nx.from_pandas_edgelist(data, 'PLTSP_CODE', 'VISSP_CODE')
# Calculate closeness centrality
closeness_centrality = nx.closeness_centrality(G)
print("Closeness centrality: ",closeness_centrality)
# Find the maximum value of closeness centrality
max_value_cc = max(closeness_centrality.values())
print("Max value of closeness centrality: ", max_value_cc)


# Calculate PageRank
pagerank = nx.pagerank(G)
print("PageRank: ", pagerank)
# Find the maximum value of PageRank
max_value_pr = max(pagerank.values())
print("Max value of Pagerank: ", max_value_pr)


#to extract the column number of PLTSP_CODE
data.columns.get_loc('PLTSP_CODE')
#to extract the column number of VISSP_CODE
data.columns.get_loc('VISSP_CODE')
data.info()


# Extracting Edge Data from the DataFrame
X = data[['PLTSP_CODE', 'VISSP_CODE']].values
X = X.astype(str)
#Creating an Undirected Graph
G = nx.Graph()
G.add_edges_from(X)
#Performing Community Detection using Greedy Modularity
clusters = nx.algorithms.community.greedy_modularity_communities(G)
#Printing the Clusters
for i, cluster in enumerate(clusters):
  print(f"Cluster {i + 1}: {cluster}")


#Selecting Columns from the DataFrame
sc = data[['PLTSP_CODE', 'VISSP_CODE']]
#Creating a Pivot Table
pivot_table = sc.pivot_table(index='PLTSP_CODE', columns='VISSP_CODE', aggfunc=len)
#Creating a Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt='g')
#Displaying the Heatmap
plt.show()


# Initialize a bipartite graph
B = nx.Graph()
# Add nodes with the node attribute "bipartite"
vissp_nodes = data['VISSP_CODE'].unique()
pltsp_nodes = data['PLTSP_CODE'].unique()
B.add_nodes_from(vissp_nodes, bipartite=0)
B.add_nodes_from(pltsp_nodes, bipartite=1)
# Add edges between VISSP_CODE and PLTSP_CODE based on the DataFrame
for _, row in data.iterrows():
    B.add_edge(row['VISSP_CODE'], row['PLTSP_CODE'])
# Position nodes
pos = {}
pos.update((node, (1, index)) for index, node in enumerate(vissp_nodes))  # X-axis = 1
pos.update((node, (2, index)) for index, node in enumerate(pltsp_nodes))  # X-axis = 2
# Draw the bipartite graph
fig, ax = plt.subplots()  # Create a figure and an axes object explicitly
nx.draw(B, pos, with_labels=True, node_color=['skyblue']*len(vissp_nodes) + ['lightgreen']*len(pltsp_nodes), node_size=500, font_size=12, font_color='black', ax=ax) # Pass the axes object to nx.draw
plt.show()


#bi-cluster of VISSP_CODE and PLTSP_CODE for each unique value of PLOT_ID
import pandas as pd
# Group the data by PLOT_ID and create a bi-cluster for each group
grouped_data = data.groupby('PLOT_ID')
bi_clusters = {}
for plot_id, group_data in grouped_data:
    # Create a bi-cluster of VISSP_CODE and PLTSP_CODE
    bi_cluster = pd.crosstab(group_data['VISSP_CODE'], group_data['PLTSP_CODE'])
    bi_clusters[plot_id] = bi_cluster
# Print the bi-clusters
for plot_id, bi_cluster in bi_clusters.items():
    print(f'Bi-cluster for PLOT_ID {plot_id}')
    print(bi_cluster)
#a seperate csv file for each clustered PLOT_ID
for plot_id, bi_cluster in bi_clusters.items():
    filename = f'bi_cluster_plot_id_{plot_id}.csv'
    bi_cluster.to_csv(filename)
#calculate number of csv produced
num_csv_files = len(bi_clusters)
print(f'Number of CSV files produced: {num_csv_files}')



from sklearn.metrics.pairwise import cosine_similarity
interaction_matrix = pd.crosstab(data['VISSP_CODE'], data['PLTSP_CODE']) # Create interaction matrix
# Compute cosine similarity between VISSP_CODEs
cosine_sim = cosine_similarity(interaction_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=interaction_matrix.index, columns=interaction_matrix.index)


from sklearn.metrics.pairwise import cosine_similarity
# Compute cosine similarity between VISSP_CODEs
cosine_sim = cosine_similarity(interaction_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=interaction_matrix.index, columns=interaction_matrix.index)
# Function to get top similar VISSP_CODEs and predict links
def get_top_predicted_links_collaborative(vissp_code, top_n=5):
    # Get similarity scores for the given VISSP_CODE
    sim_scores = cosine_sim_df[vissp_code]
    # Get the most similar VISSP_CODEs
    similar_vissp_codes = sim_scores.sort_values(ascending=False).index.tolist()
    # Collect the links associated with the most similar VISSP_CODEs
    predicted_links = pd.Series(0, index=interaction_matrix.columns)
    for similar_vissp in similar_vissp_codes:
        predicted_links += interaction_matrix.loc[similar_vissp]
    # Filter out existing links and get the top N new links
    existing_links = interaction_matrix.loc[vissp_code]
    predicted_links = predicted_links[existing_links == 0].sort_values(ascending=False).head(top_n)
    return predicted_links
# Example prediction for a specific VISSP_CODE
example_vissp_code = 'CHRYFASC'  # Replace with an actual VISSP_CODE from your dataset
predicted_links_collaborative = get_top_predicted_links_collaborative(example_vissp_code)
print(f"Predicted links for {example_vissp_code} using collaborative filtering:")
print(predicted_links_collaborative)



from sklearn.metrics.pairwise import cosine_similarity
# Compute cosine similarity between VISSP_CODEs
cosine_sim = cosine_similarity(interaction_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim, index=interaction_matrix.index, columns=interaction_matrix.index)
# Function to get top similar VISSP_CODEs and predict links
def get_top_predicted_links_collaborative(vissp_code, top_n=3):
    # Get similarity scores for the given VISSP_CODE
    sim_scores = cosine_sim_df[vissp_code]
    # Get the most similar VISSP_CODEs
    similar_vissp_codes = sim_scores.sort_values(ascending=False).index.tolist()
    # Collect the links associated with the most similar VISSP_CODEs
    predicted_links = pd.Series(0, index=interaction_matrix.columns)
    for similar_vissp in similar_vissp_codes:
        predicted_links += interaction_matrix.loc[similar_vissp]
    # Filter out existing links and get the top N new links
    existing_links = interaction_matrix.loc[vissp_code]
    predicted_links = predicted_links[existing_links == 0].sort_values(ascending=False).head(top_n)
    return predicted_links
# Initialize a dictionary to hold predictions for all VISSP_CODEs
all_predictions = {}
# Iterate over each unique VISSP_CODE in the dataset
for vissp_code in interaction_matrix.index:
    predicted_links_collaborative = get_top_predicted_links_collaborative(vissp_code)
    all_predictions[vissp_code] = predicted_links_collaborative
# Print the predicted links for all VISSP_CODEs
for vissp_code, predictions in all_predictions.items():
    print(f"Predicted links for {vissp_code} using collaborative filtering:")
    print(predictions)
    print("\n")



import matplotlib.pyplot as plt
# Initialize a bipartite graph
B = nx.Graph()
# Add nodes with the node attribute "bipartite"
vissp_nodes = data['VISSP_CODE'].unique()
pltsp_nodes = data['PLTSP_CODE'].unique()
B.add_nodes_from(vissp_nodes, bipartite=0)
B.add_nodes_from(pltsp_nodes, bipartite=1)
# Add edges between VISSP_CODE and PLTSP_CODE based on the DataFrame
for _, row in data.iterrows():
    B.add_edge(row['VISSP_CODE'], row['PLTSP_CODE'])
# Position nodes
pos = {}
pos.update((node, (1, index)) for index, node in enumerate(vissp_nodes))  # X-axis = 1
pos.update((node, (2, index)) for index, node in enumerate(pltsp_nodes))  # X-axis = 2
# Draw the bipartite graph using nx.draw_networkx
nx.draw_networkx(B, pos, with_labels=True, node_color=['skyblue']*len(vissp_nodes) + ['lightgreen']*len(pltsp_nodes), node_size=500, font_size=12, font_color='black')
plt.show()


# Bi cluster including these missing link values
import pandas as pd
# Assuming 'all_predictions' contains predicted links for each VISSP_CODE
# Iterate over PLOT_IDs and update bi-clusters
for plot_id, bi_cluster in bi_clusters.items():
    plot_data = grouped_data.get_group(plot_id)
    vissp_codes_in_plot = plot_data['VISSP_CODE'].unique()
    for vissp_code in vissp_codes_in_plot:
        predictions = all_predictions.get(vissp_code, pd.Series(0, index=bi_cluster.columns))
        for pltsp_code, score in predictions.items():
            if score > 0:
                if pltsp_code not in bi_cluster.columns:
                    bi_cluster[pltsp_code] = 0  # Add new column for missing PLTSP_CODE
                bi_cluster.loc[vissp_code, pltsp_code] += score  # Update the count
# Print the updated bi-clusters
for plot_id, bi_cluster in bi_clusters.items():
    print(f'Updated Bi-cluster for PLOT_ID {plot_id}')
    print(bi_cluster)


# Bipartite graph for each PLOT_ID
import matplotlib.pyplot as plt
# Iterate through each PLOT_ID and its corresponding bi-cluster
for plot_id, bi_cluster in bi_clusters.items():
    # Create a bipartite graph for the current PLOT_ID
    B_plot = nx.Graph()
    # Add nodes from the bi-cluster
    vissp_nodes = bi_cluster.index
    pltsp_nodes = bi_cluster.columns
    B_plot.add_nodes_from(vissp_nodes, bipartite=0)
    B_plot.add_nodes_from(pltsp_nodes, bipartite=1)
    # Add edges based on the bi-cluster values
    for vissp_code in vissp_nodes:
        for pltsp_code in pltsp_nodes:
            value = bi_cluster.loc[vissp_code, pltsp_code]
            if value > 0:
                B_plot.add_edge(vissp_code, pltsp_code, weight=value)
    # Position nodes
    pos = {}
    pos.update((node, (1, index)) for index, node in enumerate(vissp_nodes))  # X-axis = 1
    pos.update((node, (2, index)) for index, node in enumerate(pltsp_nodes))  # X-axis = 2
    # Draw the bipartite graph for the current PLOT_ID
    plt.figure()  # Create a new figure for each plot
    nx.draw(B_plot, pos, with_labels=True, node_color=['skyblue']*len(vissp_nodes) + ['lightgreen']*len(pltsp_nodes), node_size=500, font_size=12, font_color='black')
    plt.title(f"Bipartite Graph for PLOT_ID {plot_id}")
    plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
# Step 1: Map VISSP_CODE and PLTSP_CODE to indices
vissp_to_index = {vissp_code: idx for idx, vissp_code in enumerate(interaction_matrix.index)}
index_to_vissp = {idx: vissp_code for vissp_code, idx in vissp_to_index.items()}
pltsp_to_index = {pltsp_code: idx for idx, pltsp_code in enumerate(interaction_matrix.columns)}
index_to_pltsp = {idx: pltsp_code for pltsp_code, idx in pltsp_to_index.items()}
# Step 2: Split the interaction matrix into train and test sets
def train_test_split_matrix(interaction_matrix, test_size=0.2):
    test_mask = np.random.rand(*interaction_matrix.shape) < test_size
    train_matrix = interaction_matrix.copy()
    train_matrix[test_mask] = 0
    test_matrix = np.zeros_like(interaction_matrix)
    test_matrix[test_mask] = interaction_matrix[test_mask]
    return train_matrix, test_matrix

train_matrix, test_matrix = train_test_split_matrix(interaction_matrix.values, test_size=0.2)
# Recompute the similarity on the training set
cosine_sim_train = cosine_similarity(train_matrix)
cosine_sim_train_df = pd.DataFrame(cosine_sim_train, index=interaction_matrix.index, columns=interaction_matrix.index)
# Step 3: Predict missing links on the training set
def get_top_predicted_links_collaborative(vissp_code, top_n=5):
    sim_scores = cosine_sim_train_df[vissp_code]
    similar_vissp_codes = sim_scores.sort_values(ascending=False).index.tolist()
    predicted_links = pd.Series(0, index=interaction_matrix.columns)
    for similar_vissp in similar_vissp_codes:
        similar_vissp_index = vissp_to_index[similar_vissp]  # Get the index of the similar VISSP_CODE
        predicted_links += train_matrix[similar_vissp_index, :]
    vissp_index = vissp_to_index[vissp_code]
    existing_links = train_matrix[vissp_index, :]
    predicted_links = predicted_links[existing_links == 0].sort_values(ascending=False).head(top_n)
    return predicted_links
# Step 4: Generate predictions for the test set
def predict_all_links():
    all_predictions = {}
    for vissp_code in cosine_sim_train_df.index:
        all_predictions[vissp_code] = get_top_predicted_links_collaborative(vissp_code)
    return all_predictions
# Get the predicted links
all_predictions = predict_all_links()
# Step 5: Evaluate accuracy by comparing predicted links with the actual test links
def evaluate_accuracy(predictions, test_matrix, threshold=0):
    y_true = []
    y_pred = []
    for vissp_code, predicted_links in predictions.items():
        for pltsp_code, score in predicted_links.items():
            vissp_index = vissp_to_index[vissp_code]
            pltsp_index = pltsp_to_index[pltsp_code]
            actual = test_matrix[vissp_index, pltsp_index]
            y_true.append(actual)
            y_pred.append(int(score > threshold))
    return y_true, y_pred
y_true, y_pred = evaluate_accuracy(all_predictions, test_matrix)
# Calculate evaluation metrics
#precision = precision_score(y_true, y_pred)
#recall = recall_score(y_true, y_pred)
#f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='micro') # Change the average parameter
recall = recall_score(y_true, y_pred, average='micro') # Change the average parameter
f1 = f1_score(y_true, y_pred, average='micro') # Change the average parameter

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")


# Node2Vec
!pip install node2vec
import networkx as nx
from node2vec import Node2Vec
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# Create the graph
B = nx.Graph()
for _, row in data.iterrows():
    B.add_edge(row['VISSP_CODE'], row['PLTSP_CODE'])
# Generate embeddings using Node2Vec
node2vec = Node2Vec(B, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
# Predict missing links using cosine similarity of embeddings
def predict_links_node2vec(vissp_code):
    predictions = {}
    vissp_vec = model.wv[vissp_code]
    for pltsp_code in data['PLTSP_CODE'].unique():
        pltsp_vec = model.wv[pltsp_code]
        score = cosine_similarity([vissp_vec], [pltsp_vec])[0][0]
        predictions[pltsp_code] = score
    return pd.Series(predictions).sort_values(ascending=False)
# Example: Predict links for CHRYFASC
predicted_links = predict_links_node2vec('CHRYFASC')
print(predicted_links.head())


import networkx as nx
from node2vec import Node2Vec
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# Assuming 'data' is your DataFrame with columns 'VISSP_CODE', 'PLTSP_CODE'
# Initialize the bipartite graph
B = nx.Graph()
# Add edges between VISSP_CODE and PLTSP_CODE based on the DataFrame
for _, row in data.iterrows():
    B.add_edge(row['VISSP_CODE'], row['PLTSP_CODE'])
# Generate embeddings using Node2Vec
node2vec = Node2Vec(B, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
# Function to predict links for all VISSP_CODE
def predict_all_links_node2vec():
    all_predictions = []
    vissp_codes = data['VISSP_CODE'].unique()
    pltsp_codes = data['PLTSP_CODE'].unique()
    for vissp_code in vissp_codes:
        predictions = {}
        vissp_vec = model.wv[vissp_code]
        for pltsp_code in pltsp_codes:
            pltsp_vec = model.wv[pltsp_code]
            score = cosine_similarity([vissp_vec], [pltsp_vec])[0][0]
            predictions[pltsp_code] = score
        predictions_df = pd.DataFrame(list(predictions.items()), columns=['PLTSP_CODE', 'score'])
        predictions_df['VISSP_CODE'] = vissp_code
        all_predictions.append(predictions_df)
    # Combine all predictions into a single DataFrame
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    return all_predictions_df
# Get predictions for all VISSP_CODE
all_predictions_df = predict_all_links_node2vec()
# Display the results
print(all_predictions_df)


import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from node2vec import Node2Vec
# Assuming 'data' is your DataFrame with columns 'VISSP_CODE', 'PLTSP_CODE', 'PLOT_ID'
# Initialize the bipartite graph
B = nx.Graph()
# Add edges between VISSP_CODE and PLTSP_CODE based on the DataFrame
for _, row in data.iterrows():
    B.add_edge(row['VISSP_CODE'], row['PLTSP_CODE'])
# Generate embeddings using Node2Vec
node2vec = Node2Vec(B, dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
# Function to predict links for each PLOT_ID and generate bipartite graphs
def generate_bipartite_graphs_by_plot_id():
    vissp_codes = data['VISSP_CODE'].unique()
    pltsp_codes = data['PLTSP_CODE'].unique()
    # Group by PLOT_ID and process each group separately
    for plot_id, group in data.groupby('PLOT_ID'):
        B_plot = nx.Graph()
        predictions = []
        for vissp_code in group['VISSP_CODE'].unique():
            vissp_vec = model.wv[vissp_code]
            for pltsp_code in pltsp_codes:
                pltsp_vec = model.wv[pltsp_code]
                score = cosine_similarity([vissp_vec], [pltsp_vec])[0][0]
                predictions.append((vissp_code, pltsp_code, score))

        # Convert predictions to DataFrame for easier manipulation
        predictions_df = pd.DataFrame(predictions, columns=['VISSP_CODE', 'PLTSP_CODE', 'score'])

        # Filter based on a threshold (e.g., score > 0.5 for link prediction)
        filtered_predictions = predictions_df[predictions_df['score'] > 0.5]

        # Create a bipartite graph for the filtered predictions
        for _, row in filtered_predictions.iterrows():
            B_plot.add_edge(row['VISSP_CODE'], row['PLTSP_CODE'])

        # Plot the bipartite graph for the current PLOT_ID
        plot_bipartite_graph(B_plot, plot_id)
# Function to visualize the bipartite graph
def plot_bipartite_graph(B_plot, plot_id):
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(B_plot)  # Layout for the graph
    # Extract the two sets of nodes
    vissp_nodes = [n for n in B_plot.nodes() if n in data['VISSP_CODE'].values]
    pltsp_nodes = [n for n in B_plot.nodes() if n in data['PLTSP_CODE'].values]
    # Draw the nodes and edges
    nx.draw_networkx_nodes(B_plot, pos, nodelist=vissp_nodes, node_color='lightblue', label='VISSP_CODE')
    nx.draw_networkx_nodes(B_plot, pos, nodelist=pltsp_nodes, node_color='lightgreen', label='PLTSP_CODE')
    nx.draw_networkx_edges(B_plot, pos, edge_color='gray')
    nx.draw_networkx_labels(B_plot, pos)
    plt.title(f'Bipartite Graph for PLOT_ID: {plot_id}')
    plt.legend(loc='best')
    plt.show()
# Run the function to generate and plot bipartite graphs for each PLOT_ID
generate_bipartite_graphs_by_plot_id()


# Extract true links from your original data
true_links = set()
for _, row in data.iterrows():
    true_links.add((row['VISSP_CODE'], row['PLTSP_CODE']))
# Extract predicted links (assuming a threshold of 0.5 for prediction)
predicted_links = set()
for _, row in all_predictions_df[all_predictions_df['score'] > 0.5].iterrows():
    predicted_links.add((row['VISSP_CODE'], row['PLTSP_CODE']))
# Calculate accuracy metrics
correct_predictions = predicted_links.intersection(true_links)
accuracy = len(correct_predictions) / len(predicted_links) if predicted_links else 0  # Avoid division by zero
precision = len(correct_predictions) / len(predicted_links) if predicted_links else 0
recall = len(correct_predictions) / len(true_links) if true_links else 0
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


pip install networkx numpy pandas scikit-learn
pip install lightfm
#Bayesian prediction
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
# Step 1: Prepare data
def prepare_bpr_data(data):
    dataset = Dataset()
    dataset.fit(data['VISSP_CODE'], data['PLTSP_CODE'])
    (interactions, weights) = dataset.build_interactions((row['VISSP_CODE'], row['PLTSP_CODE']) for _, row in data.iterrows())
    return dataset, interactions
# Step 2: Train BPR model
def train_bpr_model(interactions, epochs=30):
    model = LightFM(no_components=30, loss='bpr')
    model.fit(interactions, epochs=epochs, num_threads=4)
    return model
# Step 3: Predict missing links with error handling
def predict_missing_links(model, dataset, vissp_code, pltsp_codes, top_n=5):
    vissp_id = dataset.mapping()[0].get(vissp_code, None)
    if vissp_id is None:
        return []  # If VISSP_CODE not in the mapping, skip
    pltsp_ids = [dataset.mapping()[2].get(code, None) for code in pltsp_codes]
    pltsp_ids = [id for id in pltsp_ids if id is not None]  # Filter out None values
    if not pltsp_ids:
        return []  # If no valid PLTSP_CODEs, skip
    scores = model.predict(vissp_id, pltsp_ids)
    top_items = np.argsort(-scores)[:top_n]
    return [(vissp_code, pltsp_codes[i], scores[i]) for i in top_items]
# Step 4: Get missing links for all VISSP_CODE with error handling
def get_missing_links_for_all(data, top_n=5):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    dataset, interactions = prepare_bpr_data(train_data)
    model = train_bpr_model(interactions)
    all_predictions = []
    for vissp_code in data['VISSP_CODE'].unique():
        all_pltsp_codes = data['PLTSP_CODE'].unique()

        missing_links = predict_missing_links(model, dataset, vissp_code, all_pltsp_codes, top_n=top_n)
        all_predictions.extend(missing_links)

    return pd.DataFrame(all_predictions, columns=['VISSP_CODE', 'PLTSP_CODE', 'Score'])
# Step 5: Print missing links
missing_links = get_missing_links_for_all(data, top_n=5)
print(missing_links)


#precision and recall and f1 score and accuracy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightfm import LightFM
from lightfm.data import Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# Step 1: Prepare data
def prepare_bpr_data(data):
    dataset = Dataset()
    dataset.fit(data['VISSP_CODE'], data['PLTSP_CODE'])
    (interactions, _) = dataset.build_interactions((row['VISSP_CODE'], row['PLTSP_CODE']) for _, row in data.iterrows())
    return dataset, interactions
# Step 2: Train BPR model
def train_bpr_model(interactions, epochs=30):
    model = LightFM(no_components=30, loss='bpr')
    model.fit(interactions, epochs=epochs, num_threads=4)
    return model
# Step 3: Predict missing links with error handling
def predict_missing_links(model, dataset, vissp_code, pltsp_codes, top_n=5):
    vissp_id = dataset.mapping()[0].get(vissp_code, None)
    if vissp_id is None:
        return []
    pltsp_ids = [dataset.mapping()[2].get(code, None) for code in pltsp_codes]
    pltsp_ids = [id for id in pltsp_ids if id is not None]
    if not pltsp_ids:
        return []
    scores = model.predict(vissp_id, pltsp_ids)
    top_items = np.argsort(-scores)[:top_n]
    return [(vissp_code, pltsp_codes[i], scores[i]) for i in top_items]
# Step 4: Evaluate metrics
def evaluate_metrics(predicted_links, test_data):
    y_true = []
    y_pred = []

    for vissp_code, pltsp_code, _ in predicted_links:
        is_link_in_test = test_data[(test_data['VISSP_CODE'] == vissp_code) & (test_data['PLTSP_CODE'] == pltsp_code)].shape[0] > 0
        y_true.append(int(is_link_in_test))
        y_pred.append(1)  # Assuming all predicted links are positive

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy
# Main workflow
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
dataset, interactions = prepare_bpr_data(train_data)
model = train_bpr_model(interactions)
all_predictions = []
for vissp_code in data['VISSP_CODE'].unique():
    all_pltsp_codes = data['PLTSP_CODE'].unique()
    missing_links = predict_missing_links(model, dataset, vissp_code, all_pltsp_codes, top_n=3)
    all_predictions.extend(missing_links)
# Calculate metrics
precision, recall, f1, accuracy = evaluate_metrics(all_predictions, test_data)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")


#Auto Encoder
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
# Step 1: Prepare the Data
# Create a pivot table of interactions
interaction_matrix = data.pivot_table(index='VISSP_CODE', columns='PLTSP_CODE', aggfunc='size', fill_value=0)
# Convert the interaction matrix to numpy array
interaction_matrix_np = interaction_matrix.values
# Step 2: Build the Autoencoder Model
input_dim = interaction_matrix_np.shape[1]  # Number of PLTSP_CODEs
# Define the layers of the autoencoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(64, activation="relu")(input_layer)
encoder = Dense(32, activation="relu")(encoder)
encoder = Dense(16, activation="relu")(encoder)
decoder = Dense(32, activation="relu")(encoder)
decoder = Dense(64, activation="relu")(decoder)
decoder = Dense(input_dim, activation="sigmoid")(decoder)
# Combine into a model
autoencoder = Model(inputs=input_layer, outputs=decoder)
# Compile the model
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# Step 3: Train the Autoencoder
autoencoder.fit(interaction_matrix_np, interaction_matrix_np, epochs=50, batch_size=16, shuffle=True)
# Step 4: Predict Missing Links
reconstructed_matrix = autoencoder.predict(interaction_matrix_np)
# Step 5: Identify Potential Missing Links
def get_top_missing_links(reconstructed_matrix, original_matrix, top_n=5):
    # Subtract the original matrix from the reconstructed to find missing links
    diff_matrix = reconstructed_matrix - original_matrix
    predicted_links = []
    for i in range(diff_matrix.shape[0]):
        # Get the top N predictions
        top_indices = np.argsort(diff_matrix[i])[::-1][:top_n]
        vissp_code = interaction_matrix.index[i]
        for idx in top_indices:
            pltsp_code = interaction_matrix.columns[idx]
            score = diff_matrix[i, idx]
            if score > 0:  # Only consider positive scores as potential links
                predicted_links.append((vissp_code, pltsp_code, score))
    return pd.DataFrame(predicted_links, columns=['VISSP_CODE', 'PLTSP_CODE', 'Score'])
# Get the top 5 missing links for each VISSP_CODE
top_missing_links = get_top_missing_links(reconstructed_matrix, interaction_matrix_np, top_n=5)
# Display the results
print(top_missing_links)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
# Step 1: Prepare the Data
interaction_matrix = data.pivot_table(index='VISSP_CODE', columns='PLTSP_CODE', aggfunc='size', fill_value=0)
interaction_matrix_np = interaction_matrix.values
# Step 2: Check for Data Balance
interaction_count = np.sum(interaction_matrix_np)
if interaction_count == 0:
    print("No interactions found in the data. Check the dataset.")
else:
    # Step 3: Split the Data into Training and Test Sets
    train_matrix, test_matrix = train_test_split(interaction_matrix_np, test_size=0.2, random_state=42)
    # Step 4: Build the Autoencoder Model
    input_dim = train_matrix.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(128, activation="relu")(input_layer)  # Increased complexity
    encoder = Dense(64, activation="relu")(encoder)
    encoder = Dense(32, activation="relu")(encoder)
    encoder = Dense(16, activation="relu")(encoder)
    decoder = Dense(32, activation="relu")(encoder)
    decoder = Dense(64, activation="relu")(decoder)
    decoder = Dense(128, activation="relu")(decoder)
    decoder = Dense(input_dim, activation="sigmoid")(decoder)  # Using sigmoid to output probabilities
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    # Step 5: Train the Autoencoder with More Epochs
    autoencoder.fit(train_matrix, train_matrix, epochs=100, batch_size=16, shuffle=True)
    # Step 6: Predict Missing Links
    reconstructed_matrix = autoencoder.predict(test_matrix)
    # Step 7: Adjust Thresholding and Calculate Accuracy Metrics
    threshold = 0.1  # Lowered threshold to account for weak signals
    test_matrix_binary = (test_matrix > 0).astype(int)
    reconstructed_matrix_binary = (reconstructed_matrix > threshold).astype(int)
    # Flatten the matrices to calculate metrics
    test_matrix_flat = test_matrix_binary.flatten()
    reconstructed_matrix_flat = reconstructed_matrix_binary.flatten()
    # Calculate Precision, Recall, and F1-Score
    precision = precision_score(test_matrix_flat, reconstructed_matrix_flat)
    recall = recall_score(test_matrix_flat, reconstructed_matrix_flat)
    f1 = f1_score(test_matrix_flat, reconstructed_matrix_flat)
    # Output the accuracy metrics
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')


#accuracy
#Adjusting threshold for link prediction
threshold = 0.5
# Binarize the matrices based on the threshold
predicted_links = (reconstructed_matrix > threshold).astype(int)
true_links = (test_matrix > 0).astype(int)
# Flatten the matrices for easier comparison
predicted_links_flat = predicted_links.flatten()
true_links_flat = true_links.flatten()
# Calculate accuracy
accuracy = accuracy_score(true_links_flat, predicted_links_flat)
print(f"Accuracy: {accuracy:.4f}")


