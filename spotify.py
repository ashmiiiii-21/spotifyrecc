import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

file_name = 'spotify_dataset.csv.csv'
df = pd.read_csv(file_name)

df = df.dropna()
df = df.drop_duplicates()

audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

df[audio_features] = df[audio_features].apply(pd.to_numeric, errors='coerce')

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[audio_features])
df_scaled = pd.DataFrame(df_scaled, columns=audio_features)

print("\nDescriptive Statistics:")
print(df[audio_features].describe())

plt.figure(figsize=(15, 10))
for i, feature in enumerate(audio_features, 1):
    plt.subplot(3, 4, i)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Histogram of {feature}')
    plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 10))
sns.boxplot(data=df[audio_features])
plt.title('Boxplots of Audio Features')
plt.xticks(rotation=45)
plt.show()

sample_df = df[audio_features].sample(500)
sns.pairplot(sample_df)
plt.suptitle('Pairplot of Audio Features (Sampled)')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(y='playlist_genre', data=df, order=df['playlist_genre'].value_counts().index)
plt.title('Distribution of Playlist Genres')
plt.show()

genre_group = df.groupby('playlist_genre')[audio_features].mean()
genre_group.plot(kind='bar', figsize=(15, 8))
plt.title('Average Audio Features by Playlist Genre')
plt.ylabel('Average Value')
plt.xticks(rotation=0)
plt.show()

top_subgenres = df['playlist_subgenre'].value_counts().head(10).index
subgenre_group = df[df['playlist_subgenre'].isin(top_subgenres)].groupby('playlist_subgenre')[audio_features].mean()
subgenre_group.plot(kind='bar', figsize=(15, 8))
plt.title('Average Audio Features by Top Playlist Subgenres')
plt.ylabel('Average Value')
plt.xticks(rotation=45)
plt.show()

corr_matrix = df[audio_features].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Audio Features')
plt.show()

inertia = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(df_scaled, kmeans.labels_))

plt.figure(figsize=(10, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(k_range, sil_scores, marker='o')
plt.title('Silhouette Scores for Different K')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

optimal_k = 6
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
df['pca1'] = df_pca[:, 0]
df['pca2'] = df_pca[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=df, palette='viridis')
plt.title(f'Clusters Visualized with PCA (K={optimal_k})')
plt.show()

plt.figure(figsize=(10, 8))
sns.scatterplot(x='pca1', y='pca2', hue='playlist_genre', data=df, palette='Set2')
plt.title('PCA Scatterplot Colored by Playlist Genres')
plt.show()

confusion_like = pd.crosstab(df['cluster'], df['playlist_genre'])
print("\nConfusion Matrix (Clusters vs Playlist Genres):")
print(confusion_like)

plt.figure(figsize=(10, 6))
sns.heatmap(confusion_like, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix: Clusters vs Playlist Genres')
plt.show()

top_subgenres_df = df[df['playlist_subgenre'].isin(top_subgenres)]
sub_confusion = pd.crosstab(top_subgenres_df['cluster'], top_subgenres_df['playlist_subgenre'])
plt.figure(figsize=(12, 8))
sns.heatmap(sub_confusion, annot=True, cmap='Greens', fmt='d')
plt.title('Confusion Matrix: Clusters vs Top Playlist Subgenres')
plt.show()

def recommend_songs(track_name, n_recommendations=5):
    if track_name not in df['track_name'].values:
        return "Track not found."
    
    cluster = df[df['track_name'] == track_name]['cluster'].values[0]
    recommendations = df[(df['cluster'] == cluster) & (df['track_name'] != track_name)]
    return recommendations[['track_name', 'track_artist', 'playlist_genre']].sample(n_recommendations)

example_track = df['track_name'].iloc[0]
print(f"\nRecommendations for '{example_track}':")
print(recommend_songs(example_track))

print("\nFinal Insights:")
print("- Clustering groups songs based on audio features for genre segmentation.")
print(f"- With {optimal_k} clusters, alignment with genres shown in confusion matrix.")
print("- Model enables recommendations by cluster similarity.")
