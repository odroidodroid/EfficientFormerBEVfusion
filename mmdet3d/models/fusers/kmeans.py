import torch

def initialize_centroids_kmeans_plus_plus(X, k):
    centroids = [X[torch.randint(0, X.size(0), (1,)).item()]]
    for _ in range(1, k):
        dist_sq = torch.cdist(X, torch.stack(centroids)).min(dim=1)[0]
        probs = dist_sq / dist_sq.sum()
        cumulative_probs = torch.cumsum(probs, dim=0)
        r = torch.rand(1).item()
        for i, p in enumerate(cumulative_probs):
            if r < p:
                centroids.append(X[i])
                break
    return torch.stack(centroids)

def closest_centroid(X, centroids):
    distances = torch.cdist(X, centroids)
    return torch.argmin(distances, dim=1)

def move_centroids(X, closest, centroids):
    new_centroids = torch.zeros_like(centroids)
    for i in range(centroids.size(0)):
        if (closest == i).sum() > 0:
            new_centroids[i] = X[closest == i].mean(dim=0)
    return new_centroids

def kmeans(X, k, num_iterations):
    centroids = initialize_centroids_kmeans_plus_plus(X, k)
    for _ in range(num_iterations):
        closest = closest_centroid(X, centroids)
        centroids = move_centroids(X, closest, centroids)
    return closest, centroids
