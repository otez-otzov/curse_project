import os
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# ----------------------------
# Граф и матрица переходов
# ----------------------------
def create_graph():
    G = nx.Graph()

    rooms = [
        'Entrance', 'Exit', 'Hallway', 'Shop1', 'Shop2', 'Shop3', 'Shop4', 'Shop5', 'Shop6',
        'FoodCourt', 'Toilets', 'Elevator', 'Stairs', 'Parking', 'Office1', 'Office2'
    ]
    G.add_nodes_from(rooms)

    connections = [
        ('Entrance', 'Hallway'),
        ('Hallway', 'Shop1'),
        ('Hallway', 'Shop2'),
        ('Hallway', 'Shop3'),
        ('Shop3', 'Shop4'),
        ('Shop4', 'Shop5'),
        ('Shop5', 'Shop6'),
        ('Hallway', 'FoodCourt'),
        ('FoodCourt', 'Toilets'),
        ('Hallway', 'Elevator'),
        ('Elevator', 'Stairs'),
        ('Stairs', 'Parking'),
        ('Parking', 'Office1'),
        ('Office1', 'Office2'),
        ('Exit', 'Shop6'),
        ('Exit', 'Office2')
    ]
    G.add_edges_from(connections)
    return G

def create_transition_matrix(G):
    nodes = list(G.nodes)
    n = len(nodes)
    transition_matrix = np.zeros((n, n))
    for i, node in enumerate(nodes):
        neighbors = list(G.neighbors(node))
        if neighbors:
            prob = 1 / len(neighbors)
            for neighbor in neighbors:
                j = nodes.index(neighbor)
                transition_matrix[i][j] = prob
    return transition_matrix, nodes

# ----------------------------
# Случайное блуждание
# ----------------------------
def random_walk_with_transition_matrix(transition_matrix, nodes, start_node, steps, end_node=None):
    path = [start_node]
    current_node = start_node
    edge_visit_count = {}
    node_visit_count = defaultdict(int)

    current_index = nodes.index(start_node)
    node_visit_count[start_node] += 1

    for _ in range(steps):
        probabilities = transition_matrix[current_index]
        if probabilities.sum() == 0:
            break  # Нет переходов
        next_index = np.random.choice(len(nodes), p=probabilities)
        next_node = nodes[next_index]
        path.append(next_node)

        edge = tuple(sorted((current_node, next_node)))
        edge_visit_count[edge] = edge_visit_count.get(edge, 0) + 1
        node_visit_count[next_node] += 1

        if next_node == end_node:
            break

        current_node = next_node
        current_index = next_index

    return path, edge_visit_count, node_visit_count

# ----------------------------
# Координаты для визуализации
# ----------------------------
node_positions = {
    'Entrance': (1, 3),
    'Hallway': (3, 3),
    'Shop1': (3, 5),
    'Shop2': (5, 5),
    'Shop3': (7, 5),
    'Shop4': (9, 5),
    'Shop5': (11, 5),
    'Shop6': (13, 5),
    'FoodCourt': (3, 1),
    'Toilets': (5, 1),
    'Elevator': (3, 7),
    'Stairs': (5, 7),
    'Parking': (7, 7),
    'Office1': (9, 7),
    'Office2': (11, 7),
    'Exit': (15, 5)
}

floor_boundary = [(-1, -1), (17, -1), (17, 9), (-1, 9), (-1, -1)]

# ----------------------------
# Визуализация плана с путём и тепловой картой
# ----------------------------
def visualize_floorplan(G, path, edge_visit_count, node_visit_count, output_file='static/floorplan.png'):
    fig, ax = plt.subplots(figsize=(12, 6))
    boundary_x, boundary_y = zip(*floor_boundary)
    ax.plot(boundary_x, boundary_y, color='black', linewidth=2)

    # Нарисовать комнаты
    for node, (x, y) in node_positions.items():
        # Цвет комнаты в зависимости от посещаемости (heatmap)
        visits = node_visit_count.get(node, 0)
        # Нормируем для цветовой карты (0-1)
        norm_visits = min(visits / max(1, max(node_visit_count.values())), 1)
        color = plt.cm.Reds(norm_visits)  # Красная гамма

        ax.add_patch(plt.Rectangle((x - 0.4, y - 0.4), 0.8, 0.8, edgecolor='black', facecolor=color))
        ax.text(x, y, f"{node}\n({visits})", fontsize=10, ha='center', va='center')

    # Связи
    for edge in G.edges():
        x1, y1 = node_positions[edge[0]]
        x2, y2 = node_positions[edge[1]]
        ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.7, linewidth=1)

    # Подсветка маршрута
    path_edges = list(zip(path, path[1:]))
    for edge in path_edges:
        x1, y1 = node_positions[edge[0]]
        x2, y2 = node_positions[edge[1]]
        ax.plot([x1, x2], [y1, y2], color='blue', linewidth=2)

    # Кол-во посещений ребер
    for edge, count in edge_visit_count.items():
        if count > 0:
            mid_x = (node_positions[edge[0]][0] + node_positions[edge[1]][0]) / 2
            mid_y = (node_positions[edge[0]][1] + node_positions[edge[1]][1]) / 2
            ax.text(mid_x, mid_y, str(count), fontsize=10, color='green')

    ax.set_xlim(-2, 18)
    ax.set_ylim(-2, 10)
    ax.axis('off')
    plt.title("Floorplan with Path and Heatmap")

    plt.savefig(output_file)
    plt.close(fig)

# ----------------------------
# KDE-анализ плотности посещений
# ----------------------------
def kde_density_analysis(node_visit_count):
    coords = []
    counts = []
    for node, count in node_visit_count.items():
        coords.append(node_positions[node])
        counts.append(count)

    coords = np.array(coords)
    counts = np.array(counts)

    # KDE с весамипосещения
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(coords, sample_weight=counts)

    # Сетка для визуализации KDE
    log_dens = kde.score_samples(coords)
    dens = np.exp(log_dens)
    dens_map = dict(zip(node_visit_count.keys(), dens))
    return dens_map

# ----------------------------
# Кластеризация паттернов
# ----------------------------
def cluster_patterns(path, nodes):
    # Преобразуем путь в координаты
    coords = np.array([node_positions[node] for node in path])

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42).fit(coords)
    labels_kmeans = kmeans.labels_

    # DBSCAN
    dbscan = DBSCAN(eps=1.5, min_samples=2).fit(coords)
    labels_dbscan = dbscan.labels_

    return {
        'kmeans': labels_kmeans.tolist(),
        'dbscan': labels_dbscan.tolist()
    }

# ----------------------------
# Обнаружение аномалий
# ----------------------------
def detect_anomalies(path):
    coords = np.array([node_positions[node] for node in path])

    # One-Class SVM
    oc_svm = OneClassSVM(gamma='scale').fit(coords)
    preds_svm = oc_svm.predict(coords)  # -1 аномалия, 1 норма

    # Isolation Forest
    iso_forest = IsolationForest(random_state=42).fit(coords)
    preds_if = iso_forest.predict(coords)  # -1 аномалия, 1 норма

    # Простое правило: если точка далеко от центра графа (по евклид. расстоянию)
    center = np.mean(coords, axis=0)
    dists = np.linalg.norm(coords - center, axis=1)
    threshold = np.percentile(dists, 90)  # топ 10% по расстоянию - аномалии
    preds_rule = np.array([1 if dist <= threshold else -1 for dist in dists])

    return {
        'one_class_svm': preds_svm.tolist(),
        'isolation_forest': preds_if.tolist(),
        'rule_based': preds_rule.tolist()
    }

# ----------------------------
# Анализ переходов и времени пребывания
# ----------------------------
def analyze_metrics(path, edge_visit_count, node_visit_count, step_time_sec=5):
    # Частота посещений - node_visit_count уже есть

    # Среднее время пребывания (примерно шаг_time_sec * количество посещений)
    avg_time_per_zone = {node: visits * step_time_sec for node, visits in node_visit_count.items()}

    # Форматируем таблицу
    metrics = []
    for node in node_visit_count:
        metrics.append({
            'zone': node,
            'visit_count': node_visit_count[node],
            'avg_time_sec': avg_time_per_zone[node]
        })

    return metrics

# ----------------------------
# Инициализация
# ----------------------------
graph = create_graph()
transition_matrix, nodes = create_transition_matrix(graph)

# ----------------------------
# Flask routes
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/random-walk', methods=['POST'])
def random_walk():
    data = request.json
    start_room = data.get('start_room')
    end_room = data.get('end_room')
    steps = data.get('steps', 10)

    if start_room not in graph.nodes:
        return jsonify({"error": "Invalid start room"}), 400
    if end_room and end_room not in graph.nodes:
        return jsonify({"error": "Invalid end room"}), 400

    path, edge_visit_count, node_visit_count = random_walk_with_transition_matrix(
        transition_matrix, nodes, start_room, steps, end_room
    )

    # Визуализация
    timestamp = int(time.time())
    output_file = f"static/floorplan_{timestamp}.png"
    visualize_floorplan(graph, path, edge_visit_count, node_visit_count, output_file)

    # Анализ плотности KDE
    kde_density = kde_density_analysis(node_visit_count)

    # Кластеризация
    clustering = cluster_patterns(path, nodes)

    # Аномалии
    anomalies = detect_anomalies(path)

    # Метрики
    metrics = analyze_metrics(path, edge_visit_count, node_visit_count)

    edge_visit_counts_str = {
        f"{edge[0]}-{edge[1]}": count for edge, count in edge_visit_count.items() if count > 0
    }

    return jsonify({
        "visualization_url": "/" + output_file,
        "edge_visit_counts": edge_visit_counts_str,
        "node_visit_counts": node_visit_count,
        "path": path,
        "kde_density": kde_density,
        "clustering": clustering,
        "anomalies": anomalies,
        "metrics": metrics
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.mkdir('static')
    app.run(debug=True)
