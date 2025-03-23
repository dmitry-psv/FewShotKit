import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, adjusted_mutual_info_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from hdbscan import HDBSCAN
from umap import UMAP
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from kneed import KneeLocator
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from dataclasses import dataclass

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@dataclass
class ClusteringResult:
    """Класс для хранения результатов кластеризации"""
    labels: np.ndarray
    metrics: Dict[str, float]
    method: str
    params: Dict[str, Any]
    
class DialogClustering:
    def __init__(self, embeddings: np.ndarray, original_data: Optional[List] = None, use_scaling: bool = True):
        """
        Инициализация кластеризации с эмбеддингами диалогов
        
        Args:
            embeddings: Массив эмбеддингов диалогов
            original_data: Исходные тексты диалогов для визуализации
            use_scaling: Применять ли стандартное масштабирование к эмбеддингам.
                        По умолчанию True. Установите False, если эмбеддинги уже нормализованы
                        или используется метрика, не требующая масштабирования (например, косинусная близость).
        """
        self._validate_input(embeddings, original_data)
        self.use_scaling = use_scaling
        self.embeddings = self._preprocess_embeddings(embeddings)
        self.original_data = original_data
        self.best_result: Optional[ClusteringResult] = None
        logger.info(f"Инициализирован DialogClustering с {len(embeddings)} эмбеддингами, use_scaling={use_scaling}")

    def _validate_input(self, embeddings: np.ndarray, original_data: Optional[List]) -> None:
        """Валидация входных данных"""
        if embeddings is None or len(embeddings) == 0:
            raise ValueError("Embeddings не могут быть пустыми")
        if not isinstance(embeddings, np.ndarray):
            raise ValueError("Embeddings должны быть numpy массивом")
        if original_data is not None and len(original_data) != len(embeddings):
            raise ValueError("Размер original_data должен совпадать с размером embeddings")

    def _preprocess_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Предобработка эмбеддингов
        
        Args:
            embeddings: Входные эмбеддинги
            
        Returns:
            np.ndarray: Обработанные эмбеддинги
            
        Note:
            По умолчанию применяется StandardScaler. Если ваши эмбеддинги уже нормализованы
            или вы используете метрики, не требующие масштабирования (например, косинусная близость),
            установите параметр use_scaling=False в конструкторе.
        """
        if not hasattr(self, 'use_scaling') or self.use_scaling:
            scaler = StandardScaler()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                return scaler.fit_transform(embeddings)
        return embeddings

    def _evaluate_clustering(self, labels: np.ndarray, data: np.ndarray) -> Dict[str, float]:
        """
        Комплексная оценка качества кластеризации
        
        Args:
            labels: Метки кластеров
            data: Данные для оценки
            
        Returns:
            Dict[str, float]: Словарь метрик качества
        """
        if len(set(labels)) < 2:
            return {"error": -1.0}
            
        # Учитываем метку -1 (шум) как отдельный кластер для вычисления метрик
        if -1 in labels:
            logger.info("Обнаружены точки шума (label = -1), они будут учтены как отдельный кластер")
            
        metrics = {}
        try:
            metrics['silhouette'] = silhouette_score(data, labels)
            metrics['davies_bouldin'] = davies_bouldin_score(data, labels)
            metrics['calinski_harabasz'] = calinski_harabasz_score(data, labels)
            
            # Оценка стабильности кластеризации
            # Для оценки стабильности используем тот же метод кластеризации, что и основной
            if len(set(labels)) > 1:
                # Разделяем данные на две части для оценки стабильности
                n_samples = len(data)
                indices = np.random.permutation(n_samples)
                n_subset = n_samples // 2  # Используем половину данных для каждой части
                
                subset1_idx = indices[:n_subset]
                subset2_idx = indices[n_subset:2*n_subset]  # Берем равные части
                
                X_subset1 = data[subset1_idx]
                X_subset2 = data[subset2_idx]
                
                # Обучаем модель на первой части
                model = KMeans(n_clusters=len(set(labels)), random_state=42)
                model.fit(X_subset1)
                
                # Получаем предсказания для обеих частей
                labels1 = model.predict(X_subset1)
                labels2 = model.predict(X_subset2)
                
                # Оцениваем стабильность на равных по размеру частях
                metrics['stability'] = adjusted_mutual_info_score(labels1, labels2)
            else:
                metrics['stability'] = 0.0
        except Exception as e:
            logger.warning(f"Ошибка при вычислении метрик: {str(e)}")
            
        return metrics

    def _estimate_optimal_clusters(self, data: np.ndarray) -> int:
        """
        Оценка оптимального количества кластеров
        
        Args:
            data: Данные для кластеризации
            
        Returns:
            int: Оптимальное количество кластеров
        """
        # Метод локтя
        # Максимальное количество кластеров зависит от размера данных
        max_clusters = min(
            max(15, int(np.log2(len(data)))),  # Минимум 15, максимум log2(N)
            len(data) // 2  # Не более половины объектов
        )
        logger.info(f"Оценка оптимального количества кластеров: max_clusters={max_clusters}")
        distortions = []
        K = range(2, max_clusters + 1)
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            distortions.append(kmeans.inertia_)
            
        # Используем KneeLocator для автоматического определения точки перегиба
        kl = KneeLocator(
            list(K), distortions, curve='convex', direction='decreasing'
        )
        optimal_k = kl.elbow if kl.elbow else 3
        
        # Проверка через силуэтный анализ
        silhouette_scores = []
        # Проверяем силуэт в диапазоне от 2 до max_clusters
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            try:
                score = silhouette_score(data, labels)
                silhouette_scores.append(score)
            except:
                continue
                
        if silhouette_scores:
            silhouette_k = np.argmax(silhouette_scores) + 2
            # Берем среднее между методами
            optimal_k = int((optimal_k + silhouette_k) / 2)
            
        return optimal_k

    def _optimize_dbscan_params(self, data: np.ndarray) -> Tuple[float, int]:
        """
        Оптимизация параметров DBSCAN
        
        Returns:
            Tuple[float, int]: Оптимальные eps и min_samples
            
        Note:
            Вычисление cdist(data, data) имеет сложность O(N^2) и может быть
            очень ресурсоемким для больших наборов данных. Для N > 10,000
            рекомендуется использовать более эффективные методы.
        """
        if len(data) > 10000:
            logger.warning("Вычисление cdist для больших данных может быть медленным. "
                         "Рекомендуется использовать подвыборку или более эффективные методы.")
        # Автоматический расчет eps на основе распределения расстояний
        distances = cdist(data, data)
        eps_candidates = np.percentile(distances, [5, 10, 15, 20, 25])
        min_samples_candidates = range(3, 10)
        
        best_score = -1
        best_params = (0.5, 5)  # Значения по умолчанию
        
        for eps in eps_candidates:
            for min_samples in min_samples_candidates:
                try:
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    labels = dbscan.fit_predict(data)
                    if len(set(labels)) > 1:  # Проверяем, что есть больше одного кластера
                        score = silhouette_score(data, labels)
                        if score > best_score:
                            best_score = score
                            best_params = (eps, min_samples)
                except:
                    continue
                    
        return best_params

    def _optimize_hdbscan_params(self, data: np.ndarray) -> Tuple[int, int]:
        """
        Оптимизация параметров HDBSCAN
        
        Returns:
            Tuple[int, int]: Оптимальные min_cluster_size и min_samples
            
        Note:
            Оптимизация параметров HDBSCAN может быть вычислительно затратной,
            особенно для больших наборов данных (N > 10,000). Рекомендуется
            использовать подвыборку или ограничить диапазон параметров.
        """
        if len(data) > 10000:
            logger.warning("Оптимизация HDBSCAN для больших данных может быть медленной. "
                         "Рекомендуется использовать подвыборку или ограничить диапазон параметров.")
        min_cluster_sizes = range(5, 20, 2)
        min_samples_multipliers = [1, 2, 3]  # Множители для min_samples
        
        best_score = -1
        best_params = (5, 5)  # Значения по умолчанию
        
        for min_cluster_size in min_cluster_sizes:
            for multiplier in min_samples_multipliers:
                min_samples = min_cluster_size * multiplier
                try:
                    clusterer = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_epsilon=0.0
                    )
                    labels = clusterer.fit_predict(data)
                    if len(set(labels)) > 1:
                        score = silhouette_score(data, labels)
                        if score > best_score:
                            best_score = score
                            best_params = (min_cluster_size, min_samples)
                except:
                    continue
                    
        return best_params

    def find_best_clustering(self, allowed_methods: Optional[Tuple[str]] = None) -> ClusteringResult:
        """
        Поиск наилучшего метода кластеризации и его параметров
        
        Args:
            allowed_methods: Кортеж допустимых методов кластеризации для перебора.
                            Если None, используются все доступные методы.
                            Доступные методы: 'kmeans', 'agglomerative', 'gaussian_mixture',
                            'dbscan', 'hdbscan'
        
        Returns:
            ClusteringResult: Результат лучшей кластеризации
            
        Note:
            Используется комбинированная метрика с весами:
            - silhouette_score: 0.4 (чем больше, тем лучше)
            - davies_bouldin_score: 0.3 (чем меньше, тем лучше, поэтому используется 1 - db)
            - calinski_harabasz_score: 0.1 (чем больше, тем лучше)
            - stability: 0.2 (чем больше, тем лучше)
            
            Эти веса выбраны эмпирически для баланса между компактностью кластеров,
            их разделимостью и стабильностью. При необходимости можно настроить веса
            в зависимости от конкретной задачи.
        """
        # Все доступные методы кластеризации
        all_methods = {
            'kmeans': lambda: KMeans(n_clusters=n_clusters, random_state=42),
            'agglomerative': lambda: AgglomerativeClustering(n_clusters=n_clusters),
            'gaussian_mixture': lambda: GaussianMixture(n_components=n_clusters, random_state=42),
            'dbscan': lambda: DBSCAN(
                eps=self._optimize_dbscan_params(self.embeddings)[0],
                min_samples=self._optimize_dbscan_params(self.embeddings)[1]
            ),
            'hdbscan': lambda: HDBSCAN(
                *self._optimize_hdbscan_params(self.embeddings)
            )
        }
        
        # Если не указаны допустимые методы, используем все
        if allowed_methods is None:
            allowed_methods = tuple(all_methods.keys())
        else:
            # Проверяем, что все указанные методы существуют
            invalid_methods = set(allowed_methods) - set(all_methods.keys())
            if invalid_methods:
                raise ValueError(f"Недопустимые методы кластеризации: {invalid_methods}")
        best_result = None
        best_score = -float('inf')
        
        # Определяем оптимальное количество кластеров
        n_clusters = self._estimate_optimal_clusters(self.embeddings)
        logger.info(f"Оптимальное количество кластеров: {n_clusters}")

        # Создаем список методов для перебора на основе allowed_methods
        clustering_methods = [
            (method_name, all_methods[method_name])
            for method_name in allowed_methods
        ]
        
        for method_name, get_clusterer in clustering_methods:
            try:
                logger.info(f"Пробуем метод: {method_name}")
                clusterer = get_clusterer()
                labels = clusterer.fit_predict(self.embeddings)
                
                # Оцениваем качество
                metrics = self._evaluate_clustering(labels, self.embeddings)
                if 'error' in metrics:
                    continue
                    
                # Используем комбинированную метрику для оценки
                combined_score = (
                    metrics.get('silhouette', 0) * 0.4 +
                    (1 - metrics.get('davies_bouldin', 1)) * 0.3 +
                    metrics.get('calinski_harabasz', 0) * 0.1 +
                    metrics.get('stability', 0) * 0.2
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    params = {}
                    if hasattr(clusterer, 'get_params'):
                        params = clusterer.get_params()
                    best_result = ClusteringResult(
                        labels=labels,
                        metrics=metrics,
                        method=method_name,
                        params=params
                    )
                    
            except Exception as e:
                logger.warning(f"Ошибка при использовании {method_name}: {str(e)}")
                continue
                
        if best_result is None:
            raise ValueError("Не удалось найти подходящий метод кластеризации")
            
        self.best_result = best_result
        logger.info(f"Лучший метод: {best_result.method} с метриками: {best_result.metrics}")
        return best_result

    def visualize_clusters(self, labels: Optional[np.ndarray] = None,
                         n_components: int = 2,
                         interactive: bool = True) -> np.ndarray:
        """
        Визуализация кластеров с использованием UMAP
        
        Args:
            labels: Метки кластеров
            n_components: Количество компонент для визуализации
            interactive: Использовать ли интерактивную визуализацию
            
        Returns:
            np.ndarray: Массив с координатами точек в пространстве UMAP
        """
        if labels is None and self.best_result is not None:
            labels = self.best_result.labels
            
        try:
            reducer = UMAP(n_components=n_components, random_state=42)
            embedding_nd = reducer.fit_transform(self.embeddings)
            
            if interactive:
                hover_data = {}
                if self.original_data is not None:
                    hover_data['text'] = self.original_data
                    
                if n_components == 2:
                    fig = px.scatter(
                        x=embedding_nd[:, 0],
                        y=embedding_nd[:, 1],
                        color=labels if labels is not None else None,
                        hover_data=hover_data,
                        title=f"Визуализация кластеров. Метод: {self.best_result.method}"
                    )
                else:
                    fig = px.scatter_3d(
                        x=embedding_nd[:, 0],
                        y=embedding_nd[:, 1],
                        z=embedding_nd[:, 2],
                        color=labels if labels is not None else None,
                        hover_data=hover_data,
                        title=f" Визуализация кластеров. Метод: {self.best_result.method}"
                    )
                    
                fig.update_traces(
                    marker=dict(size=8),
                    hovertemplate="<br>".join([
                        "Кластер: %{color}",
                        "X: %{x}",
                        "Y: %{y}",
                        "Текст: %{customdata[0]}" if self.original_data is not None else "",
                    ])
                )
                fig.show()
            else:
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(
                    embedding_nd[:, 0],
                    embedding_nd[:, 1],
                    c=labels if labels is not None else None,
                    cmap='viridis'
                )
                if labels is not None:
                    plt.colorbar(scatter, label='Кластер')
                plt.title("Визуализация кластеров")
                plt.xlabel("UMAP компонента 1")
                plt.ylabel("UMAP компонента 2")
                plt.show()
                
            return embedding_nd
            
        except Exception as e:
            logger.error(f"Ошибка в visualize_clusters: {str(e)}")
            raise

    def get_cluster_representatives(self, n_representatives: int = 3) -> Dict[int, List[int]]:
        """
        Получение репрезентативных примеров для каждого кластера
        
        Args:
            n_representatives: Количество представителей от каждого кластера
            
        Returns:
            Dict[int, List[int]]: Словарь с индексами представителей для каждого кластера
        """
        if self.best_result is None:
            raise ValueError("Сначала необходимо выполнить кластеризацию")
            
        representatives = {}
        unique_labels = np.unique(self.best_result.labels)
        
        for label in unique_labels:
            if label == -1:  # Пропускаем выбросы
                continue
                
            # Получаем индексы точек в текущем кластере
            cluster_indices = np.where(self.best_result.labels == label)[0]
            cluster_points = self.embeddings[cluster_indices]
            
            # Находим центроид кластера
            centroid = np.mean(cluster_points, axis=0)
            
            # Вычисляем расстояния до центроида
            distances = cdist([centroid], cluster_points)[0]
            
            # Выбираем точки:
            # 1. Ближайшую к центроиду (типичный представитель)
            # 2. Несколько разнообразных представителей
            central_idx = cluster_indices[np.argmin(distances)]
            
            # Используем максимальное минимальное расстояние для разнообразия
            selected = [central_idx]
            remaining_indices = list(set(cluster_indices) - {central_idx})
            
            while len(selected) < n_representatives and remaining_indices:
                # Находим точку, максимально удаленную от уже выбранных
                max_min_dist = -1
                best_idx = None
                
                for idx in remaining_indices:
                    point = self.embeddings[idx].reshape(1, -1)
                    min_dist = min(cdist(point, self.embeddings[selected])[0])
                    
                    if min_dist > max_min_dist:
                        max_min_dist = min_dist
                        best_idx = idx
                        
                if best_idx is not None:
                    selected.append(best_idx)
                    remaining_indices.remove(best_idx)
                else:
                    break
                    
            representatives[label] = selected
            
        return representatives

    def visualize_cluster_distribution(self) -> None:
        """Визуализация распределения размеров кластеров"""
        if self.best_result is None:
            raise ValueError("Сначала необходимо выполнить кластеризацию")
            
        try:
            unique_labels = np.unique(self.best_result.labels)
            counts = [np.sum(self.best_result.labels == label) for label in unique_labels]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=unique_labels,
                    y=counts,
                    text=counts,
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title=f"Распределение размеров кластеров (метод: {self.best_result.method})",
                xaxis_title="Номер кластера",
                yaxis_title="Количество объектов",
                showlegend=False
            )
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Ошибка в visualize_cluster_distribution: {str(e)}")
            raise