import numpy as np
import os
from pathlib import Path
import sys
import logging

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from src.clustering import DialogClustering
from src.llminteraction import load_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_clustering(embeddings: np.ndarray, original_texts: list):
    """
    Example usage of the DialogClustering class.
    
    Args:
        embeddings: Embeddings for clustering.
        original_texts: Original texts corresponding to the embeddings (optional).
    """
    try:
        # Initialize the clusterer
        clusterer = DialogClustering(embeddings, original_texts)
        
        # Find the optimal clustering method
        logger.info("Starting the search for the optimal clustering method...")
        best_result = clusterer.find_best_clustering()
        
        # Output the results
        logger.info("\nClustering results:")
        logger.info(f"Best method: {best_result.method}")
        logger.info(f"Parameters: {best_result.params}")
        logger.info(f"Quality metrics:")
        for metric, value in best_result.metrics.items():
            logger.info(f"- {metric}: {value:.3f}")
            
        # 2D visualization
        logger.info("2D visualization...")
        clusterer.visualize_clusters(n_components=2)
        
        # 3D visualization
        logger.info("3D visualization...")
        clusterer.visualize_clusters(n_components=3)
        
        # Cluster size distribution
        logger.info("luster size distribution...")
        clusterer.visualize_cluster_distribution()
        representatives = clusterer.get_cluster_representatives(n_representatives=3)
                    
        return clusterer, best_result, representatives
        
    except Exception as e:
        logger.error(f"Clusterization error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Setup paths
        embeddings_dir = project_root / 'embeddings'
        embeddings_dir.mkdir(exist_ok=True)
        file_name = 'har1_mts_dialogue-clinical.parquet'
        file_path = embeddings_dir / file_name
        data_df = load_file(str(file_path))
        
        # Convert embeddings from list to numpy array
        embeddings = np.array(data_df['embedding'].tolist())
        original_data = data_df['dialogue'].tolist()
        clusterer, result = run_clustering(embeddings, original_data)
        

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        sys.exit(1)
