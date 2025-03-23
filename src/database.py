# database.py (обновлённая версия)

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, MilvusClient
import pandas as pd

class VectorDatabase:
    """
    A class encapsulating the logic for interacting with Milvus
    to store e.g.(dialog, embedding, summary) and search for relevant records.
    """

    def __init__(
        self,
        uri: str,
        token: str,
        collection_name: str,
        embedding_dim: int,
        metric_type: str = "IP"  # Inner Product, как одна из популярных (косинус при нормированных векторах)
    ):
        """
        Parameters:
         - uri: URL for connecting to Milvus (e.g., "https://example.com")
         - token: Authentication token
         - collection_name: Name of the collection where records are stored
         - embedding_dim: Dimension of the embeddings (e.g., 768)
         - metric_type: Metric type for search ("IP", "L2", "COSINE", "HAMMING", etc.)
        """
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.metric_type = metric_type

        # Connection ...
        connections.connect(
            alias="default",
            uri=self.uri,
            token=self.token
        )
        
        # Connection check
        if not connections.has_connection("default"):
            raise ConnectionError("Failed to establish connection with Milvus")

        # Milvus Client creation
        self.client = MilvusClient(uri=self.uri, token=self.token)

        # Prepare/create the collection schema
        self._create_collection_if_not_exists()

    def _create_collection_if_not_exists(self):
        """
        Checks the existence of a Milvus collection, and if it does not exist,
        creates it with the necessary fields and index.
        """
        # Check if the collection exists
        has_collection = self.client.has_collection(self.collection_name)
        if not has_collection:
            print(f"Collection '{self.collection_name}' does not exist, creating...")

            dialog_text_field = FieldSchema(
                name="x_label",
                dtype=DataType.VARCHAR,
                max_length=12096,
                is_primary=False,
                description="input text"
            )

            summary_field = FieldSchema(
                name="y_label",
                dtype=DataType.VARCHAR,
                max_length=2048,
                is_primary=False,
                description="output text"
            )

            embedding_field = FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                is_primary=False,
                dim=self.embedding_dim
            )

            # You can add "id" (auto id) as the primary key
            # Milvus v2 allows auto_id=True
            auto_id_field = FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
                description="unique id for each record"
            )

            schema = CollectionSchema(
                fields=[auto_id_field, dialog_text_field, summary_field, embedding_field],
                description="collection for storing data"
            )

            collection = Collection(name=self.collection_name, schema=schema)
            
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": self.metric_type,
                "params": {"nlist": 128}
            }
            collection.create_index(
                field_name="embedding",
                index_params=index_params
            )
            
            collection.load()
            print(f"Collection '{self.collection_name}' created!")
        else:
            collection = Collection(name=self.collection_name)
            collection.load()

    def upsert(
        self,
        input_text: str,
        embedding: list[float],
        output_text: str
    ) -> None:
        """
        Adds a record to Milvus containing:
        - input_text: the original dialog text
        - embedding: the embedding (list[float], length self.embedding_dim)
        - output_text: the resulting summary
        """

        entities = {
            "x_label": [input_text],
            "embedding": [embedding],
            "y_label": [output_text]
        }
        
        collection = Collection(name=self.collection_name)
        collection.insert([entities[field.name] for field in collection.schema.fields if field.name != "id"])        
        collection.flush()
        
    def bulk_upsert(
        self,
        input_texts: list[str],
        embeddings: list[list[float]],
        output_texts: list[str],
        batch_size: int = 100
    ) -> None:
        """
        Bulk insert data into Milvus using batches.
        Args:
            input_texts: list of dialog texts
            embeddings: list of embeddings
            output_texts: list of summaries
            batch_size: size of the batch for insertion
        """
        assert len(input_texts) == len(embeddings) == len(output_texts), "All input lists must have same length"
        
        total_records = len(input_texts)
        total_rt = 0  # total response time for insert
        
        print(f"Inserting {total_records} records into collection: {self.collection_name}")
        
        # Process in batches
        for i in range(0, total_records, batch_size):
            batch_end = min(i + batch_size, total_records)
            
            # Format data as expected by Milvus
            rows = [
                {
                    "x_label": str(input_texts[j]),
                    "embedding": embeddings[j],
                    "y_label": str(output_texts[j])
                }
                for j in range(i, batch_end)
            ]
            
            # Insert batch and measure time
            self.client.insert(
                collection_name=self.collection_name,
                data=rows
            )
        
        print(f"Total insert time: {round(total_rt, 4)} seconds")
        
        # Flush and measure time
        print("Flushing collection...")

    def search_engine(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        search_metric: str = None
    ) -> list[dict]:
        """
        Performs a search for the nearest vectors in Milvus based on the "embedding" field.
        You can specify search_metric to override the metric (otherwise self.metric_type is used).
        Returns a list of dictionaries:
          [
            {
              "id": 12345,
              "x_label": "...",
              "y_label": "...",
              "distance": 0.85
            },
            ...
          ]
        """
        if search_metric is None:
            metric = self.metric_type
        else:
            metric = search_metric

        search_params = {
            "metric_type": metric,
            "params": {"nprobe": 10}
        }

        if top_k == 0:
            return []

        collection = Collection(name=self.collection_name)
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k + 1,
            output_fields=["x_label", "y_label"]
        )
        
        if len(results) == 0:
            return []
        
        hits = results[0][1:]  # skip the first result (it's the query itself)
        output = []
        for hit in hits:
            row = {
                "id": hit.id,
                "x_label": hit.entity.get("x_label"),
                "y_label": hit.entity.get("y_label"),
                "distance": hit.score
            }
            output.append(row)
        
        return output

    def get_collection_data(self, fields):
        """
        Retrieves specified fields for all records from the collection.
        
        Args:
            fields: List of field names to retrieve
               For example: ['embedding', 'dialog_text', 'summary']
        
        Returns:
            dict[str, list]: Dictionary where keys are field names,
                    values are lists of data for the corresponding field
                    For example: {
                    'embedding': [[0.1, 0.2, ...], ...],
                    'x': ['text1', 'text2', ...],
                    'y': ['summary1', 'summary2', ...]
                    }
        
        Raises:
            ValueError: If the specified field is not present in the collection schema
        """
        collection = Collection(name=self.collection_name)
        

        available_fields = {field.name for field in collection.schema.fields}
        invalid_fields = set(fields) - available_fields
        if invalid_fields:
            raise ValueError(f"Field {invalid_fields} is missed in schema. "
                           f"Available fields: {available_fields}")
        
        # Retrieve all records from the collection with the specified fields
        results = collection.query(
            expr="id > 0",  
            output_fields=fields
        )
        
        if not results:
            return pd.DataFrame()
        
        results = pd.DataFrame([item for item in results])   
        return results
    


# Прописать класс для локального milvus c функционалом как в классе VectorDatabase
class LocalVectorDatabase(VectorDatabase):
    """
    A class for interacting with local Milvus instance.
    Inherits from VectorDatabase but simplifies connection setup.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_dim: int,
        host: str = "localhost",
        port: str = "19530",
        metric_type: str = "IP"
    ):
        """
        Parameters:
         - collection_name: Name of the collection where records are stored
         - embedding_dim: Dimension of the embeddings
         - host: Milvus server host (default: localhost)
         - port: Milvus server port (default: 19530)
         - metric_type: Metric type for search (default: IP)
        """
        uri = f"http://{host}:{port}"
        super().__init__(
            uri=uri,
            token="",
            collection_name=collection_name,
            embedding_dim=embedding_dim,
            metric_type=metric_type
        )