import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import pickle
from datetime import datetime
import os

class EmbeddingGenerator:
    def __init__(self, model_name: str, device: str = None, trust_remote_code: bool = True):
        """
        Инициализация генератора эмбеддингов.
        Args:
            model_name: Название модели из transformers.
            device: Устройство для вычислений ("cuda" или "cpu").
            trust_remote_code: Разрешить выполнение пользовательского кода из репозитория модели.
        """
        # Check device availability
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            if device == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA requested but not available. Using CPU instead.")
                device = "cpu"
            
        if device == "cpu":
            print("Warning: Using CPU for computations. This might be slow for large datasets.")
        else:
            print(f"Using device: {device}")

        self.device = device
        print(f"Loading tokenizer and model from {model_name}")
        print(f"Trust remote code: {trust_remote_code}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        
        # Retry mechanism for model loading
        retries = 3
        for attempt in range(retries):
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=trust_remote_code
                ).to(self.device)
                break
            except FileNotFoundError as e:
                if attempt < retries - 1:
                    print(f"Retrying model download ({attempt + 1}/{retries})...")
                else:
                    raise Exception(f"Failed to load model after {retries} attempts. Error: {str(e)}")

        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size
        print(f"Model embedding dimension: {self.embedding_dim}")
        
        self.max_model_length = self.tokenizer.model_max_length

    def _generate_filename(self, prefix: str) -> str:
        """
        Generate filename for embeddings.
        Format: {prefix}_{model_name}_{batch_size}_{max_length}_{timestamp}.pkl
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short_name = self.model.config.name_or_path.split('/')[-1]
        return f"{prefix}_{model_short_name}_{timestamp}.pkl"

    def _validate_params(self, max_length: int) -> int:
        """
        Validates and adjusts the max_length parameter.
        Returns adjusted max_length.
        """
        if max_length > self.max_model_length:
            print(f"\nWarning: Requested max_length ({max_length}) exceeds model's maximum "
                  f"({self.max_model_length}). Setting to {self.max_model_length}.")
            return self.max_model_length
        return max_length

    def mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on the token embeddings using attention mask.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_batch(self, texts: List[str], batch_size: int = 16, max_length: int = 512, 
                    save_path: str = "../embeddings") -> np.ndarray:
        """
        Кодирование батча текстов в эмбеддинги.
        Args:
            texts: Список текстов для кодирования.
            batch_size: Размер батча.
            max_length: Максимальная длина последовательности.
            save_path: Путь для сохранения эмбеддингов.
        Returns:
            Массив эмбеддингов.
        """
        self.model.eval()
        all_embeddings = []

        # Validate and adjust max_length
        max_length = self._validate_params(max_length)

        # Show tokenization statistics
        sample_tokens = [len(self.tokenizer.encode(text)) for text in texts[:100]]
        avg_length = sum(sample_tokens) / len(sample_tokens)
        max_tokens = max(sample_tokens)
        print(f"\nTokenization statistics (from first 100 texts):")
        print(f"Average length: {avg_length:.1f} tokens")
        print(f"Maximum length: {max_tokens} tokens")
        print(f"Texts will be truncated to {max_length} tokens if longer\n")

        # Process texts in batches
        num_samples = len(texts)
        embedding_dim = None  # Will be set after first batch

        for i in tqdm(range(0, num_samples, batch_size), desc="Encoding texts", unit="batch"):
            # Get current batch
            batch_end = min(i + batch_size, num_samples)
            batch_texts = texts[i:batch_end]
            
            # Tokenize batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt',
                return_attention_mask=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use mean pooling for sentence embeddings
                batch_embeddings = self.mean_pooling(outputs, encoded['attention_mask']).cpu().numpy()
                
                # Verify embedding dimension
                if batch_embeddings.shape[1] != self.embedding_dim:
                    raise ValueError(
                        f"Unexpected embedding dimension in batch {i//batch_size + 1}:\n"
                        f"Expected: {self.embedding_dim}, Got: {batch_embeddings.shape[1]}"
                    )
                
                all_embeddings.append(batch_embeddings)

        try:
            embeddings = np.vstack(all_embeddings)
            print(f"\nFinal embedding shape: {embeddings.shape} "
                  f"({embeddings.shape[0]} samples, {embeddings.shape[1]} dimensions)")
        except ValueError as e:
            print("\nError while combining embeddings:")
            print(f"Number of batches: {len(all_embeddings)}")
            print(f"Embedding shapes: {[emb.shape for emb in all_embeddings]}")
            raise e

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = self._generate_filename("batch")
            filepath = os.path.join(save_path, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"Embeddings saved to {filepath}")
        
        return embeddings

    def encode_dataset(self, dataset: List[Dict[str, str]], text_key: str = "dialogue", 
                      batch_size: int = 16, max_length: int = 512,
                      save_path: str = "embeddings") -> np.ndarray:
        """
        Кодирование датасета в эмбеддинги.
        Args:
            dataset: Датасет в виде списка словарей.
            text_key: Ключ для доступа к тексту в словаре.
            batch_size: Размер батча.
            max_length: Максимальная длина последовательности.
            save_path: Путь для сохранения эмбеддингов.
        Returns:
            Массив эмбеддингов.
        """
        texts = [item[text_key] for item in dataset]
        embeddings = self.encode_batch(texts, batch_size=batch_size, max_length=max_length, save_path=None)
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            filename = self._generate_filename("dataset")
            filepath = os.path.join(save_path, filename)
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"Embeddings saved to {filepath}")
        
        return embeddings
