"""Model caching and management for MLX distributed inference."""

import json
import shutil
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from .errors import ModelDownloadError, CacheError
from .logging import get_logger


class ModelCacheManager:
    """Manages model downloads and caching."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the model cache manager.
        
        Args:
            cache_dir: Directory to store cached models. 
                      Defaults to ~/.cache/mlx_models/
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "mlx_models"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()
        
        # Metadata file to track cached models
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache metadata: {e}")
    
    def _get_model_cache_path(self, model_name: str) -> Path:
        """Get the cache path for a model."""
        # Replace slashes with underscores for filesystem compatibility
        safe_name = model_name.replace("/", "_")
        return self.cache_dir / safe_name
    
    def is_cached(self, model_name: str) -> bool:
        """
        Check if a model is cached.
        
        Args:
            model_name: HuggingFace model identifier
            
        Returns:
            True if model is cached, False otherwise
        """
        cache_path = self._get_model_cache_path(model_name)
        
        # Check if directory exists and has files
        if cache_path.exists() and any(cache_path.iterdir()):
            # Verify it has essential files including model weights
            has_config = (cache_path / "config.json").exists()
            has_tokenizer = any(
                (cache_path / f).exists() 
                for f in ["tokenizer.json", "tokenizer_config.json", "tokenizer.model"]
            )
            # Check for model weights (either safetensors or bin files)
            has_weights = any(
                cache_path.glob("*.safetensors")
            ) or any(
                cache_path.glob("*.bin")
            ) or (cache_path / "model.safetensors.index.json").exists()
            
            return has_config and has_tokenizer and has_weights
        
        return False
    
    def get_model_path(
        self,
        model_name: str,
        allow_patterns: Optional[List[str]] = None,
        force_download: bool = False,
        max_retries: int = 3
    ) -> Path:
        """
        Get the path to a model, downloading if necessary.
        
        Args:
            model_name: HuggingFace model identifier or local path
            allow_patterns: Patterns for files to download
            force_download: Force re-download even if cached
            max_retries: Maximum number of download retries
            
        Returns:
            Path to the model directory
            
        Raises:
            ModelDownloadError: If download fails
        """
        # Check if it's a local path
        local_path = Path(model_name)
        if local_path.exists() and local_path.is_dir():
            self.logger.info(f"Using local model path: {local_path}")
            return local_path
        
        # Get cache path
        cache_path = self._get_model_cache_path(model_name)
        
        # Check if we need to download
        if not force_download and self.is_cached(model_name):
            self.logger.info(f"Using cached model: {model_name}")
            return cache_path
        
        # Download model with retries
        self.logger.info(f"Downloading model: {model_name}")
        
        for attempt in range(max_retries):
            try:
                # Remove existing cache if force download
                if force_download and cache_path.exists():
                    self.logger.info(f"Removing existing cache for {model_name}")
                    shutil.rmtree(cache_path)
                
                # Download to cache
                start_time = time.time()
                downloaded_path = snapshot_download(
                    model_name,
                    cache_dir=str(self.cache_dir),
                    allow_patterns=allow_patterns,
                    local_dir=str(cache_path),
                    local_dir_use_symlinks=False
                )
                
                download_time = time.time() - start_time
                self.logger.info(
                    f"Downloaded {model_name} in {download_time:.2f}s"
                )
                
                # Update metadata
                self.metadata[model_name] = {
                    "path": str(cache_path),
                    "download_time": download_time,
                    "timestamp": time.time()
                }
                self._save_metadata()
                
                return Path(downloaded_path)
                
            except HfHubHTTPError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(
                        f"Download failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {wait_time}s: {e}"
                    )
                    time.sleep(wait_time)
                else:
                    raise ModelDownloadError(
                        f"Failed to download {model_name} after {max_retries} attempts: {e}"
                    )
            except Exception as e:
                raise ModelDownloadError(f"Failed to download {model_name}: {e}")
    
    def download_model_files(
        self,
        model_name: str,
        file_patterns: List[str],
        force_download: bool = False
    ) -> Path:
        """
        Download specific model files.
        
        Args:
            model_name: HuggingFace model identifier
            file_patterns: List of file patterns to download
            force_download: Force re-download even if cached
            
        Returns:
            Path to the model directory
        """
        return self.get_model_path(
            model_name,
            allow_patterns=file_patterns,
            force_download=force_download
        )
    
    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear cached models.
        
        Args:
            model_name: Specific model to clear, or None to clear all
        """
        try:
            if model_name:
                # Clear specific model
                cache_path = self._get_model_cache_path(model_name)
                if cache_path.exists():
                    shutil.rmtree(cache_path)
                    self.logger.info(f"Cleared cache for {model_name}")
                    
                    # Update metadata
                    if model_name in self.metadata:
                        del self.metadata[model_name]
                        self._save_metadata()
                else:
                    self.logger.warning(f"No cache found for {model_name}")
            else:
                # Clear all cache
                for path in self.cache_dir.iterdir():
                    if path.is_dir() and path != self.metadata_file.parent:
                        shutil.rmtree(path)
                
                # Clear metadata
                self.metadata = {}
                self._save_metadata()
                
                self.logger.info("Cleared all model cache")
                
        except Exception as e:
            raise CacheError(f"Failed to clear cache: {e}")
    
    def get_cache_size(self) -> Dict[str, float]:
        """
        Get the size of cached models.
        
        Returns:
            Dictionary mapping model names to sizes in GB
        """
        sizes = {}
        
        for model_name in self.metadata:
            cache_path = self._get_model_cache_path(model_name)
            if cache_path.exists():
                total_size = sum(
                    f.stat().st_size for f in cache_path.rglob("*") if f.is_file()
                )
                sizes[model_name] = total_size / (1024 ** 3)  # Convert to GB
        
        return sizes
    
    def list_cached_models(self) -> List[str]:
        """
        List all cached models.
        
        Returns:
            List of cached model names
        """
        return [
            model_name for model_name in self.metadata
            if self.is_cached(model_name)
        ]
