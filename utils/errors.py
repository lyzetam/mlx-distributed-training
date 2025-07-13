"""Custom exceptions for MLX distributed inference."""


class MLXDistributedError(Exception):
    """Base exception for MLX distributed inference errors."""
    pass


class ModelDownloadError(MLXDistributedError):
    """Raised when model download fails."""
    pass


class ModelLoadError(MLXDistributedError):
    """Raised when model loading fails."""
    pass


class DistributedSetupError(MLXDistributedError):
    """Raised when distributed setup fails."""
    pass


class InferenceError(MLXDistributedError):
    """Raised when inference fails."""
    pass


class CacheError(MLXDistributedError):
    """Raised when cache operations fail."""
    pass
