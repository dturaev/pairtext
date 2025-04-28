# Heavy modules/functions for lazy loading
_lazy_imports = {
    "align_sentences": "pairtext.aligner",
    "detect_gaps": "pairtext.gaps",
    "sentence_tokenize": "pairtext.tokenize",
}


def __getattr__(name):
    """Lazy import for heavy modules/functions"""
    if name in _lazy_imports:
        module_path = _lazy_imports[name]
        # logger.debug(f"Lazily importing {name} from {module_path}...")
        try:
            # Import the module containing the name
            module = __import__(module_path, fromlist=[name])
            # Get the actual object (function, class, module itself)
            obj = getattr(module, name)
            # Optional: Cache the imported object on the module for faster future access
            # setattr(__import__(__name__), name, obj) # This part can be tricky depending on exactly what you're doing
            return obj
        except ImportError as e:
            raise ImportError(f"Cannot import name '{name}' from '{__name__}'") from e
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = list(_lazy_imports.keys())
