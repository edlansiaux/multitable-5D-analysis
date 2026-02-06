# Note : Gestion de la typo potentielle dans le nom de fichier structure.py
try:
    from .structure import RelationalExplainer
except ImportError:
    # Fallback si le fichier s'appelle strcuture.py (typo observée)
    try:
        from .structure import RelationalExplainer
    except ImportError:
        pass
