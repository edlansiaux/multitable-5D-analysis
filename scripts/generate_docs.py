import inspect
import sys
from pathlib import Path

# Ajout du path
sys.path.append(str(Path(__file__).parent.parent))

from mt5d.core.pipeline.mt5d_pipeline import MT5DPipeline
from mt5d.models.architectures.rht import RelationalHypergraphTransformer

def doc_to_md(cls):
    """Convertit la docstring d'une classe en Markdown."""
    doc = inspect.getdoc(cls) or "Pas de documentation."
    name = cls.__name__
    return f"## Class `{name}`\n\n{doc}\n\n"

def main():
    print("Génération de la documentation API...")
    
    classes_to_document = [
        MT5DPipeline,
        RelationalHypergraphTransformer
    ]
    
    output_path = Path("docs/api_reference.md")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("# API Reference\n\n")
        f.write("Documentation générée automatiquement.\n\n")
        
        for cls in classes_to_document:
            f.write(doc_to_md(cls))
            
    print(f"Documentation écrite dans {output_path}")

if __name__ == "__main__":
    main()
