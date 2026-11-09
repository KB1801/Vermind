"""
Fine-Tunes BAAI/bge-small-en-v1.5 Embedding Model on Domain Triples.
Uses SentenceTransformers MultipleNegativesRankingLoss for contrastive learning.
Saves fine-tuned weights to data/models/verimind-bge-finetuned/.
Runs on CPU on 8GB RAM laptop.
"""
import os
from pathlib import Path
from backend.app.core.logger import logger

def train_fine_tuned_embedder():
    output_dir = Path("data/models/verimind-bge-finetuned")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader

        logger.info("Loading base embedding model BAAI/bge-small-en-v1.5 for fine-tuning...")
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")

        # Fine-Tuning Corpus (Query -> Positive Anchor Passage)
        train_examples = [
            InputExample(texts=["What are VeriMind system requirements?", "VeriMind is optimized to run on an 8GB RAM laptop with Groq and Qdrant."]),
            InputExample(texts=["How does I-JEPA evaluate alignment?", "PyTorch I-JEPA computes energy distances in 384-dimensional latent Z-space."]),
            InputExample(texts=["What is the Evidence Integrity Index?", "EII measures claim accuracy by evaluating supported vs contradicted assertions."])
        ]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        logger.info("Training SentenceTransformer fine-tuning epoch...")
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=1,
            output_path=str(output_dir)
        )
        logger.info(f"✓ Fine-tuned embedding model saved to {output_dir}")
        return True

    except Exception as e:
        logger.warning(f"! Fine-tuning skipped ({str(e)}). Placeholder marker written.")
        (output_dir / "README.txt").write_text(f"Fine-tuning marker: {str(e)}")
        return False

if __name__ == "__main__":
    train_fine_tuned_embedder()