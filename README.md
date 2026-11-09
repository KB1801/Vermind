# VeriMind v2.2.0 (Flagship Enterprise Edition)
> **Hybrid RAG + Agentic AI Platform** with Dual-Engine Offline Resilience, Atomic Claim Verification, 4 Native ML Models, and 3D WebGL Galaxy Visualizer. Bounded for 8GB RAM Laptop Deployment.

---

## 🌟 Key Features
* **19 Native RAG Patterns:** Naive, Retrieve+Rerank, CRAG, Self-RAG, Adaptive, HyDE, RAG-Fusion, RAPTOR, Graph-RAG, CAG, and Auto-Router.
* **4 Native ML Models:**
  1. Scikit-Learn TF-IDF + Logistic Regression Intent Classifier (`data/models/intent_classifier.joblib`).
  2. PyTorch I-JEPA Latent World Model (`data/models/ijepa_latent_world_model.pt`).
  3. Fine-tuned BAAI/bge-small-en-v1.5 SentenceTransformer (`data/models/verimind-bge-finetuned/`).
  4. Offline Extractive TF-IDF Sentence Scoring Engine.
* **Evidence Integrity Index (EII):** Atomic claim decomposition and contradiction detection.
* **Bidirectional MCP Server:** Connects VeriMind to Claude Desktop and Cursor.
* **Dual-Engine Resilience:** Automatic fallback cascade: Groq -> OpenAI -> Ollama -> Extractive Engine.

---

## 🚀 Quickstart
```powershell
# 1. Activate Environment & Run Master Server
.\.venv\Scripts\Activate.ps1
python run.py

# 2. Run Streamlit UI
cd frontend
streamlit run streamlit_app.py


---

### File 152: `backend/app/multimodal/vision_processor.py`
**Path:** `verimind/backend/app/multimodal/vision_processor.py`

```python
"""
Multimodal Vision & Image OCR Processing Engine.
Extracts text descriptions from document image attachments.
"""
from typing import Dict, Any

class VisionProcessor:
    def process_image_bytes(self, image_bytes: bytes, filename: str = "image.png") -> Dict[str, Any]:
        """Extracts text descriptions from image payloads using local OCR."""
        try:
            import pytesseract
            from PIL import Image
            import io

            image = Image.open(io.BytesIO(image_bytes))
            ocr_text = pytesseract.image_to_string(image).strip()
            return {
                "filename": filename,
                "extracted_text": ocr_text or "No text detected in image.",
                "status": "success"
            }
        except Exception as e:
            return {
                "filename": filename,
                "extracted_text": f"OCR processing unavailable: {str(e)}",
                "status": "fallback"
            }

vision_processor = VisionProcessor()