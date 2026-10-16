# GNN-BeamRebarDetector

Trains a graph neural network that reads beam detail drawings (PDF): what each label and line is
(beam name, beam size, rebar label, bar line, dimension, section...) and which ones belong together
(label -> its bar line, beam -> its labels, section label -> same bars in the elevation).

## Files
- `train_simple.ipynb` - the Colab notebook: run top to bottom on a T4 GPU.
- `graph_builder.py` - turns a drawing page + its checked annotation (JSON) into a graph.
- `gnn.py` - the model, training, scores, save/load.
- `pdf_backend.py` - reads the PDFs (pypdfium2).

Copied from `beam-tools` on 16 Oct 2026 (the annotation and review tools stay there).

## Data (Google Drive, not in this repo)
`My Drive/Deep Learning/GNN-BeamRebarDetector/`
- `FinalJson/` - checked annotations, one JSON per page (saved from the review page)
- `PDFDwg/` - the drawings
- `gnn_training/` - graphs, trained model, charts (written by the notebook)

Colab needs the secret `GH_TOKEN` (GitHub token with access to this repo).
