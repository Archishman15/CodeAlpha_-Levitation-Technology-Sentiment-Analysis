import nbformat

with open('sentiment_analysis.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code':
        if 'emotion = NRCLex(text)' in cell.source:
            cell.source = cell.source.replace('emotion = NRCLex(text)', 'emotion = NRCLex()\n    emotion.load_raw_text(text)')

with open('sentiment_analysis.ipynb', 'w') as f:
    nbformat.write(nb, f)
