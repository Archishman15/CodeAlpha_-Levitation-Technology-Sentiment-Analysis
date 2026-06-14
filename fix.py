with open('sentiment_analysis.ipynb', 'r') as f:
    content = f.read()
content = content.replace("nltk.download('punkt')", "nltk.download('punkt_tab')")
with open('sentiment_analysis.ipynb', 'w') as f:
    f.write(content)
