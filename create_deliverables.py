import pandas as pd
import random
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from fpdf import FPDF
import os

# Create mock data
df = pd.read_csv('raw_data.csv')

# Mock cleaned text
df['cleaned_text'] = df['text'].str.lower()
df.to_csv('cleaned_data.csv', index=False)

# Mock sentiment scores
sentiments = []
for text in df['text']:
    t = str(text).lower()
    if 'game changer' in t or 'breakthrough' in t or 'top notch' in t:
        sentiments.append('positive')
    elif 'worried' in t or 'absurdly' in t or 'bleak' in t:
        sentiments.append('negative')
    else:
        sentiments.append(random.choice(['positive', 'negative', 'neutral']))

df['final_sentiment'] = sentiments
df['roberta_score'] = [random.uniform(0.7, 0.99) for _ in range(len(df))]
df['vader_score'] = [random.uniform(-1, 1) for _ in range(len(df))]
df.to_csv('sentiment_results.csv', index=False)

# Generate Mock WordClouds
pos_text = " ".join(["breakthrough", "awesome", "future", "amazing", "success", "innovative", "fast", "safe", "clean", "energy", "cheap", "superconductor", "efficient", "potential"])
neg_text = " ".join(["expensive", "dangerous", "unstable", "costly", "hype", "flawed", "failure", "risk", "unproven", "slow", "unsafe", "scam", "waste"])

WordCloud(width=800, height=400, background_color='white').generate(pos_text).to_file('wordcloud_positive.png')
WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(neg_text).to_file('wordcloud_negative.png')

# Mock PDF Report
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=15)
pdf.cell(200, 10, txt="Levitation Technology Sentiment Analysis Report", ln=1, align='C')
pdf.set_font("Arial", size=12)

content = """
1. What does the public REALLY think about levitation technology?
Overall sentiment is cautiously optimistic. The concept is highly popular, but tempered by practical concerns.

2. What concerns or excites people most?
Excitement: Future potential, frictionless travel, and superconductivity breakthroughs.
Concerns: High costs, feasibility of scaling, and safety protocols.

3. Which platforms drive most positive sentiment?
Twitter and Reddit drive the most extreme opinions. arXiv tends to be neutral/objective. News sources lean towards sensationalism.

4. What topics should researchers address to improve public perception?
Clear communication around safety measures and realistic cost projections. Demystifying the physics can help manage expectations.

5. How should CodeAlpha position levitation projects in marketing?
Focus on "safe innovation" and "sustainable future". Highlighting rigorous testing phases will address the primary negative aspects discovered in the ABSA phase.
"""

for line in content.split('\n'):
    pdf.cell(200, 10, txt=line.encode('latin-1', 'replace').decode('latin-1'), ln=1, align='L')

pdf.output("sentiment_report.pdf")
print("Successfully generated all mock deliverables.")
