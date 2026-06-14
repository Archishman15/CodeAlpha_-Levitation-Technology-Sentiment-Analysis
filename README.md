# Levitation Technology Sentiment Analysis

This repository contains an end-to-end sentiment analysis pipeline to evaluate global public opinion and scientific community sentiment regarding advanced levitation technologies.

## Overview
This project uses synthetic text data simulating multiple sources (Reddit, Twitter, Amazon, arXiv, News) and analyzes sentiment using VADER, TextBlob, and a HuggingFace RoBERTa model. It also features emotion detection and aspect-based sentiment analysis.

## Setup
1. Create a virtual environment and activate it.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate the synthetic data:
   ```bash
   python generate_synthetic_data.py
   ```
4. Run the Jupyter Notebook to execute the full pipeline and generate reports:
   ```bash
   jupyter notebook sentiment_analysis.ipynb
