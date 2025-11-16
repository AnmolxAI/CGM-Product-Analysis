"""Visualization helpers for the DCGM consumer posts project.

Word clouds, bar charts and other plots can be moved here from analysis.py
and reused across notebooks or scripts.
"""

import matplotlib.pyplot as plt
from wordcloud import WordCloud

def plot_wordcloud(tokens, title: str = "Word Cloud"):
    """Quick helper to plot a word cloud from a list of tokens."""
    text = " ".join(tokens)
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()
