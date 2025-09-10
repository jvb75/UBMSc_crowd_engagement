#  Assessing Crowd Engagement using AI-based Audio Signal Processing
## Project Overview
The dissertation project titled "Assessing Crowd Engagement Using AI-Based Audio Signal Processing" is part of a Master's program in Artificial Intelligence and Machine Learning. This project analyses audio recordings collected during multicultural events in Bradford, which are part of the festival program celebrating Bradford 2025 City of Culture. 

The objectives of the project include extracting 101 audio features from annotated audio recordings. These recordings will be classified into categories such as music, conversation, speech, crowd noise, and environmental context (indoor or outdoor). Additionally, the project aims to determine crowd density and measure crowd engagement using classical machine learning techniques, including Random Forest, Naïve Bayes, Decision Trees, and Support Vector Machines (SVM), as well as a hybrid deep learning approach using Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. 

# Data Analysis
## Annotated Dataset Annalysis
```python
  import librosa, librosa.display
  import IPython.display as ipd
  import matplotlib.pyplot as plt
  from scipy.stats import pearsonr
  from pydub import AudioSegment
  import soundfile as sf
  from matplotlib.patches import Rectangle
  from sklearn.cluster import KMeans
  from sklearn.preprocessing import StandardScaler
  import numpy as np
  import pandas as pd
  import seaborn as sns
  import re, os
```
## Audio Signal Analysis
## Feature Importance and Selection
## Cluster  Analysis and Dimmentinality Reduction
