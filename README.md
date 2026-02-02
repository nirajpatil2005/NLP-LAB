# 🤖 Natural Language Processing Laboratory

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-green.svg)](https://www.nltk.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A comprehensive collection of Natural Language Processing assignments covering fundamental to advanced NLP concepts using Python and NLTK.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Assignments](#assignments)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Concepts Covered](#key-concepts-covered)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This repository contains a complete set of NLP laboratory assignments designed to provide hands-on experience with various natural language processing techniques and algorithms. Each assignment is implemented as a Jupyter notebook with detailed explanations, code, and visualizations.

---

## 🛠️ Technologies Used

- **Python 3.8+**
- **NLTK** - Natural Language Toolkit
- **Scikit-learn** - Machine Learning library
- **Gensim** - Topic modeling and Word2Vec
- **SpaCy** - Industrial-strength NLP
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization

---

## 📋 Assignments

### Assignment 1: Tokenization and Stemming
**📁 File:** `01_tokenization_stemming.ipynb`

Comprehensive implementation of various tokenization techniques and stemming algorithms.

**Topics Covered:**
- ✅ Whitespace Tokenization
- ✅ Punctuation-based Tokenization
- ✅ Treebank Tokenization
- ✅ Tweet Tokenization
- ✅ Multi-Word Expression (MWE) Tokenization
- ✅ Porter Stemmer
- ✅ Snowball Stemmer
- ✅ Lemmatization using WordNet

**Key Functions:**
```python
- whitespace_tokenize()
- treebank_tokenize()
- tweet_tokenize()
- porter_stem()
- snowball_stem()
- wordnet_lemmatize()
```

---

### Assignment 2: Feature Extraction and Embeddings
**📁 File:** `02_bow_tfidf_word2vec.ipynb`

Implementation of traditional and modern text representation techniques.

**Topics Covered:**
- ✅ Bag-of-Words (Count Occurrence)
- ✅ Normalized Count Occurrence
- ✅ TF-IDF (Term Frequency-Inverse Document Frequency)
- ✅ Word2Vec Embeddings (CBOW & Skip-gram)

**Key Features:**
- Sparse matrix representations
- Dense vector embeddings
- Similarity calculations
- Visualization of word embeddings

---

### Assignment 3: Text Preprocessing Pipeline
**📁 File:** `03_text_preprocessing_pipeline.ipynb`

Complete text preprocessing workflow with feature extraction.

**Topics Covered:**
- ✅ Text Cleaning (lowercase, special characters, numbers)
- ✅ Lemmatization
- ✅ Stop Words Removal
- ✅ Label Encoding
- ✅ TF-IDF Representation
- ✅ Save Processed Outputs

**Pipeline:**
```
Raw Text → Cleaning → Lemmatization → Stop Words Removal → TF-IDF → Save
```

---

### Assignment 4: Named Entity Recognition (NER)
**📁 File:** `04_named_entity_recognition.ipynb`

Build and evaluate a Named Entity Recognition system for real-world text.

**Topics Covered:**
- ✅ Entity Extraction (Person, Organization, Location, Date, etc.)
- ✅ SpaCy NER Model
- ✅ Custom NER Training
- ✅ Evaluation Metrics

**Performance Metrics:**
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

**Data Sources:**
- News articles
- Social media posts
- Custom text datasets

---

### Assignment 5: WordNet Semantic Relationships
**📁 File:** `05_wordnet_semantic_analysis.ipynb`

Explore semantic relationships using WordNet lexical database.

**Topics Covered:**
- ✅ Synonymy (Similar meanings)
- ✅ Antonymy (Opposite meanings)
- ✅ Hypernymy (General-to-specific)
- ✅ Hyponymy (Specific-to-general)
- ✅ Meronymy (Part-whole relationships)
- ✅ Semantic Similarity

**Features:**
- Interactive word explorer
- Hypernym tree visualization
- Similarity scoring between words
- Text-level semantic analysis

---

### Assignment 6: Machine Translation System
**📁 File:** `06_machine_translation.ipynb`

Develop a Machine Translation system for English ↔ Indian Language translation.

**Topics Covered:**
- ✅ Sequence-to-Sequence Models
- ✅ Attention Mechanism
- ✅ Translation between English and Indian Languages (Hindi/Telugu/Tamil)
- ✅ BLEU Score Evaluation
- ✅ Public Information Content Translation

**Models Used:**
- Neural Machine Translation (NMT)
- Transformer-based models
- Pre-trained translation APIs

---

### Assignment 7: NLTK Text Processing Application
**📁 File:** `07_nltk_text_processing.ipynb`

Comprehensive text preprocessing and analysis application using NLTK.

**Topics Covered:**
- ✅ Advanced Tokenization
- ✅ Part-of-Speech (POS) Tagging
- ✅ Named Entity Recognition
- ✅ Chunking and Parsing
- ✅ Dependency Parsing
- ✅ Sentiment Analysis

**Features:**
- Interactive GUI/Command-line interface
- Batch processing capabilities
- Export results to various formats

---

### Assignment 8: Word Sense Disambiguation
**📁 File:** `08_word_sense_disambiguation.ipynb`

Apply WordNet-based algorithms to disambiguate word meanings in context.

**Topics Covered:**
- ✅ Lesk Algorithm
- ✅ Context-based Disambiguation
- ✅ Synset Selection
- ✅ Ambiguous Sentence Analysis
- ✅ Accuracy Evaluation

**Example:**
```
Sentence: "The bank is near the river"
Word: "bank"
→ Disambiguated meaning: "riverbank" (not financial institution)
```

---

### Assignment 9: Indian Language Sentiment Analysis
**📁 File:** `09_indian_language_sentiment.ipynb`

Sentiment analysis system for Indian languages (Hindi/Telugu/Tamil/Bengali).

**Topics Covered:**
- ✅ Indian Language Text Processing
- ✅ Sentiment Classification (Positive/Negative/Neutral)
- ✅ Feature Extraction for Indian Languages
- ✅ Machine Learning Models (SVM, Naive Bayes, LSTM)
- ✅ Model Evaluation

**Challenges Addressed:**
- Unicode handling
- Code-mixing (Hinglish)
- Limited labeled datasets
- Regional language variations

---

### Assignment 10: N-gram Auto-Complete System
**📁 File:** `10_ngram_autocomplete.ipynb`

Build an intelligent auto-complete system using N-gram language models.

**Topics Covered:**
- ✅ Unigram, Bigram, Trigram Models
- ✅ N-gram Probability Calculation
- ✅ Language Model Training
- ✅ Next Word Prediction
- ✅ Perplexity Evaluation

**Applications:**
- Text prediction
- Speech recognition
- Machine translation
- Authorship attribution

**Features:**
- Multiple completion suggestions
- Probability scoring
- Context-aware predictions
- Real-time inference

---

## 🎓 Key Concepts Covered

### Text Processing
- Tokenization techniques
- Stemming algorithms
- Lemmatization methods
- Stop words removal
- Text normalization

### Feature Extraction
- Bag-of-Words (BoW)
- TF-IDF
- Word embeddings (Word2Vec, GloVe)
- N-grams
- Label encoding

### NLP Tasks
- Named Entity Recognition (NER)
- Part-of-Speech (POS) Tagging
- Sentiment Analysis
- Machine Translation
- Word Sense Disambiguation

### Semantic Analysis
- WordNet relationships
- Semantic similarity
- Hypernym/Hyponym hierarchies
- Synonym/Antonym detection

### Language Modeling
- N-gram models
- Probability estimation
- Perplexity calculation
- Auto-completion systems

### Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- BLEU Score (Machine Translation)
- Perplexity (Language Models)
- Confusion Matrix

---

## 📊 Results and Outputs

Each assignment generates various outputs:

- **Visualizations**: Word clouds, distribution plots, embedding visualizations
- **Models**: Trained models saved in `.pkl` or `.h5` format
- **Reports**: Performance metrics and evaluation results
- **Processed Data**: Cleaned and transformed datasets

---

## 🎯 Learning Outcomes

After completing these assignments, you will be able to:

1. ✅ Preprocess text data effectively
2. ✅ Extract meaningful features from text
3. ✅ Build NLP applications from scratch
4. ✅ Evaluate NLP model performance
5. ✅ Understand semantic relationships in text
6. ✅ Implement machine translation systems
7. ✅ Develop sentiment analysis models
8. ✅ Create intelligent text prediction systems
9. ✅ Work with multilingual text data
10. ✅ Apply NLP to real-world problems

---