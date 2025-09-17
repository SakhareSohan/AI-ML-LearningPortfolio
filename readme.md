# Sohan Sakhare's AI & Machine Learning Portfolio

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![ML/DL](https://img.shields.io/badge/ML%20%26%20DL-Scikit--learn%20%7C%20TensorFlow%20%7C%20PyTorch-orange.svg)
![NLP](https://img.shields.io/badge/NLP-Hugging%20Face%20%7C%20Transformers%20%7C%20OpeanAI%20%7C%20Gemini-yellow.svg)
![Backend](https://img.shields.io/badge/Backend-Flask%20%7C%20Node.js%20%7C%20Streamlit-green.svg)
![Database](https://img.shields.io/badge/Database-Supabase%20%7C%20Postgres-purple.svg)
![Deployment](https://img.shields.io/badge/Deployment-Docker-blueviolet.svg)

---

### Hello! 👋 I'm Sohan Sakhare.

Welcome to my personal portfolio of AI and Machine Learning projects. This repository is a curated showcase of my journey in building practical, end-to-end, data-driven applications. My passion lies in bridging the gap between complex algorithms and real-world impact, moving from foundational concepts to advanced systems in Deep Learning and Natural Language Processing.

Each project folder contains a detailed, standalone `README.md` file explaining its architecture and implementation.

---

## 🛠️ Core Competencies & Technologies

This portfolio demonstrates practical experience across the full machine learning lifecycle.

| Area                    | Technologies & Skills                                                                                           |
| :---------------------- | :-------------------------------------------------------------------------------------------------------------- |
| **Backend & Deployment** | `Python`, `Node.js`, `Flask`, `Streamlit`, `Docker`, `REST API Design`                                          |
| **ML & Data Science** | `Scikit-learn`, `Pandas`, `NumPy`, `Seaborn`, `Linear/Logistic Regression`, `SVM`, `Ensemble Methods`           |
| **Deep Learning** | `TensorFlow`, `Keras`, `PyTorch`, `Recurrent Neural Networks (LSTM)`, `Time-Series Forecasting`                 |
| **Natural Language Processing (NLP)** | `Hugging Face Transformers`, `NLTK`, `Wav2Vec2`, `T5`, `RAG`, `Vector Databases (Supabase/pgvector)`, `Prompt Engineering` |

---

## 🚀 Featured End-to-End Projects

Here is a selection of my key projects, each designed to solve a unique challenge.

### 🧠 NLP (Natural Language Processing)

#### 1. RAG Pipeline with Gemini & Supabase
> This project implements a production-ready **Retrieval-Augmented Generation (RAG)** pipeline to combat LLM knowledge limitations. It ingests unstructured PDF documents, chunks the text while preserving context, generates vector embeddings via the Google Gemini API, and stores them in Supabase. The result is a powerful semantic search engine that allows an AI to accurately "chat with your documents."
> 
> **Technical Highlight:** The core of this project is a robust, containerized (Docker) Node.js backend that orchestrates a complex data ingestion and retrieval workflow, demonstrating skills in both AI integration and modern backend engineering.
> 
> **[Go to Project ->](./NLP-Projects/RAG/)**

#### 2. Audio Transcription & Grammar Correction Pipeline
> A high-fidelity audio processing pipeline that solves the problem of messy, grammatically incorrect speech-to-text output. This command-line tool leverages two distinct Hugging Face transformer models: **Wav2Vec2** for transcription and **T5** for sequence-to-sequence grammar correction, turning raw spoken audio into clean, application-ready text.
> 
> **Technical Highlight:** This project showcases the ability to chain different state-of-the-art transformer architectures to solve a complex, multi-step NLP problem, combined with audio preprocessing using `Librosa`.
>
> **[Go to Project ->](./NLP-Projects/ASR/)**

---
### 🧠 DL (Deep Learning)

#### 1. Disaster Tweet Classification (Ensemble Model)
> This project addresses the challenge of filtering critical information from social media during a crisis. It documents a classic ML journey, starting with baseline models (Naive Bayes, Random Forest) and culminating in a more robust **majority-vote ensemble** that incorporates an **LSTM**. The final model is deployed in a Flask web app.
> 
> **Technical Highlight:** The key achievement is the measurable performance increase from the ensemble method over any single model, demonstrating a practical understanding of how to mitigate individual model weaknesses to build a more reliable classifier.
>
> **[Go to Project ->](./DL-Projects/Ensemble/)**

#### 2. Stock Price Prediction with LSTM
> A time-series forecasting project that uses a **Long Short-Term Memory (LSTM)** neural network to predict future stock prices. The application, with both Flask and Streamlit versions, allows users to input a stock ticker, fetches historical data via the `yfinance` API, and visualizes trends, moving averages, and predictions.
> 
> **Technical Highlight:** This project demonstrates the complete deep learning workflow: from data fetching and time-series preprocessing (scaling, windowing) to building, training, and deploying a Keras-based LSTM model in a user-facing application.
>
> **[Go to Project ->](./DL-Projects/Regression/)**

---
### 🧠 ML (Machine Learning)

#### 1. Multiple Disease Prediction System
> A user-friendly web application built with Streamlit for rapid prototyping and deployment. It uses classic machine learning models (**SVM, Logistic Regression**) to predict the likelihood of three different diseases (Diabetes, Heart Disease, Parkinson's), demonstrating the ability to serve multiple models in a single, clean interface.
> 
> **Technical Highlight:** This project focuses on the practical application of fundamental classification algorithms and the creation of an intuitive UI for non-technical users, bridging the gap between a model and its real-world utility.
>
> **[Go to Project ->](./ML-Projects/Classification/)**

#### 2. Student Performance Prediction
> A full-stack data science demonstration using **Linear Regression**. This Flask application features a teacher-facing dashboard for exploratory data analysis and visualization (Seaborn/Matplotlib) and a student-facing form for data collection, showcasing the complete cycle from data insight to model deployment and new data acquisition.
> 
> **Technical Highlight:** The dual-interface design (teacher vs. student) and the focus on data visualization underscore a strong understanding of building comprehensive data products, not just predictive models.
>
> **[Go to Project ->](./ML-Projects/Regression/)**

---

## 📚 Research & Continuous Learning

This section contains my personal knowledge base of curated research materials and analyses of reference applications, reflecting my commitment to staying current with the rapidly evolving field of AI.

-   **[AI Research Papers](./Research-Papers-Materials/):** Notes and summaries on key AI research topics.
-   **[ML Web App References](./Research-Papers-Materials/):** An extensive library of over 20 reference ML web applications that I have analyzed to study different architectures and techniques.

---

## 📫 Get In Touch

I am always looking for new challenges and opportunities to collaborate. Please feel free to reach out!

-   **LinkedIn:** [Sohan Sakhare](https://www.linkedin.com/in/sohan-sakhare-0940591ba/)
-   **Email:** [sohansakhare2001@gmail.com](mailto:sohansakhare2001@gmail.com)