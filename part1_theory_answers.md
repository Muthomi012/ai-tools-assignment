# Part 1: Theoretical Understanding

---

## Q1: Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?

- **TensorFlow** is developed by Google and is known for production-ready deployment, TensorBoard support, and model optimization tools (like TensorFlow Lite and TF Serving).
- **PyTorch** is developed by Meta (Facebook) and is known for dynamic computation graphs (eager execution), flexibility, and ease of debugging.

📌 **Use TensorFlow** for mobile apps, embedded systems, or cloud deployments.  
📌 **Use PyTorch** for research, experimentation, and when flexibility is needed.

---

## Q2: Describe two use cases for Jupyter Notebooks in AI development.

1. **Exploratory Data Analysis (EDA):** Visualize datasets using charts, graphs, and tables inline.
2. **Model Prototyping:** Train and tweak models interactively before integrating into production.

---

## Q3: How does spaCy enhance NLP tasks compared to basic Python string operations?

- spaCy provides **pre-trained language models**, **tokenization**, **part-of-speech tagging**, **named entity recognition (NER)**, and **dependency parsing**.
- Basic string operations like `.split()`, `.lower()`, or `.replace()` don’t understand context or grammar.
- spaCy processes full language pipelines, enabling much more intelligent text analysis.

---

## Comparative Analysis: Scikit-learn vs TensorFlow

| Feature             | Scikit-learn                  | TensorFlow                        |
|---------------------|-------------------------------|-----------------------------------|
| Main Focus          | Classical ML (SVM, Trees)     | Deep Learning (Neural Networks)   |
| API Style           | Simple, intuitive              | More complex, functional/graph-based |
| Learning Curve      | Easy for beginners             | Moderate to steep                 |
| Deployment Tools    | Minimal                        | Strong tools: TFLite, TF Serving  |
| Community Support   | Excellent for ML               | Huge and active for DL            |

📌 Use **Scikit-learn** for fast implementation of ML algorithms  
📌 Use **TensorFlow** when working with neural networks or deep learning applications

---
