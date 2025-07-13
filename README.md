# Deep Analysis of Persian News 📰🧠🇮🇷

This project performs deep natural language analysis on Persian news articles using modern neural network architectures. It leverages an LSTM-based model built with [Trax](https://github.com/google/trax) and [JAX](https://github.com/google/jax) to understand and classify the explicit content of Persian news text. Also i have provided seprated Pytorch branch for those whom prefers traditional way. Final model has been traiend on over 6 Millions parameters.

## 🔍 Project Goals

- Perform **context-aware analysis** on Persian news headlines or full articles
- Leverage **LSTM-based deep learning models** for semantic understanding
- Achieve strong performance benchmarks on large-scale Persian datasets

## 🚀 Key Features

- 🧠 **Deep Neural Network**: LSTM-based model implemented using Trax and JAX
- 📰 **Persian Language Support**: Tailored preprocessing and custom tokenization for Persian Lang
- 📊 **Large Dataset**: Trained on over **200,000 news articles**
- ✅ **High Accuracy**: Achieves **80% classification accuracy** on validation set

## Benchmark
On first try with a simple Trax network.
| Dataset    | Loss | Accuracy | Parameters |
|------------|------|----------|------------|
| Train      | 0.60223989   | 0.84375000  | 6 Millions |
| Validation | 0.65386836   | 0.80412000  | 6 Millions |

---

## 📦 Try It

1. Clone the repository:
   ```bash
   git clone https://github.com/mahdizynali/Deep-Analysis-Persian-News.git
   cd Deep-Analysis-Persian-News
   ```
   ```
   pip3 install -r requirements.txt
   ```
   For train a new model :
   ```
   python3 main.py train
   ```
   Try prediction:
   ```
   python3 main.py predict
   ```
