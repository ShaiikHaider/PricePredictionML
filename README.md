# PricePredictionML
# Smart Product Pricing Challenge – End-to-End ML Project

The dataset included:
- 🖼️ **Product Images**
- 📝 **Catalog Descriptions**
- 💰 **Price Labels**

# Project Overview
we built a complete **ML pipeline** that combines **Computer Vision + NLP + Tabular** features using:
- 🔹 **ResNet50** (for image embeddings via transfer learning)
- 🔹 **TF-IDF** (for text features from product descriptions)
- 🔹 **Custom Feature Engineering** (weights, pack counts, etc.)
- 🔹 **XGBoost Regressor** (for price prediction)
- 🔹 **Outlier Removal, Scaling, PCA, and Feature Caching** for efficient runs

# Results
The model achieved **strong validation performance** and successfully generated predictions for **75K+ test samples**

# Key Learnings

✅ Merging **multimodal data** (text + images + numeric)  
✅ Designing **efficient feature pipelines**  
✅ Importance of **caching, GPU acceleration, and interpretability**

## 🛠️ Tools & Technologies
- 🐍 Python  
- 🧠 TensorFlow  
- ⚙️ Scikit-learn  
- 🌳 XGBoost  
- 📦 Pandas, NumPy  
- 💻 Google Colab + Drive Integration  

# Environment

- Google Colab (GPU Runtime)
- Google Drive for Data Storage & Caching


