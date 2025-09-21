# Zoo Animal Classification Project

A comprehensive machine learning project that classifies animals into different categories based on their biological and physical characteristics using the UCI Zoo Animal Classification dataset.

## 🎯 Project Overview

This project demonstrates the complete machine learning pipeline from data exploration to model deployment, focusing on multi-class classification. The goal is to predict animal categories (mammals, birds, reptiles, fish, amphibians, insects, invertebrates) based on various features like whether they have hair, feathers, lay eggs, produce milk, etc.

## 📊 Dataset Information

- **Source**: [UCI Zoo Animal Classification Dataset](https://www.kaggle.com/datasets/uciml/zoo-animal-classification)
- **Total Animals**: 101 animals
- **Features**: 17 attributes (16 boolean features + 1 numeric feature for number of legs)
- **Classes**: 7 different animal types
- **Challenge**: Small dataset with imbalanced classes

### Dataset Features:
- `hair`: Boolean (has hair or not)
- `feathers`: Boolean (has feathers or not)
- `eggs`: Boolean (lays eggs or not)
- `milk`: Boolean (produces milk or not)
- `airborne`: Boolean (can fly or not)
- `aquatic`: Boolean (lives in water or not)
- `predator`: Boolean (is predator or not)
- `toothed`: Boolean (has teeth or not)
- `backbone`: Boolean (has backbone or not)
- `breathes`: Boolean (breathes air or not)
- `venomous`: Boolean (is venomous or not)
- `fins`: Boolean (has fins or not)
- `legs`: Integer (number of legs: 0, 2, 4, 5, 6, 8)
- `tail`: Boolean (has tail or not)
- `domestic`: Boolean (is domestic or not)
- `catsize`: Boolean (is cat-sized or not)
- `class_type`: Target variable (1-7)

### Animal Classes:
1. **Mammal** (41 animals)
2. **Bird** (20 animals)
3. **Reptile** (5 animals)
4. **Fish** (13 animals)
5. **Amphibian** (4 animals)
6. **Insect** (8 animals)
7. **Invertebrate** (10 animals)

## 🔧 Technologies Used

- **Python 3.8+**
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn
- **Development Environment**: Jupyter Notebook

## 📁 Project Structure

```
zoo-animal-classification/
│
├── data/
│   ├── zoo.data              # Main dataset
│   └── class.data            # Class definitions
│
├── notebooks/
│   └── zoo_classification.ipynb    # Main analysis notebook
│
├── src/
│   ├── data_preprocessing.py      # Data cleaning and preprocessing
│   ├── eda.py                     # Exploratory data analysis
│   ├── model_training.py          # Model training and evaluation
│   └── utils.py                   # Utility functions
│
├── results/
│   ├── figures/                   # Generated plots and visualizations
│   └── models/                    # Saved model files
│
├── requirements.txt               # Project dependencies
├── README.md                     # Project documentation
└── LICENSE                       # Project license
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8 or higher
pip (Python package installer)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/zoo-animal-classification.git
cd zoo-animal-classification
```

2. **Install required packages**
```bash
pip install -r requirements.txt
```

3. **Download the dataset**
- Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/uciml/zoo-animal-classification)
- Download `zoo.data` and `class.data` files
- Place them in the `data/` directory

### Running the Project

1. **Open the Jupyter notebook**
```bash
jupyter notebook notebooks/zoo_classification.ipynb
```

2. **Or run the Python scripts**
```bash
python src/model_training.py
```

## 📈 Project Methodology

### 1. Data Exploration and Analysis
- **Data Quality Assessment**: Checking for missing values, duplicates, and data types
- **Statistical Analysis**: Descriptive statistics and distribution analysis
- **Visualization**: Creating comprehensive plots to understand feature relationships
- **Class Distribution Analysis**: Understanding the imbalanced nature of the dataset

### 2. Feature Engineering
- **Feature Creation**: Added derived features like `total_features`, `mobility_score`, and `physical_traits`
- **Feature Selection**: Analyzed feature importance and correlations
- **Data Scaling**: Applied StandardScaler for algorithms requiring normalized data

### 3. Model Development
Implemented and compared multiple classification algorithms:

- **Random Forest Classifier**: Ensemble method with feature importance
- **Gradient Boosting Classifier**: Sequential boosting algorithm
- **Logistic Regression**: Linear classification with regularization
- **Support Vector Machine**: Kernel-based classification
- **Decision Tree**: Interpretable tree-based model
- **Naive Bayes**: Probabilistic classifier

### 4. Model Evaluation
- **Cross-Validation**: 5-fold stratified cross-validation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score
- **Confusion Matrix**: Detailed error analysis
- **Feature Importance**: Understanding model decision-making

### 5. Hyperparameter Tuning
- **Grid Search**: Comprehensive parameter optimization
- **Model Selection**: Choosing the best performing model
- **Performance Improvement**: Quantifying gains from tuning

## 📊 Key Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | CV Score |
|-------|----------|-----------|--------|----------|----------|
| Random Forest | 0.95 | 0.95 | 0.95 | 0.95 | 0.92 |
| Gradient Boosting | 0.90 | 0.91 | 0.90 | 0.90 | 0.89 |
| SVM | 0.95 | 0.96 | 0.95 | 0.95 | 0.93 |
| Logistic Regression | 0.85 | 0.86 | 0.85 | 0.85 | 0.83 |
| Decision Tree | 0.80 | 0.82 | 0.80 | 0.81 | 0.78 |
| Naive Bayes | 0.90 | 0.91 | 0.90 | 0.90 | 0.88 |

### Key Insights

1. **High Accuracy**: Achieved 95% accuracy with Random Forest and SVM
2. **Feature Importance**: `milk`, `feathers`, and `eggs` are the most discriminative features
3. **Class Separability**: Most animal classes are well-separated by the given features
4. **Small Dataset Challenge**: Despite limited data, models generalize well due to clear feature patterns

### Most Important Features:
1. **milk** - Strong indicator for mammals
2. **feathers** - Unique to birds
3. **eggs** - Distinguishes egg-laying species
4. **aquatic** - Separates water-dwelling animals
5. **legs** - Number of legs is highly discriminative

## 🔍 Technical Highlights

### Data Science Techniques Demonstrated:
- **Comprehensive EDA**: Multi-dimensional analysis with advanced visualizations
- **Feature Engineering**: Creation of meaningful derived features
- **Model Comparison**: Systematic evaluation of multiple algorithms
- **Hyperparameter Optimization**: Grid search for optimal performance
- **Cross-Validation**: Robust model validation strategy
- **Statistical Analysis**: Correlation analysis and significance testing

### Code Quality Features:
- **Modular Design**: Well-organized code structure
- **Documentation**: Comprehensive comments and docstrings
- **Error Handling**: Robust error handling and validation
- **Reproducibility**: Fixed random seeds for consistent results
- **Visualization**: Professional-quality plots and charts

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **Machine Learning Pipeline**: End-to-end ML project development
- **Data Analysis**: Advanced exploratory data analysis techniques
- **Model Selection**: Systematic approach to choosing optimal models
- **Performance Evaluation**: Comprehensive model assessment
- **Python Programming**: Clean, efficient, and well-documented code
- **Data Visualization**: Creating insightful and professional visualizations

## 🚧 Future Enhancements

1. **Deep Learning**: Implement neural networks for comparison
2. **Feature Selection**: Advanced feature selection techniques
3. **Ensemble Methods**: Custom ensemble model creation
4. **Web Application**: Deploy model as a web service
5. **Real-time Prediction**: Create an interactive prediction interface
6. **More Data**: Incorporate additional animal datasets
7. **Explainable AI**: Implement SHAP/LIME for model interpretability

## 📝 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 Contact

**Your Name** - [kk2085@bath.ac.uk](mailto:your.email@example.com)

Project Link: [https://github.com/Kaloyan2004/zoo-animal-classification](https://github.com/Kaloyan2004/zoo-animal-classification)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Kaggle community for inspiration and reference implementations
- scikit-learn developers for the excellent ML library
- Open source community for the tools and libraries used

---
