# HR-GNN

This repository contains the PyTorch implementation of the paper: 

**[A Graph Attention Network-based Heart Disease Prediction Model for Reflecting High-order Representations Between Patients](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002869131)**. 
Kim Young Jin, LEE HYUN JAE Ha Young Kim.(2022).

## Introduction
Heart disease is one of the leading causes of death in modern society. Over 50% of adults have at least one of three risk factors, high blood pressure, high cholesterol and smoking, for heart disease. Also, there are other major causes including diabetes, obesity, lack of exercise and heavy drinking. Diagnosing heart disease is a complex medical problem that requires an analysis of a combination of clinical and pathological data. Meeting a need for a high level of clinical expertise for the analysis gives rise to excessive cost. Reducing the cost has become an important issue. To resolve the issue, we propose Heart Relation Graph Neural Network(HRGNN) model with a novel-designed architecture. Specifically, we design Graph Attention Network(GAT)-based heart disease prediction model with three-type clinical information and propose dual-level attention mechanism to learn high-order representation of each type and attribute. HRGNN is can learn the hidden patterns in the high-order graph structure, which is a framework considering the complex type-specific patient data correlations. Health survey data of 401,958 adults provided by the Centers for Disease Control and Prevention are used in this study. Our experiments show that HRGNN outperforms existing Graph Neural Networks(GNN) and traditional machine learning models at node classification tasks. Also, it is experimentally demonstrated that three-type branch, dual-level attention modules and data imbalance resolving methods applied to classification models improve the performance.


## Model Training


### Prerequisites

python == 3.8
pandas == 1.4.2
numpy == 1.23.0

### torch module
pytorch == 1.11.0
pytorch-cluster == 1.6.0 
pytorch-lightning == 1.6.5
pytorch-mutex == 1.0
pytorch-scatter == 2.0.9
pytorch-sparse == 0.6.13
pytorch-spline-conv == 1.2.1

### graph module
igraph == 0.9.11
graphviz == 2.50.0

Nvidia GPU with cuda 11.2 are required for training models.

### Data

Dataset : https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

The preprocessing and make graph data Run the command:
```
python3 4096_1_train_smt.py
python3 4096_1_val_smt.py
python3 4096_1_test_smt.py
```

### Train

GNN for predicting heart disease running command:

```
python3 train.py --data_path {storage_path} --embedding_size 512 --result_path {model_path}
```

## Architecture

<img src="https://github.com/NYUMedML/GNN_for_EHR/blob/master/plots/model.png" alt="drawing" width="900"/>

## Acknowledgement
https://github.com/Diego999/pyGAT

Thanks to pytorch implementation of GAT!!
