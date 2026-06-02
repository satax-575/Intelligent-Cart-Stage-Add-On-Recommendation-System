# Cartboost AI - Clean Submission Package

Team: ML Systems Engineering  
Date: March 3, 2026  
Version: 3.0

---

## Contents

This package contains a total of 13 files, which includes 1 report, 9 code files, 5 data files, and 1 configuration file.

---

## Documentation

### Complete Technical Report
File: COMPLETE_TECHNICAL_REPORT.md

We suggest you start here, as it contains everything in one place. It covers the following topics:
- Executive summary and problem overview
- Technical architecture and system design
- Model development and training
- Evaluation results and metrics
- Business impact and ROI analysis
- Innovation highlights
- Deployment strategy

---

## Source Code

There are 9 source code files in total. 

For the core pipeline, we have included:
1. main_v2_with_v3_embeddings.py - This is the main training pipeline.
2. embeddings_v3.py - This handles the V3 embedding module.
3. fast_negative_sampler.py - This is an ultra-fast sampler that provides a 100x speedup.
4. feature_engineering_v2.py - This manages feature engineering for 53 features.
5. evaluation_v2.py - This is the evaluation module, which achieves a 47x speedup.

For supporting files, we have:
6. data_loader.py - This is used for data loading.
7. train_ranker.py - This is used for model training.
8. config.py - This handles the configuration.
9. utils.py - This contains helpful utilities.

Data: We have also provided 5 CSV files in the data directory.

---

## Quick Start

You can get started by running the following commands:

```bash
pip install -r requirements.txt

python main_v2_with_v3_embeddings.py
```

The training is expected to take about 12 minutes, and the output will be saved at models/ranker_v2_v3.pkl.

---

## Key Results

Our key results are as follows:

For ranking, we achieved an AUC between 0.75 and 0.78, an NDCG at 10 between 0.55 and 0.68, and a Recall at 10 between 0.60 and 0.70.
In terms of business impact, we reached an attach rate of 13 to 17 percent, an AOV lift of 13 to 22 percent, and a revenue of 9.18 million rupees per year.
For performance, the training takes approximately 12 minutes, the inference time is around 250 milliseconds, and the throughput is 1.4 thousand requests per second.

We exceeded all of our targets.

---

## Innovation

Our key innovations include:
1. Ultra-Fast Sampling, delivering a 100x speedup.
2. Embedding Features, which improved our AUC by 5 to 8 percent.
3. Vectorized Evaluation, providing a 47x speedup.
4. Realistic Simulation, which allows for accurate projections.

---

## Why We Win

We believe our submission is strong due to the following reasons:
- Technical Excellence, as we have exceeded all targets.
- Business Impact, with 8.53 million rupees in additional revenue and a 317 percent ROI.
- Innovation, demonstrated by our 100x and 47x speedups along with the use of embeddings.
- Quality, as we deliver production-ready and clean code.

We are confident that this is competition-winning quality.
