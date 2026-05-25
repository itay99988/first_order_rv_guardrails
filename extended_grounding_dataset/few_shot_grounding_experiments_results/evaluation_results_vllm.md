# Evaluation Results - test+ood.set.1295 (vLLM Local Inference)

**Dataset:** `test+ood.set.1295/dataset.validated.jsonl` | 1,295 records | 19 domains (5 OOD)
**Inference:** vLLM local server | Concurrency: 16
**Speed test:** 100 samples (PHASE 1) | Accuracy eval: all 1,295 samples (PHASE 2)

---

## 1. Models Evaluated

| Model | Family | Params | Gen. Errors (accuracy phase) |
|---|---|---|---|
| `Qwen/Qwen3.5-2B` | Qwen | 2B | warning 1 |
| `Qwen/Qwen3.5-4B` | Qwen | 4B | ok 0 |
| `Qwen/Qwen3.5-9B` | Qwen | 9B | ok 0 |
| `mistralai/Ministral-3-8B-Instruct-2512` | mistralai | 8B | ok 0 |
| `mistralai/Ministral-3-3B-Instruct-2512` | mistralai | 3B | warning 1 |
| `meta-llama/Llama-3.1-8B-Instruct` | meta-llama | 8B | warning 1 |
| `meta-llama/Llama-3.2-3B-Instruct` | meta-llama | 3B | warning 35 |
| `google/gemma-3-12b-it` | google | 12B | warning 3 |
| `google/gemma-3-4b-it` | google | 4B | ok 0 |

## 2. Accuracy Metrics

### 2.1 Overall Accuracy (ranked by sample accuracy)

| # | Model | Params | sample_acc | found_acc | mention_acc | canonical_acc | full_inst_acc |
|---|---|---|---|---|---|---|---|
| 1 | Qwen3.5-9B | 9B | **0.9629** | 0.9799 | 0.9514 | 0.9558 | 0.9497 |
| 2 | Qwen3.5-4B | 4B | **0.9537** | 0.9761 | 0.9438 | 0.9362 | 0.9361 |
| 3 | Ministral-8B | 8B | **0.9514** | 0.9776 | 0.9532 | 0.9689 | 0.9514 |
| 4 | Ministral-3B | 3B | **0.9097** | 0.9653 | 0.8825 | 0.8642 | 0.8782 |
| 5 | LLaMA-3.1-8B | 8B | **0.8919** | 0.9429 | 0.9421 | 0.8822 | 0.9182 |
| 6 | Gemma-3-12B | 12B | **0.8888** | 0.9189 | 0.9591 | 0.9574 | 0.9557 |
| 7 | Qwen3.5-2B | 2B | **0.8680** | 0.9629 | 0.8748 | 0.8020 | 0.8313 |
| 8 | Gemma-3-4B | 4B | **0.7668** | 0.8517 | 0.9020 | 0.8478 | 0.8807 |
| 9 | LLaMA-3.2-3B | 3B | **0.6911** | 0.8170 | 0.7794 | 0.7136 | 0.7368 |

### 2.2 Found (Binary Classification)

| Model | Params | P | R | F1 |
|---|---|---|---|---|
| Qwen3.5-9B | 9B | 0.9914 | 0.9719 | **0.9816** |
| Qwen3.5-4B | 4B | 0.9885 | 0.9677 | **0.9780** |
| Ministral-8B | 8B | 0.9763 | 0.9831 | **0.9797** |
| Ministral-3B | 3B | 0.9955 | 0.9410 | **0.9675** |
| LLaMA-3.1-8B | 8B | 0.9253 | 0.9747 | **0.9494** |
| Gemma-3-12B | 12B | 0.8752 | 0.9944 | **0.9310** |
| Qwen3.5-2B | 2B | 0.9611 | 0.9719 | **0.9665** |
| Gemma-3-4B | 4B | 0.7876 | 1.0000 | **0.8812** |
| LLaMA-3.2-3B | 3B | 0.7771 | 0.9354 | **0.8489** |

### 2.3 Mention-Instance

| Model | Params | P | R | F1 |
|---|---|---|---|---|
| Qwen3.5-9B | 9B | 0.9738 | 0.9514 | **0.9625** |
| Qwen3.5-4B | 4B | 0.9711 | 0.9438 | **0.9572** |
| Ministral-8B | 8B | 0.9532 | 0.9532 | **0.9532** |
| Ministral-3B | 3B | 0.9300 | 0.8825 | **0.9056** |
| LLaMA-3.1-8B | 8B | 0.8963 | 0.9421 | **0.9186** |
| Gemma-3-12B | 12B | 0.8749 | 0.9591 | **0.9151** |
| Qwen3.5-2B | 2B | 0.9186 | 0.8748 | **0.8962** |
| Gemma-3-4B | 4B | 0.8005 | 0.9020 | **0.8482** |
| LLaMA-3.2-3B | 3B | 0.7367 | 0.7794 | **0.7575** |

### 2.4 Canonical-History

| Model | Params | P | R | F1 | Note |
|---|---|---|---|---|---|
| Qwen3.5-9B | 9B | 1.0318 | 0.9558 | **0.9924** | dagger-P>1 artifact |
| Qwen3.5-4B | 4B | 1.0124 | 0.9362 | **0.9728** | dagger-P>1 artifact |
| Ministral-8B | 8B | 1.0332 | 0.9689 | **1.0000** | dagger-P>1 artifact |
| Ministral-3B | 3B | 0.9103 | 0.8642 | **0.8866** |  |
| LLaMA-3.1-8B | 8B | 0.8998 | 0.8822 | **0.8909** |  |
| Gemma-3-12B | 12B | 0.9112 | 0.9574 | **0.9338** |  |
| Qwen3.5-2B | 2B | 0.7729 | 0.8020 | **0.7871** |  |
| Gemma-3-4B | 4B | 0.8464 | 0.8478 | **0.8471** |  |
| LLaMA-3.2-3B | 3B | 0.2361 | 0.7136 | **0.3548** |  |

> Canonical precision > 1 occurs when `canonical_source=new`: the model correctly identifies a mention
> not in `related_object_history`, counted as TP without incrementing the predicted-history denominator.

### 2.5 Full-Instance (ranked)

| # | Model | Params | P | R | F1 |
|---|---|---|---|---|---|
| 1 | Qwen3.5-9B | 9B | 0.9721 | 0.9497 | **0.9608** |
| 2 | Ministral-8B | 8B | 0.9514 | 0.9514 | **0.9514** |
| 3 | Qwen3.5-4B | 4B | 0.9632 | 0.9361 | **0.9495** |
| 4 | Gemma-3-12B | 12B | 0.8718 | 0.9557 | **0.9118** |
| 5 | Ministral-3B | 3B | 0.9255 | 0.8782 | **0.9012** |
| 6 | LLaMA-3.1-8B | 8B | 0.8736 | 0.9182 | **0.8953** |
| 7 | Qwen3.5-2B | 2B | 0.8730 | 0.8313 | **0.8517** |
| 8 | Gemma-3-4B | 4B | 0.7816 | 0.8807 | **0.8282** |
| 9 | LLaMA-3.2-3B | 3B | 0.6965 | 0.7368 | **0.7161** |

## 3. Latency (PHASE 1 — Speed Test, 100 samples)

> All measurements from local vLLM inference (single GPU, concurrency=16).

| # | Model | Params | Avg (s) | Median (s) | P95 (s) | P99 (s) | Tok/s |
|---|---|---|---|---|---|---|---|
| 1 | Qwen3.5-2B | 2B | **0.808** | 0.831 | 1.640 | 2.292 | 188.3 |
| 2 | Ministral-3B | 3B | **0.852** | 0.893 | 1.980 | 2.557 | 157.7 |
| 3 | LLaMA-3.2-3B | 3B | **0.995** | 0.887 | 2.334 | 2.950 | 123.3 |
| 4 | Qwen3.5-4B | 4B | **1.521** | 1.677 | 3.480 | 4.605 | 89.6 |
| 5 | Ministral-8B | 8B | **1.816** | 1.937 | 3.898 | 5.039 | 81.1 |
| 6 | Gemma-3-4B | 4B | **2.037** | 2.036 | 3.913 | 5.267 | 89.6 |
| 7 | LLaMA-3.1-8B | 8B | **2.285** | 2.390 | 4.916 | 6.464 | 57.3 |
| 8 | Qwen3.5-9B | 9B | **4.624** | 5.081 | 10.596 | 14.041 | 30.7 |
| 9 | Gemma-3-12B | 12B | **5.175** | 5.390 | 10.371 | 13.840 | 34.7 |

### 3.1 Accuracy vs. Latency Efficiency

| # | Model | Params | sample_acc | Avg latency (s) | Acc/Latency |
|---|---|---|---|---|---|
| 1 | Qwen3.5-2B | 2B | 0.8680 | 0.808 | **1.074** |
| 2 | Ministral-3B | 3B | 0.9097 | 0.852 | **1.068** |
| 3 | LLaMA-3.2-3B | 3B | 0.6911 | 0.995 | **0.694** |
| 4 | Qwen3.5-4B | 4B | 0.9537 | 1.521 | **0.627** |
| 5 | Ministral-8B | 8B | 0.9514 | 1.816 | **0.524** |
| 6 | LLaMA-3.1-8B | 8B | 0.8919 | 2.285 | **0.390** |
| 7 | Gemma-3-4B | 4B | 0.7668 | 2.037 | **0.376** |
| 8 | Qwen3.5-9B | 9B | 0.9629 | 4.624 | **0.208** |
| 9 | Gemma-3-12B | 12B | 0.8888 | 5.175 | **0.172** |

## 4. Per-Role Accuracy

| Model | Params | Assistant | User | Delta |
|---|---|---|---|---|
| Qwen3.5-9B | 9B | 0.9603 (653/680) | 0.9659 (594/615) | -0.0056 |
| Qwen3.5-4B | 4B | 0.9574 (651/680) | 0.9496 (584/615) | +0.0078 |
| Ministral-8B | 8B | 0.9485 (645/680) | 0.9545 (587/615) | -0.0059 |
| Ministral-3B | 3B | 0.9132 (621/680) | 0.9057 (557/615) | +0.0075 |
| LLaMA-3.1-8B | 8B | 0.8926 (607/680) | 0.8911 (548/615) | +0.0016 |
| Gemma-3-12B | 12B | 0.8750 (595/680) | 0.9041 (556/615) | -0.0291 |
| Qwen3.5-2B | 2B | 0.8809 (599/680) | 0.8537 (525/615) | +0.0272 |
| Gemma-3-4B | 4B | 0.7721 (525/680) | 0.7610 (468/615) | +0.0111 |
| LLaMA-3.2-3B | 3B | 0.7265 (494/680) | 0.6520 (401/615) | +0.0744 |

## 5. Per-Domain Accuracy

| Domain | n | Q-2B | Q-4B | Q-9B | Min-8B | Min-3B | Llm-8B | Llm-3B | Gem-12B | Gem-4B |
|---|---|---|---|---|---|---|---|---|---|---|
| academia | 53 | 0.868 | 0.943 | 0.925 | 0.925 | 0.962 | 0.868 | 0.679 | 0.887 | 0.679 |
| ecommerce | 111 | 0.910 | 0.973 | 0.964 | 0.973 | 0.883 | 0.874 | 0.604 | 0.856 | 0.721 |
| energy and utilities | 87 | 0.943 | 0.977 | 0.977 | 0.954 | 0.954 | 0.920 | 0.782 | 0.908 | 0.724 |
| finance | 133 | 0.872 | 0.970 | 0.985 | 0.977 | 0.887 | 0.902 | 0.752 | 0.887 | 0.805 |
| food and restaurants * | 15 | 0.800 | 0.933 | 0.933 | 0.867 | 0.867 | 0.800 | 0.533 | 0.867 | 1.000 |
| government and public services * | 61 | 0.770 | 0.885 | 0.918 | 0.885 | 0.852 | 0.820 | 0.705 | 0.852 | 0.738 |
| human resources and employment * | 45 | 0.933 | 0.956 | 1.000 | 0.978 | 0.867 | 0.978 | 0.733 | 0.933 | 0.822 |
| information security | 74 | 0.865 | 0.959 | 0.946 | 0.946 | 0.932 | 0.959 | 0.797 | 0.905 | 0.730 |
| insurance | 71 | 0.986 | 0.972 | 0.986 | 0.972 | 0.944 | 0.958 | 0.817 | 0.930 | 0.915 |
| legal services * | 50 | 0.860 | 0.900 | 0.980 | 0.920 | 0.880 | 0.940 | 0.640 | 0.800 | 0.780 |
| media | 63 | 0.778 | 0.952 | 0.937 | 0.937 | 0.889 | 0.810 | 0.667 | 0.857 | 0.730 |
| medicine | 97 | 0.845 | 0.948 | 0.928 | 0.897 | 0.897 | 0.784 | 0.649 | 0.876 | 0.763 |
| real estate | 63 | 0.873 | 0.984 | 1.000 | 1.000 | 0.984 | 0.968 | 0.603 | 0.952 | 0.730 |
| software development | 50 | 0.800 | 0.960 | 1.000 | 1.000 | 0.860 | 0.860 | 0.560 | 0.920 | 0.840 |
| sports | 60 | 0.850 | 0.883 | 0.900 | 0.917 | 0.917 | 0.900 | 0.533 | 0.933 | 0.733 |
| technology | 69 | 0.812 | 0.913 | 0.986 | 0.986 | 0.913 | 0.899 | 0.681 | 0.841 | 0.725 |
| telecommunications | 70 | 0.814 | 0.986 | 0.957 | 0.943 | 0.871 | 0.886 | 0.729 | 0.886 | 0.814 |
| transportation | 44 | 0.886 | 0.977 | 0.977 | 0.977 | 0.932 | 0.886 | 0.773 | 0.932 | 0.750 |
| travel and hospitality * | 79 | 0.911 | 0.975 | 0.975 | 0.949 | 0.962 | 0.911 | 0.709 | 0.886 | 0.759 |

`*` = OOD domain. Columns: Q-2B=Qwen3.5-2B, Q-4B=Qwen3.5-4B, Q-9B=Qwen3.5-9B, Min-8B=Ministral-8B, Min-3B=Ministral-3B, Llm-8B=LLaMA-3.1-8B, Llm-3B=LLaMA-3.2-3B, Gem-12B=Gemma-3-12B, Gem-4B=Gemma-3-4B

### 5.1 OOD Domain Average (5 new domains)

| Model | Params | OOD avg | Overall | Delta |
|---|---|---|---|---|
| Qwen3.5-9B | 9B | 0.9612 | 0.9629 | -0.0017 |
| Qwen3.5-4B | 4B | 0.9298 | 0.9537 | -0.0239 |
| Ministral-8B | 8B | 0.9198 | 0.9514 | -0.0315 |
| Ministral-3B | 3B | 0.8856 | 0.9097 | -0.0241 |
| LLaMA-3.1-8B | 8B | 0.8898 | 0.8919 | -0.0021 |
| Gemma-3-12B | 12B | 0.8677 | 0.8888 | -0.0211 |
| Qwen3.5-2B | 2B | 0.8550 | 0.8680 | -0.0129 |
| Gemma-3-4B | 4B | 0.8199 | 0.7668 | +0.0531 |
| LLaMA-3.2-3B | 3B | 0.6641 | 0.6911 | -0.0270 |

## 6. Error Analysis

| Model | false_neg | false_pos | mention_err | canonical_err | inst_count_err | parse_err |
|---|---|---|---|---|---|---|
| Qwen3.5-9B | 20 | 6 | 16 | 2 | 4 | 0 |
| Qwen3.5-4B | 23 | 8 | 16 | 7 | 6 | 0 |
| Ministral-8B | 12 | 17 | 26 | 2 | 6 | 0 |
| Ministral-3B | 42 | 3 | 61 | 5 | 6 | 1 |
| LLaMA-3.1-8B | 17 | 56 | 37 | 22 | 7 | 1 |
| Gemma-3-12B | 3 | 101 | 32 | 3 | 4 | 3 |
| Qwen3.5-2B | 19 | 28 | 42 | 37 | 44 | 1 |
| Gemma-3-4B | 0 | 192 | 35 | 18 | 57 | 0 |
| LLaMA-3.2-3B | 13 | 191 | 59 | 38 | 66 | 35 |

## 7. Rankings Summary

| Metric | #1 | #2 | #3 |
|---|---|---|---|
| sample_acc | Qwen3.5-9B (0.9629) | Qwen3.5-4B (0.9537) | Ministral-8B (0.9514) |
| found_F1 | Qwen3.5-9B (0.9816) | Ministral-8B (0.9797) | Qwen3.5-4B (0.9780) |
| mention_F1 | Qwen3.5-9B (0.9625) | Qwen3.5-4B (0.9572) | Ministral-8B (0.9532) |
| canonical_F1 | Ministral-8B (1.0000) | Qwen3.5-9B (0.9924) | Qwen3.5-4B (0.9728) |
| full_inst_F1 | Qwen3.5-9B (0.9608) | Ministral-8B (0.9514) | Qwen3.5-4B (0.9495) |
| fastest avg | Qwen3.5-2B (0.808s) | Ministral-3B (0.852s) | LLaMA-3.2-3B (0.995s) |
| fastest median | Qwen3.5-2B (0.831s) | LLaMA-3.2-3B (0.887s) | Ministral-3B (0.893s) |

## 8. Key Insights

### 8.1 Qwen3.5 dominates accuracy

The Qwen3.5-9B model achieves the highest sample accuracy (0.9629) and full-instance F1 (0.9608). Qwen3.5-4B ranks second (0.9537), outperforming Ministral-8B (0.9514) with fewer parameters. The Qwen3.5 family shows a clear, near-linear scaling trend: 2B (0.868) → 4B (0.954) → 9B (0.963).

### 8.2 Ministral-3B is the efficiency champion

With 3B parameters, Ministral-3B achieves 0.910 sample accuracy at only 0.852s average latency, giving the best accuracy-per-latency ratio (1.067) of all models. It also has near-perfect found precision (0.996, only 3 false positives), reflecting strong instruction-following.

### 8.3 Canonical-history is the hardest sub-task

Canonical-history F1 is consistently the lowest metric across all models. The gap is most severe for LLaMA-3.2-3B: canonical_F1=0.355, precision=0.236 — it predicts 1,847 canonical mentions against 611 ground truth (3x over-prediction). Top models (Qwen3.5-9B: F1=0.992, Ministral-8B: F1=1.000) nearly solve this sub-task. This sub-task should be highlighted as the primary difficulty of the benchmark in a paper.

### 8.4 Gemma-3-4B: extreme recall bias

Gemma-3-4B achieves found_recall=1.000 — it never misses a positive — but found_precision=0.788, producing 192 false positives. The model effectively always outputs found=true. This is useful for high-recall retrieval but harmful for precision-critical annotation pipelines.

### 8.5 Parameter count does not determine rank

Performance ranking does not follow parameter count. Qwen3.5-4B (4B params) outperforms Gemma-3-12B (12B) on sample accuracy (0.954 vs. 0.889). LLaMA-3.2-3B (3B) ranks last despite sharing its size class with Ministral-3B (0.691 vs. 0.910). Architecture, training data, and instruction tuning dominate over raw scale.

### 8.6 OOD generalisation is strong for top models

For Qwen3.5-9B, Qwen3.5-4B, Ministral-8B, and Ministral-3B, the average OOD-domain accuracy is within 1% of their overall accuracy. The hardest OOD domain across all models is "government and public services" (complex entity-dense language). Food & restaurants and human resources & employment are the easiest OOD domains, with multiple models scoring above 0.93.

### 8.7 Speed-accuracy Pareto frontier

Three distinct tiers emerge:
- **Fast & accurate:** Ministral-3B (0.852s, acc=0.910), Qwen3.5-2B (0.808s, acc=0.868) — best for latency-critical deployment
- **Balanced:** Ministral-8B (1.816s, acc=0.951), LLaMA-3.1-8B (2.285s, acc=0.892) — good throughput with high accuracy
- **High-accuracy, slow:** Qwen3.5-9B (4.624s, acc=0.963), Gemma-3-12B (5.175s, acc=0.889) — accuracy-critical offline processing

### 8.8 Role parity

All models show near-parity between assistant and user turns (|Delta| < 0.02 for 7 of 9 models). LLaMA-3.2-3B has the largest gap (assistant 0.726, user 0.652, Delta=0.074), suggesting difficulty with user-initiated queries. This is worth noting as a robustness dimension in the paper.

---

*Generated from `summary.json` files in each model subdirectory.*
*Speed metrics from PHASE 1 (100-sample speed test on vLLM). Accuracy metrics from PHASE 2 (full 1,295-sample evaluation).*
