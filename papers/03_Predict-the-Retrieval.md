# Predict the Retrieval! Test Time Adaptation for Retrieval Augmented Generation


# Predict the Retrieval! Test Time Adaptation for Retrieval Augmented Generation

###### Abstract

Retrieval-Augmented Generation (RAG) has emerged as a powerful approach for enhancing large language models’ question-answering capabilities through the integration of external knowledge. However, when adapting RAG systems to specialized domains, challenges arise from distribution shifts, resulting in suboptimal generalization performance. In this work, we propose TTARAG, a test-time adaptation method that dynamically updates the language model’s parameters during inference to improve RAG system performance in specialized domains. Our method introduces a simple yet effective approach where the model learns to predict retrieved content, enabling automatic parameter adjustment to the target domain. Through extensive experiments across six specialized domains, we demonstrate that TTARAG achieves substantial performance improvements over baseline RAG systems. Code available at <https://github.com/sunxin000/TTARAG>.


Index Terms— 
Retrieval Augmented Generation, Test-Time Adaptation, Self-Supervised Learning, Domain Adaptation


## 1 Introduction

Retrieval-Augmented Generation (RAG) [[1](https://arxiv.org/html/2601.11443v1#bib.bib1), [2](https://arxiv.org/html/2601.11443v1#bib.bib2), [3](https://arxiv.org/html/2601.11443v1#bib.bib3)] has emerged as a crucial approach for enhancing large language models (LLMs) [[4](https://arxiv.org/html/2601.11443v1#bib.bib4), [5](https://arxiv.org/html/2601.11443v1#bib.bib5), [6](https://arxiv.org/html/2601.11443v1#bib.bib6)] by addressing their inherent knowledge limitations. Through the integration of external knowledge sources [[7](https://arxiv.org/html/2601.11443v1#bib.bib7), [8](https://arxiv.org/html/2601.11443v1#bib.bib8), [9](https://arxiv.org/html/2601.11443v1#bib.bib9)], RAG systems not only improve the accuracy of LLM responses but also help mitigate hallucination issues while eliminating the need for extensive model retraining.


However, while most current research has focused on the effectiveness of RAG systems for general domains, significant challenges persist in adapting these systems to specialized domains. These systems often struggle with distribution shifts and domain-specific data dependencies [[10](https://arxiv.org/html/2601.11443v1#bib.bib10), [11](https://arxiv.org/html/2601.11443v1#bib.bib11)], frequently failing to accurately utilize information in domain-specific contexts [[12](https://arxiv.org/html/2601.11443v1#bib.bib12), [13](https://arxiv.org/html/2601.11443v1#bib.bib13)].


To address these challenges, test-time adaptation (TTA) [[14](https://arxiv.org/html/2601.11443v1#bib.bib14), [15](https://arxiv.org/html/2601.11443v1#bib.bib15)] offers a promising solution. TTA allows models to dynamically adjust their parameters at inference time, typically through self-supervised learning objectives derived from the test data itself, without requiring new labeled data [[16](https://arxiv.org/html/2601.11443v1#bib.bib16)]. This characteristic makes TTA particularly well-suited for scenarios involving domain shifts or distribution changes not seen during initial training, as it enables the model to rapidly specialize to the immediate context. Building on these insights, we propose a simple yet powerful method for adapting RAG systems during inference: TTARAG. Our approach generates self-supervised learning signals by dividing retrieved passages into prefix-suffix pairs and training the model to predict suffix content from prefix context. This technique enables LLMs to perform real-time parameter updates when encountering new domains, effectively leveraging domain knowledge stored within the model parameters.


Through extensive experiments across six specialized domains, we demonstrate that TTARAG achieves substantial performance improvements over baseline RAG systems. Our approach consistently outperforms both standard RAG and baselines like Chain-of-Thought and In-Context Learning, achieving the best results in 19 out of 24 experimental settings while maintaining computational efficiency. These results validate the effectiveness of our approach for domain-specific applications.


## 2 Related Work

### 2.1 Retrieval Augmented Generation

Retrieval Augmented Generation (RAG) [[2](https://arxiv.org/html/2601.11443v1#bib.bib2)] has emerged as a powerful paradigm for enhancing large language models (LLMs) with external knowledge. By integrating a retrieval system with LLMs, RAG enables models to access and leverage external knowledge sources during generation, effectively addressing the limitations of static, parameterized knowledge in LLMs.


Recent advances in RAG have focused on several key directions. First, researchers have explored dynamic retrieval processes [[17](https://arxiv.org/html/2601.11443v1#bib.bib17), [18](https://arxiv.org/html/2601.11443v1#bib.bib18)] to improve the relevance of retrieved content. Second, various filtering mechanisms [[19](https://arxiv.org/html/2601.11443v1#bib.bib19), [20](https://arxiv.org/html/2601.11443v1#bib.bib20)] have been developed to eliminate irrelevant contexts and enhance RAG robustness. Additionally, instruction-tuning methods [[21](https://arxiv.org/html/2601.11443v1#bib.bib21), [22](https://arxiv.org/html/2601.11443v1#bib.bib22)] have been specifically designed to improve LLMs’ search and RAG capabilities.


### 2.2 Test-time inference adaptation

Test-time inference adaptation aims to adapt pre-trained models to unlabeled test data during inference time without accessing the source training data. This paradigm has gained increasing attention as a practical solution for handling distribution shifts in real-world applications [[23](https://arxiv.org/html/2601.11443v1#bib.bib23), [24](https://arxiv.org/html/2601.11443v1#bib.bib24)]. Unlike traditional domain adaptation methods that require simultaneous access to both source and target domains, test-time adaptation only needs the pre-trained model and target data, making it more privacy-friendly and storage-efficient [[25](https://arxiv.org/html/2601.11443v1#bib.bib25)].


Early works in this direction focused on hypothesis transfer learning [[26](https://arxiv.org/html/2601.11443v1#bib.bib26)], where models trained on source domains are adapted to target domains with limited labeled data. Recent advances have extended this to fully unsupervised scenarios, leveraging techniques like entropy minimization [[23](https://arxiv.org/html/2601.11443v1#bib.bib23)], self-training [[14](https://arxiv.org/html/2601.11443v1#bib.bib14)], and test-time normalization statistics calibration [[27](https://arxiv.org/html/2601.11443v1#bib.bib27)] to adapt models using only unlabeled test samples.


Building on these advances, TTARAG introduces a simple yet effective approach for test-time adaptation in retrieval-augmented generation. By learning to predict subsequent tokens in retrieved passages, TTARAG enables fully unsupervised adaptation without requiring access to source domain data or labeled examples.


## 3 Methodology

Our approach introduces a test-time adaptation mechanism for retrieval-augmented generation that enables model optimization during inference without access to ground truth labels. The key innovation lies in designing a self-supervised learning objective using retrieved passages as supervision signals. We hypothesize that by training the model to predict a suffix of a passage given its prefix and the original query, the model is encouraged to better understand the contextual relationships and domain-specific language patterns within the retrieved documents. This task inherently pushes the model to learn how information flows and connects within relevant texts, thereby aligning its internal representations more closely with the nuances of the target domain encountered at test time.


### 3.1 Overview

Given a test input query qq and retrieved passages {p1,…,pk}\{p\_{1},...,p\_{k}\}, we formulate a self-supervised adaptation objective by splitting passages into prefix-suffix pairs for prediction:

|  | ℒa​d​a​p​t=−∑i=1klog⁡P​(pis​u​f​f​i​x|pip​r​e​f​i​x,q;θ)\mathcal{L}\_{adapt}=-\sum\_{i=1}^{k}\log P(p\_{i}^{suffix}|p\_{i}^{prefix},q;\theta) |  | (1) |
| --- | --- | --- | --- |

where θ\theta represents the model parameters.


![Refer to caption](03_Predict-the-Retrieval_images/x1.png)

Fig. 1: Comparison between Standard RAG and TTARAG systems.


### 3.2 Context Processing

The adaptation process begins with careful processing of the retrieved passages to create meaningful prefix-suffix pairs for training.


Length Filtering. To ensure sufficient context for meaningful adaptation, passages shorter than a configured minimum length threshold are filtered out.


Passage Splitting. Each passage is split into prefix-suffix pairs using a two-tier strategy:

- •
  
  Primary Strategy Passages are split at first natural linguistic boundaries marked by punctuation (periods, commas, semicolons, colons, exclamation marks, and question marks)
- •
  
  Fallback Strategy When no suitable punctuation-based split exists, the passage is divided at its midpoint, ensuring each segment contains at least three words.


### 3.3 Parameter Adaptation Process

#### 3.3.1 Initialization

Prior to the adaptation process, the model parameters are reset to their original pre-trained state to ensure a clean starting point for each adaptation iteration.


#### 3.3.2 Training Loop

For each batch of prefix-suffix pairs:

|  | θt=θt−1−α⋅1N​∑i=1N∇θℒa​d​a​p​ti\theta\_{t}=\theta\_{t-1}-\alpha\cdot\frac{1}{N}\sum\_{i=1}^{N}\nabla\_{\theta}\mathcal{L}\_{adapt}^{i} |  | (2) |
| --- | --- | --- | --- |

where NN is the gradient accumulation steps and ℒa​d​a​p​ti\mathcal{L}\_{adapt}^{i} is the loss for the ii-th pair.


During training, the complete text (prefix and suffix) is first tokenized. The model then computes the loss on the suffix prediction task, where prefix tokens are masked during label preparation. To ensure stable training, gradients are accumulated over two steps and clipped to a maximum norm threshold. The AdamW optimizer then updates model parameters using these accumulated gradients. Since we only adapt on 3 prefix-suffix pairs in our experiments, the computational overhead remains acceptable.


### 3.4 Response Generation

After parameter adaptation, the model generates the final response using the adapted parameters θ′\theta^{\prime}:

|  | y=arg​maxy⁡P​(y|q,{p1,…,pk};θ′)y=\operatorname\*{arg\,max}\_{y}P(y|q,\{p\_{1},...,p\_{k}\};\theta^{\prime}) |  | (3) |
| --- | --- | --- | --- |


## 4 Experiments

Table 1: Performance comparison across different domains. Numbers represent accuracy scores (%). Best results for each model group are shown in bold. naive-rag refers to the respective LLM with standard retrieval-augmentation (top 20 passages) but without TTARAG adaptation.

|  | CRAG | | | | | | Medical | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Finance | Sports | Music | Movie | Open | Overall | BioASQ | PubMed |
| Llama-3.1-8b-it | | | | | | | | |
| naive-rag | 17.4 | 27.6 | 34.9 | 31.3 | 42.4 | 29.8 | 55.6 | 46.6 |
| +CoT | 17.9 | 30.2 | 37.6 | 31.5 | 45.8 | 31.6 | 54.6 | 50.8 |
| +ICL | 16.1 | 24.8 | 33.5 | 29.4 | 40.4 | 28.0 | 49.8 | 53.6 |
| TTARAG | 20.1 | 29.5 | 37.7 | 34.6 | 41.5 | 31.9 | 75.0 | 57.4 |
| Δ\Delta vs naive-rag | +2.7 | +1.9 | +2.8 | +3.3 | -0.9 | +2.1 | +19.4 | +10.8 |
| Llama-2-7b-chat | | | | | | | | |
| naive-rag | 14.7 | 23.2 | 36.5 | 30.4 | 39.2 | 27.8 | 54.1 | 47.6 |
| +CoT | 15.7 | 26.7 | 34.3 | 31.4 | 41.5 | 29.1 | 55.1 | 48.2 |
| +ICL | 16.0 | 24.2 | 36.1 | 31.2 | 39.2 | 28.4 | 55.6 | 43.4 |
| TTARAG | 16.4 | 25.8 | 40.7 | 33.8 | 41.1 | 30.5 | 71.8 | 54.0 |
| Δ\Delta vs naive-rag | +1.7 | +2.6 | +4.2 | +3.4 | +1.9 | +2.7 | +17.7 | +6.4 |
| ChatGLM-3-6b | | | | | | | | |
| naive-rag | 9.8 | 18.7 | 31.4 | 22.4 | 33.4 | 22.0 | 51.4 | 19.8 |
| +CoT | 12.7 | 20.6 | 28.4 | 25.8 | 33.9 | 23.6 | 44.3 | 22.4 |
| +ICL | 9.9 | 18.2 | 30.8 | 22.1 | 33.0 | 21.8 | 50.8 | 19.2 |
| TTARAG | 14.0 | 22.1 | 33.5 | 25.5 | 38.1 | 25.7 | 58.4 | 44.8 |
| Δ\Delta vs naive-rag | +4.2 | +3.4 | +2.1 | +3.1 | +4.7 | +3.7 | +7.0 | +25.0 |


Table 2: Number of samples in each domain of CRAG dataset.

| Domain | Finance | Sports | Music | Movie | Open |
| --- | --- | --- | --- | --- | --- |
| #Samples | 661 | 519 | 373 | 611 | 542 |


### 4.1 Datasets

We conduct experiments on CRAG [[28](https://arxiv.org/html/2601.11443v1#bib.bib28)] as the evaluation benchmark. CRAG is a comprehensive RAG benchmark containing 2,706 question-answer pairs across five domains: Finance, Sports, Music, Movie, and Open domain. The questions are constructed through web content-based creation where annotators formulate real-world questions answerable through web search. For retrieval, CRAG provides each question with up to 50
full HTML pages retrieved through the Brave Search API, with
the top 5 pages showing a retrieval recall of approximately
69%. Then MiniLM-v6 is employed to reranker all the sentence in the HTML and select top 20 sentences as the retrieval content. The statistics of the CRAG dataset are shown in Table [2](https://arxiv.org/html/2601.11443v1#S4.T2 "Table 2 ‣ 4 Experiments ‣ Predict the Retrieval! Test Time Adaptation for Retrieval Augmented Generation").


To evaluate the effectiveness of TTARAG in the medical domain, we conduct additional experiments on two specialized datasets: PubMedQA [[9](https://arxiv.org/html/2601.11443v1#bib.bib9)], which contains 1,000 biomedical research question-answer pairs, and BioASQ [[29](https://arxiv.org/html/2601.11443v1#bib.bib29)], comprising 500 expert-curated question-answer pairs from the biomedical literature. For these medical datasets, as they do not come with pre-retrieved passages, we employed a standard BM25 retriever implemented using the Pyserini library to fetch the top 5 relevant passages from their respective corpora (PubMed abstracts for PubMedQA and biomedical articles for BioASQ). This retriever setup was also consistently applied when evaluating baseline models on these datasets.


### 4.2 Evaluation Metrics

Instead of relying on exact match (EM)—which is sensitive to paraphrasing, aliases, and formatting variations—we adopt the CRAG evaluation protocol and employ an LLM-as-a-judge (Qwen2.5-72B-Instruct) to determine semantic equivalence between model predictions and reference answers. This LLM-based judging approach emphasizes the underlying meaning rather than surface-level string similarity, enabling robust assessment of whether the model output conveys the same information as the gold answer. In our evaluation, the LLM-judger demonstrates 98% agreement with human annotators on a random sample of 100 examples, indicating high reliability and consistency. Overall, this methodology provides a more accurate and meaningful evaluation of model performance compared to traditional exact match metrics.


### 4.3 Baselines

We evaluate TTARAG against several strong baselines, including prompting techniques (Chain-of-Thought [[30](https://arxiv.org/html/2601.11443v1#bib.bib30)], In-Context Learning [[5](https://arxiv.org/html/2601.11443v1#bib.bib5)]) and state-of-the-art pretrained RAG models (Ret-Robust
[[19](https://arxiv.org/html/2601.11443v1#bib.bib19)], RAAT [[31](https://arxiv.org/html/2601.11443v1#bib.bib31)] , Self-RAG [[32](https://arxiv.org/html/2601.11443v1#bib.bib32)]).


### 4.4 Experimental Results

Table [1](https://arxiv.org/html/2601.11443v1#S4.T1 "Table 1 ‣ 4 Experiments ‣ Predict the Retrieval! Test Time Adaptation for Retrieval Augmented Generation") presents comprehensive evaluation results across different domains and model architectures. Several key observations emerge from our experiments: TTARAG demonstrates consistent improvements across specialized domains, with Llama-3.1-8b-it showing notable gains in Finance (+2.7%), Music (+2.8%), and Movie (+3.3%) domains, and particularly strong performance in medical domains (BioASQ +19.4%, PubMedQA +10.8%). All three model architectures benefit from our approach: Llama-3.1-8b-it achieves the highest overall accuracy (31.9%), Llama-2-7b-chat shows remarkable adaptation capability in medical domains (+17.7% on BioASQ), and ChatGLM-3-6b demonstrates significant improvements in PubMedQA (+25.0%) and consistent gains across CRAG domains (+3.7% overall). While both CoT and ICL show some improvements over the naive-rag models, TTARAG consistently outperforms these baselines in specialized domains, with the only exception being Open domain tasks where CoT occasionally shows stronger performance, particularly with Llama-3.1-8b-it (45.8% vs 41.5%).


Table 3: Comparison of our method with state-of-the-art pretrained RAG models. All models utilize Llama-2-7b-chat as the backbone.

|  | CRAG | | | | | | Medical | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | Finance | Sports | Music | Movie | Open | Overall | BioASQ | PubMedQA |
| Naive-RAG (Llama-2-7b-chat) | 14.7 | 23.2 | 36.5 | 30.4 | 39.2 | 27.8 | 54.1 | 47.6 |
| Ret-Robust | 14.6 | 20.6 | 33.2 | 32.4 | 33.5 | 26.1 | 24.7 | 28.4 |
| RAAT | 13.4 | 18.1 | 28.6 | 25.2 | 31.7 | 22.7 | 64.9 | 46.6 |
| Self-rag | 11.4 | 19.8 | 22.5 | 20.9 | 26.7 | 19.8 | 57.1 | 43.4 |
| TTARAG | 16.4 | 25.8 | 40.7 | 33.8 | 41.1 | 30.5 | 71.8 | 54.0 |


Table [3](https://arxiv.org/html/2601.11443v1#S4.T3 "Table 3 ‣ 4.4 Experimental Results ‣ 4 Experiments ‣ Predict the Retrieval! Test Time Adaptation for Retrieval Augmented Generation") presents a performance comparison between different RAG models across various domains. The compared pretrained RAG models (Ret-Robust, RAAT, and Self-RAG) are general-domain models and were not further fine-tuned on the specific downstream task domains used in our evaluation. Our method, TTARAG, by contrast, adapts at test-time using unlabeled passages from the target domain. Notably, three of the models (Ret-robust, RAAT, and Self-rag) are pre-trained RAG models based on Llama-2. Despite Ret-robust using the larger Llama-2-13b as its base, and RAAT and Self-rag using Llama-2-7b, all three pre-trained RAG models perform worse than the Llama-2-7b-chat model (which achieves 27.8% overall accuracy). This underperformance is consistent across most domains, with only RAAT showing strength in the BioASQ medical domain (64.9%). The results suggest that current RAG pre-training methods have limited generalization capabilities, as they fail to match or exceed the performance of the base model, even when using larger model architectures. TTARAG outperforms all other models across all domains, demonstrating the effectiveness of its approach compared to existing RAG pre-training methods.


The effectiveness of segment-based adaptation We compare our segment-based approach (splitting passages into prefix-suffix pairs) with a baseline that does not segment the passage, where we perform next-token prediction on the entire passage without segmentation. The results in Table [4](https://arxiv.org/html/2601.11443v1#S4.T4 "Table 4 ‣ 4.4 Experimental Results ‣ 4 Experiments ‣ Predict the Retrieval! Test Time Adaptation for Retrieval Augmented Generation") demonstrate that the segmentation strategy yields consistent performance gains across all model architectures: +1.1% for Llama-3.1-8b-it, +0.4% for Llama-2-7b-chat, and +0.7% for ChatGLM-3-6b. We attribute these improvements to the front-to-back prediction task better aligning with natural language understanding compared to token-by-token prediction, enabling more effective parameter updates. The larger improvement observed with Llama-3.1-8b-it (+1.1%) suggests that higher-capacity models may particularly benefit from structured adaptation approaches.


Table 4: The effectiveness of segmentation.

| Strategy | Llama-3.1-8b-it | Llama-2-7b-chat | ChatGLM-3-6b |
| --- | --- | --- | --- |
| TTARAG | 31.9 | 30.5 | 25.7 |
| wo seg | 30.8 | 30.1 | 25.0 |


![Refer to caption](03_Predict-the-Retrieval_images/x2.png)

Fig. 2: Accuracy vs. Learning Rate


![Refer to caption](03_Predict-the-Retrieval_images/x3.png)

Fig. 3: Accuracy vs. Number of Adaptation Pairs


Hyperparameter Analysis.
We also conduct hyper-parameter analysis about the number of adaptation pairs and learning rate.
As shown in Figure [2](https://arxiv.org/html/2601.11443v1#S4.F2 "Figure 2 ‣ 4.4 Experimental Results ‣ 4 Experiments ‣ Predict the Retrieval! Test Time Adaptation for Retrieval Augmented Generation") and Figure [3](https://arxiv.org/html/2601.11443v1#S4.F3 "Figure 3 ‣ 4.4 Experimental Results ‣ 4 Experiments ‣ Predict the Retrieval! Test Time Adaptation for Retrieval Augmented Generation"), our experiments show that learning rates between 1e-6 to 1e-5 provide optimal performance, with 3-5 adaptation pairs yielding the best balance between effectiveness and efficiency. However, the performance of TTARAG is not sensitive to the number of adaptation pairs and learning rate. In almost all cases, the performance of TTARAG is better than the naive-rag model.


On the computation efficiency.
To evaluate the computational overhead of our approach, we measure the total inference time across different configurations and compare it with baseline methods. The computation time reported in Table [5](https://arxiv.org/html/2601.11443v1#S4.T5 "Table 5 ‣ 4.4 Experimental Results ‣ 4 Experiments ‣ Predict the Retrieval! Test Time Adaptation for Retrieval Augmented Generation") was measured on the CRAG dataset, processing 2,706 queries on one NVIDIA A100 GPU. For each query, up to 5 passages were used for TTARAG adaptation. Table [5](https://arxiv.org/html/2601.11443v1#S4.T5 "Table 5 ‣ 4.4 Experimental Results ‣ 4 Experiments ‣ Predict the Retrieval! Test Time Adaptation for Retrieval Augmented Generation") shows the total and average inference times for different numbers of adaptation pairs (1-5), compared against Chain-of-Thought (CoT) and the original model without adaptation. The results are based on processing 2,706 queries from the CRAG dataset.


Table 5: Computation time analysis. Total and average inference times (in seconds) for different numbers of adaptation pairs, compared against CoT and vanilla model.

| Metric | 1pair | 2pair | 3pair | 4pair | 5pair | CoT | Naive-RAG |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Total (s) | 4,740 | 5,723 | 6,621 | 7,001 | 7,023 | 11,688 | 961 |
| Avg (s) | 1.75 | 2.11 | 2.45 | 2.59 | 2.60 | 4.32 | 0.36 |


While our method does introduce additional computational overhead compared to the original model, it remains significantly more efficient than CoT. The average processing time per query ranges from 1.75s (1-pair) to 2.60s (5-pair), which is substantially lower than CoT’s 4.32s. This demonstrates that TTARAG achieves its performance improvements with reasonable computational cost, making it practical for real-world applications.


## 5 Conclusion

In this paper, we present TTARAG, a test-time adaptation approach for retrieval-augmented generation that enables dynamic model optimization during inference. Our method introduces a simple yet effective self-supervised learning objective where the model learns to predict retrieved content, allowing automatic parameter adjustment to target domains without requiring labeled data. Through extensive experiments across six specialized domain, we demonstrate that TTARAG achieves consistent improvements over the naive-rag system. These findings suggest that test-time adaptation is a promising direction for improving RAG systems’ performance in specialized domains.


## References

- [1]
  Gautier Izacard and Edouard Grave,
  
  
  “Leveraging passage retrieval with generative models for open domain question answering,”
  
  
  in EACL 2021, Online, 2021, pp. 874–880, Association for Computational Linguistics.
- [2]
  Patrick S. H. Lewis, Ethan Perez, et al.,
  
  
  “Retrieval-augmented generation for knowledge-intensive NLP tasks,”
  
  
  in NeurIPS 2020, Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, Eds., 2020.
- [3]
  Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson,
  
  
  “From local to global: A graph rag approach to query-focused summarization,”
  
  
  arXiv preprint arXiv:2404.16130, 2024.
- [4]
  Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al.,
  
  
  “Language models are unsupervised multitask learners,”
  
  
  OpenAI blog, vol. 1, no. 8, pp. 9, 2019.
- [5]
  Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.,
  
  
  “Language models are few-shot learners,”
  
  
  Advances in neural information processing systems, vol. 33, pp. 1877–1901, 2020.
- [6]
  Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al.,
  
  
  “Sparks of artificial general intelligence: Early experiments with gpt-4,”
  
  
  arXiv preprint arXiv:2303.12712, 2023.
- [7]
  Marius Pasca,
  
  
  “Wikipedia as a resource for text analysis and retrieval,”
  
  
  in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: Tutorial Abstracts, Preslav Nakov and Alexis Palmer, Eds., Florence, Italy, July 2019, p. 24, Association for Computational Linguistics.
- [8]
  Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor,
  
  
  “Freebase: a collaboratively created graph database for structuring human knowledge,”
  
  
  in Proceedings of the 2008 ACM SIGMOD international conference on Management of data, 2008, pp. 1247–1250.
- [9]
  Qiao Jin, Bhuwan Dhingra, Zhengping Liu, William Cohen, and Xinghua Lu,
  
  
  “PubMedQA: A dataset for biomedical research question answering,”
  
  
  in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Hong Kong, China, 2019, pp. 2567–2577, Association for Computational Linguistics.
- [10]
  Ran Xu, Hui Liu, Sreyashi Nag, Zhenwei Dai, Yaochen Xie, Xianfeng Tang, Chen Luo, Yang Li, Joyce C. Ho, Carl Yang, and Qi He,
  
  
  “Simrag: Self-improving retrieval-augmented generation for adapting large language models to specialized domains,” 2025.
- [11]
  Wenqi Shi, Ran Xu, Yuchen Zhuang, Yue Yu, Hang Wu, Carl Yang, and May D Wang,
  
  
  “Medadapter: Efficient test-time adaptation of large language models towards medical reasoning,”
  
  
  in EMNLP, 2024.
- [12]
  John Miller, Karl Krauth, Benjamin Recht, and Ludwig Schmidt,
  
  
  “The effect of natural distribution shift on question answering models,”
  
  
  in Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event. 2020, vol. 119 of Proceedings of Machine Learning Research, pp. 6905–6916, PMLR.
- [13]
  Linqing Liu, Patrick Lewis, Sebastian Riedel, and Pontus Stenetorp,
  
  
  “Challenges in generalization in open domain question answering,”
  
  
  in Findings of the Association for Computational Linguistics: NAACL 2022, Marine Carpuat, Marie-Catherine de Marneffe, and Ivan Vladimir Meza Ruiz, Eds., Seattle, United States, July 2022, pp. 2014–2029, Association for Computational Linguistics.
- [14]
  Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt,
  
  
  “Test-time training with self-supervision for generalization under distribution shifts,”
  
  
  in International conference on machine learning. PMLR, 2020, pp. 9229–9248.
- [15]
  Adilbek Karmanov, Dayan Guan, Shijian Lu, Abdulmotaleb El Saddik, and Eric Xing,
  
  
  “Efficient test-time adaptation of vision-language models,” 2024.
- [16]
  Jian Liang, Ran He, and Tieniu Tan,
  
  
  “A comprehensive survey on test-time adaptation under distribution shifts,”
  
  
  International Journal of Computer Vision, pp. 1–34, 2024.
- [17]
  Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig,
  
  
  “Active retrieval augmented generation,”
  
  
  in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Houda Bouamor, Juan Pino, and Kalika Bali, Eds., Singapore, Dec. 2023, pp. 7969–7992, Association for Computational Linguistics.
- [18]
  Soyeong Jeong, Jinheon Baek, and et al.,
  
  
  “Adaptive-RAG: Learning to adapt retrieval-augmented large language models through question complexity,”
  
  
  in NAACL 2024, Kevin Duh, Helena Gomez, and Steven Bethard, Eds., Mexico City, Mexico, June 2024, pp. 7036–7050, Association for Computational Linguistics.
- [19]
  Ori Yoran, Tomer Wolfson, Ori Ram, and Jonathan Berant,
  
  
  “Making retrieval-augmented language models robust to irrelevant context,”
  
  
  in ICLR, 2024.
- [20]
  Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi, and Bryan Catanzaro,
  
  
  “Rankrag: Unifying context ranking with retrieval-augmented generation in llms,”
  
  
  in NeurIPS, 2024.
- [21]
  Zihan Liu, Wei Ping, Rajarshi Roy, Peng Xu, Mohammad Shoeybi, and Bryan Catanzaro,
  
  
  “Chatqa: Surpassing gpt-4 on conversational qa and rag,”
  
  
  in NeurIPS, 2024.
- [22]
  Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Richard James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettlemoyer, and Wen tau Yih,
  
  
  “RA-DIT: Retrieval-augmented dual instruction tuning,”
  
  
  in ICLR, 2024.
- [23]
  Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell,
  
  
  “Tent: Fully test-time adaptation by entropy minimization,”
  
  
  in International Conference on Learning Representations, 2021.
- [24]
  Malik Boudiaf, Romain Mueller, Ismail Ben Ayed, and Luca Bertinetto,
  
  
  “Parameter-free online test-time adaptation,”
  
  
  in IEEE Conference on Computer Vision and Pattern Recognition, 2022, pp. 8344–8353.
- [25]
  Jian Liang, Dapeng Hu, and Jiashi Feng,
  
  
  “Do we really need to access the source data? source hypothesis transfer for unsupervised domain adaptation,”
  
  
  in International Conference on Machine Learning (ICML), 2020, pp. 6028–6039.
- [26]
  I. Kuzborskij and F. Orabona,
  
  
  “Stability and Hypothesis Transfer Learning,”
  
  
  in International Conference on Machine Learning (ICML), 2013, pp. 942–950.
- [27]
  Steffen Schneider, Evgenia Rusak, Luisa Eck, Oliver Bringmann, Wieland Brendel, and Matthias Bethge,
  
  
  “Improving robustness against common corruptions by covariate shift adaptation,”
  
  
  in Advances in Neural Information Processing Systems, 2020, vol. 33, pp. 11539–11551.
- [28]
  Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary, Rongze Daniel Gui, Ziran Will Jiang, Ziyu Jiang, et al.,
  
  
  “Crag–comprehensive rag benchmark,”
  
  
  arXiv preprint arXiv:2406.04744, 2024.
- [29]
  George Tsatsaronis, Georgios Balikas, Prodromos Malakasiotis, Ioannis Partalas, Matthias Zschunke, Michael R Alvers, Dirk Weissenborn, Anastasia Krithara, Sergios Petridis, Dimitris Polychronopoulos, et al.,
  
  
  “An overview of the bioasq large-scale biomedical semantic indexing and question answering competition,”
  
  
  BMC bioinformatics, 2015.
- [30]
  Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.,
  
  
  “Chain-of-thought prompting elicits reasoning in large language models,”
  
  
  Advances in neural information processing systems, vol. 35, pp. 24824–24837, 2022.
- [31]
  Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang, Xiaojun Chen, and Ruifeng Xu,
  
  
  “Enhancing noise robustness of retrieval-augmented language models with adaptive adversarial training,”
  
  
  arXiv preprint arXiv:2405.20978, 2024.
- [32]
  Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi,
  
  
  “Self-rag: Learning to retrieve, generate, and critique through self-reflection,”
  
  
  arXiv preprint arXiv:2310.11511, 2023.

