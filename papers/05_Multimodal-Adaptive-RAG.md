# Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning


# Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning

Ruoshuang Du

  
Xin Sun

  
Qiang Liu

  
Bowen Song

  
Zhongqi Chen

  
Weiqiang Wang

  
Liang Wang


###### Abstract

Visual Question Answering systems face reliability issues due to hallucinations, where models generate answers misaligned with visual input or factual knowledge.
While Retrieval Augmented Generation frameworks mitigate this issue by incorporating external knowledge, static retrieval often introduces irrelevant or conflicting content, particularly in visual RAG settings where visually similar but semantically incorrect evidence may be retrieved.
To address this, we propose Multimodal Adaptive RAG (MMA-RAG), which dynamically assesses the confidence in the internal knowledge of the model to decide whether to incorporate the retrieved external information into the generation process.
Central to MMA-RAG is a decision classifier trained through a layer-wise analysis, which leverages joint internal visual and textual representations to guide the use of reverse image retrieval.
Experiments demonstrated that the model achieves a significant improvement in response performance in three VQA datasets.
Meanwhile, ablation studies highlighted the importance of internal representations in adaptive retrieval decisions.
In general, the experimental results demonstrated that MMA-RAG effectively balances external knowledge utilization and inference robustness in diverse multimodal scenarios.
We make all code and data publicly available at [github](https://anonymous.4open.science/r/Multimodal-Adaptive-RAG-20AB/).


## I Introduction

Large language models (LLMs) have achieved remarkable success in a wide range of natural language understanding and generation tasks [[25](#bib.bib48 "Survey of different large language model architectures: trends, benchmarks, and challenges")].
However, despite their impressive capabilities, LLMs are known to suffer from a critical limitation: hallucination [[15](#bib.bib3 "HIM-rag: a heuristic framework for iterative multi-source retrieval-augmented generation"), [8](#bib.bib49 "A survey on hallucination in large language models: principles, taxonomy, challenges, and open questions"), [22](#bib.bib21 "Mitigating dialogue hallucination for large vision language models via adversarial instruction tuning")].
This phenomenon refers to the generation of outputs that are factually inaccurate, unverifiable, or inconsistent with the provided input.


To address this limitation, Retrieval-Augmented Generation (RAG) has emerged as a promising solution [[9](#bib.bib14 "A survey on retrieval-augmented text generation for large language models"), [5](#bib.bib10 "Retrieval-augmented generation for large language models: a survey")].
RAG improves language model performance by incorporating external knowledge retrieved from large-scale textual corpora to complement the model’s internal parameterized representations [[14](#bib.bib12 "Retrieval-augmented generation for knowledge-intensive nlp tasks"), [31](#bib.bib13 "Retrieval-augmented generation for natural language processing: a survey"), [23](#bib.bib4 "Think before retrieving: query-response relevance retrieval augmented generation")].
This approach has been shown to improve response accuracy and factual consistency, particularly in knowledge-intensive tasks where reliance on static, pre-trained parameters alone may lead to outdated or incorrect information [[37](#bib.bib11 "Retrieval-augmented generation for ai-generated content: a survey")].


Although early implementations were confined to purely textual settings, recent work has extended this idea to multimodal contexts, leading to multimodal RAG frameworks [[38](#bib.bib5 "Retrieving multimodal information for augmented generation: a survey"), [34](#bib.bib7 "Retrieval-augmented multimodal language modeling"), [35](#bib.bib8 "MRAMG-bench: a beyondtext benchmark for multimodal retrieval-augmented multimodal generation"), [7](#bib.bib9 "Reveal: retrieval-augmented visual-language pre-training with multi-source multimodal knowledge memory"), [36](#bib.bib6 "SAM-rag: an self-adaptive framework for multimodal retrieval-augmented generation")].
In multimodal RAG, models generate image-question conditioned retrieval queries, sourcing both text and images from external databases to support reasoning.
These systems have shown promise on complex tasks such as Visual Question Answering (VQA) [[1](#bib.bib2 "Vqa: visual question answering")].
For example, REVIVE uses regional visual representations combined with the knowledge retrieved to improve the accuracy of the answers [[17](#bib.bib20 "Revive: regional visual representation matters in knowledge-based visual question answering")].
Similarly, MuRAG improves response generation by incorporating information from external knowledge bases, enabling more effective joint reasoning over images and texts [[3](#bib.bib26 "Murag: multimodal retrieval-augmented generator for open question answering over images and text")].
The multimodal alignment model introduced a multimodal large language model-based reclassification step, selecting the most relevant knowledge from the top candidates retrieved to improve performance [[2](#bib.bib25 "Seeing beyond: enhancing visual question answering with multi-modal retrieval")].


Reverse Image Retrieval (RIR) can be viewed as a specialized variant of multimodal RAG, in which additional multimodal context is provided by retrieving visually similar images from web-based sources [[32](#bib.bib19 "Reverse image retrieval cues parametric memory in multimodal llms")].
In this approach, screenshots or related images retrieved based on the query image are integrated with the original input, thereby enriching the model’s input space with complementary visual information.
Importantly, the advantage of RIR does not typically lie in directly supplying the correct answer, but rather in facilitating more accurate grounding and interpretation of the query through contextually relevant visual cues.


Despite the aforementioned advantages, visual retrieval augmented generation introduces a more challenging failure mode than its text-only counterpart.
Images returned by visual retrieval are often highly similar in appearance, yet semantically inconsistent with the query.
As illustrated in Fig. [1](#S1.F1 "Figure 1 ‣ I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning"), queries concerning plants from the Lamiaceae family may retrieve visually similar species such as Horehound, resulting in evidence that appears highly convincing but is in fact incorrect.
This phenomenon, referred to as *visual similarity with semantic mismatch*, is more difficult to defend against than interference in text-based RAG systems.
Consequently, effective mitigation requires joint reasoning over visual and textual features, enabling the model to simultaneously assess the visual similarity of retrieved content and its semantic consistency with the query.


Similar to RIR, many existing multimodal RAG methods implicitly assume that external information is always beneficial, often introducing irrelevant or misleading evidence—particularly in cases where the model already possesses sufficient internal knowledge—thereby causing retrieval redundancy that is likely to degrade overall model performance [[16](#bib.bib15 "Benchmarking multimodal retrieval augmented generation with dynamic vqa dataset and self-adaptive planning agent"), [10](#bib.bib16 "Active retrieval augmented generation"), [19](#bib.bib17 "A survey of multimodal retrieval-augmented generation")].


![Refer to caption](05_Multimodal-Adaptive-RAG_images/x1.png)

Figure 1: Overview of the Multimodal Adaptive RAG framework


In this paper, we propose Multimodal Adaptive Retrieval Augmented Generation (MMA-RAG) designed to address the challenges of avoiding harmful factors caused by visually similar but semantically incorrect evidence and the rational use of retrieved external information.
Specifically, MMA-RAG first extracts the hidden states of textual and visual features, aligns them, and integrates them into a unified vision–language joint representation.
Based on this representation, we train a four-class classifier to predict the impact of retrieval on correctness of the answer.
Crucially, we argue that simple last-layer representations may not fully capture the nuanced misalignment between visual and textual modalities.
Through a comprehensive layer-wise analysis of the model’s internal states, we observe that the semantic alignment between vision and text evolves differently across network depths.
Text-only features show limited discriminative capability in shallow layers and become effective only in deeper layers, whereas multimodal features achieve high detection accuracy even in the early layers.
This observation indicates that multimodal fusion is crucial for the early identification of erroneous or misleading evidence.
Motivated by this finding, MMA-RAG strategically selects and fuses the most informative intermediate representations to train a multimodal joint-feature classifier.
Finally, guided by the classifier’s predictions, the model adaptively employs the reverse image retrieval mechanism, avoiding the introduction of harmful external information while ensuring the generation of correct answers whenever possible.
The core contributions of this paper are summarized as follows:

- •
  
  We propose MMA-RAG, a multimodal adaptive retrieval augmented generation framework that predicts the utility of RIR from internal multimodal representations to mitigate harmful retrieval in visual question answering tasks.
- •
  
  We perform a layer-wise analysis of multimodal large language models, revealing how visual and textual confidence signals evolve and informing the selection of internal features for hallucination detection.
- •
  
  We design an internal-representation-based retrieval utility classifier that integrates multimodal features to assess whether external retrieval improves response correctness.
- •
  
  Extensive experiments on three knowledge-intensive VQA benchmarks with multiple vision–language backbone models demonstrate that MMA-RAG outperforms standard retrieval-based methods and existing baselines.


## II Methodology

Although RIR can provide valuable external context by retrieving visually similar images, it can also introduce harmful samples that mislead the model, especially when the retrieved images contain semantically irrelevant or contradictory content.
In such cases, relying on external retrieval may cause the model to generate incorrect answers, even when the original image alone would have sufficed for correct reasoning.
As illustrated in the example depicted in Fig. [1](#S1.F1 "Figure 1 ‣ I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning"), the use of RIR yields the answer ”The plant is in the horehound family”, which is in fact incorrect.
The accurate response corresponds to the case without RIR, namely, ”The plant family of this image is the Lamiaceae family, also known as the mint family”.


To address such issues in VQA tasks, we propose a Multi-modal Adaptive Retrieval-Augmented Generation (MMA-RAG) framework to address the challenge of harmful retrievals in VQA tasks.
The core idea is to adaptively determine whether externally retrieved images should be incorporated into the generation process to minimize the negative impact of irrelevant or misleading visual information.
If the retrieved images are helpful to improve the accuracy of the answers, the MMA-RAG framework adopts the RAG approach, incorporating all images and the corresponding question as input to the generation model.
If the retrieved images introduce noise and degrade answer quality, the framework relies solely on the original image and the question for answer generation.


MMA-RAG is designed to make adaptive use of retrieved visual content in a multimodal VQA setting.
It consists of three key components:


1) Reverse Image Retrieval:
For each VQA instance, the input consists of a query QQ and an image I1I\_{1}.
We perform reverse image retrieval using I1I\_{1} by querying visually similar images on Google and capturing screenshots of the retrieved results.
These screenshots serve as an additional input image I2I\_{2}, which may subsequently be fed into the large model along with the original question QQ and image I1I\_{1}.


2) Abstract feature:
In purely text-based large language models, the hidden states of the exact answer token can serve as key tokens for error detection[[21](#bib.bib41 "Llms know more than they show: on the intrinsic representation of llm hallucinations")].
We extend this observation to multimodal large language models and perform a layer-wise analysis, showing that multimodal fusion leads to more accurate error detection in multimodal settings.
Specifically, we evaluate error-detection capability across different layers using the Idefics2-8B backbone on the OK-VQA dataset. Hidden states are extracted from every even-numbered Transformer layer as well as the final layer.
We compare multiple feature configurations: textual hidden states at key token positions alone, and textual states combined with vision features using different pooling methods.
As shown in Fig. [2](#S2.F2 "Figure 2 ‣ II Methodology ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning"), the results reveal several key insights into the internal decision-making process of the model.


Primacy of Multimodal Fusion.
Error-detection methods that integrate visual features consistently outperform those based solely on textual hidden states across all layers.
This observation indicates that visual representations play a critical role in assessing the internal certainty of the model and in deciding whether external retrieval is required.


Layer-wise Evolution of Information.
Text-only features exhibit limited predictive capability in shallow layers ranging from layer 0 to layer 10, but their performance improves substantially in deeper layers. In contrast, multimodal features reach high accuracy much earlier, particularly within the intermediate layers from layer 2 to layer 16. This trend suggests that the alignment between visual and textual cues becomes sufficiently established at mid-network depths to effectively support retrieval gating.


Limited Sensitivity to Pooling Strategies. Both average and maximum pooling over visual features result in similar high-accuracy regions for error detection. This suggests that, when visual information is globally aggregated, the detector mainly relies on image-level semantics, while the specific pooling operator has only a minor impact.


![Refer to caption](05_Multimodal-Adaptive-RAG_images/x2.png)

Figure 2: Heatmap of the classifier accuracy across different key tokens and layers on the OK-VQA dataset using the Idefics2-8B model


Therefore, we leverage the model’s internal textual and visual hidden states to train the adaptive classifier, as these representations preserve rich semantic and contextual information.
Although key tokens serve as effective diagnostic signals that represent posterior knowledge in retrospective analysis, they are inaccessible during the inference phase of an adaptive system. Consequently, for the textual feature T1T\_{1}, we utilize the hidden state from the final decoding step.
This state synthesizes information from the question, the image, and the generated prefix, thereby naturally reflecting the model’s current belief state.
For the image feature, after feeding the input image into the model, multiple layers of visual representations are obtained.
We select a specific intermediate layer and compute the average pooling over all patch embeddings within that layer.
The resulting vector serves as the compact and informative vision feature V1V\_{1} that represents the input image.
We jointly input QQ, I1I\_{1}, and I2I\_{2} into the model, following a similar procedure as described above, to extract the corresponding textual characteristic T2T\_{2} and vision feature V2V\_{2}.


To effectively capture both semantic and visual cues, we concatenate the generated feature T1T\_{1}, V1V\_{1}, T2T\_{2}, V2V\_{2} to form a unified representation HcH\_{c} for classification.

|  | Hc=Concat​(T1,V1,T2,V2){H}\_{c}=\text{Concat}(T\_{1},V\_{1},T\_{2},V\_{2}) |  |
| --- | --- | --- |


3) Adaptive detect:
The extracted data HcH\_{c} are utilized to train a four-class classifier.
Considering both performance and efficiency, we adopt a multilayer perceptron architecture for the classifier, which is employed to evaluate the retrieval utility under different conditions as a retrieval gating mechanism.

|  | y^=arg⁡maxy⁡Softmax⁡(f​(Hc))\hat{y}=\arg\max\_{y}\operatorname{Softmax}(f(H\_{c})) |  |
| --- | --- | --- |

where y^\hat{y} represents the classification result.
Specifically, there are four possible scenarios:
1) S1S\_{1}: Both using external retrieval and not using external retrieval result in an incorrect answer.
2) S2S\_{2}: Using external retrieval leads to a correct answer, while not using it results in an incorrect answer.
3) S3S\_{3}: Using external retrieval results in an incorrect answer, while not using it leads to a correct answer.
4) S4S\_{4}: Both using external retrieval and not using external retrieval result in the correct answer.


At inference time, we introduce two alternative retrieval trigger strategies based on the classifier’s four-way prediction.
These strategies represent two opposing preferences regarding the reliance on external retrieval.


RIR-Pessimistic Strategy. 
This pessimistic-oriented strategy adopts a cautious stance toward external retrieval.
Retrieval is triggered only when it is predicted to be essential, i.e. when using retrieved images leads to a correct answer, and not using them would result in an incorrect one.
In all other scenarios, the model discards the retrieved content and relies solely on the original image and the question.
This strategy minimizes the risk of introducing harmful noise, favoring the default path unless a clear benefit is identified.

|  | R={1 if ​y^=S20 if ​y^≠S2R=\begin{cases}1&\text{ if }\hat{y}=S\_{2}\\ 0&\text{ if }\hat{y}\neq S\_{2}\end{cases} |  |
| --- | --- | --- |

where R decides whether to use RIR.


RIR-Optimistic Strategy. 
This optimistic oriented strategy takes a more liberal approach to retrieval usage, reflecting a retrieval-favoring bias.
It triggers external retrieval in all cases except when it is predicted to degrade performance, that is, when the retrieved images introduce noise and hurt the response quality.
In this view, an external visual context is generally helpful and should be included unless explicitly harmful.

|  | R={1 if ​y^≠S30 if ​y^=S3R=\begin{cases}1&\text{ if }\hat{y}\neq S\_{3}\\ 0&\text{ if }\hat{y}=S\_{3}\end{cases} |  |
| --- | --- | --- |

When R is 1, choose to use RIR to generate the answer, that is, input the original image, screenshot image and question together into the model to generate the answer; otherwise, do not use the screenshot image.


The two strategies embody different stances toward external retrieval.
Our subsequent experiments investigate their hierarchical impact across different datasets.
This flexible decision layer enables the system to trade off robustness and completeness while adapting to the specific characteristics of the data or application domain.


## III Experiments

In this section, we evaluate the effectiveness of the multimodal adaptive classifier across multiple datasets.
We use model-judged accuracy as metrics for answer correctness.
We systematically compare the performance of four configurations: zero-shot prompt, few-shot prompt, few-shot prompt with RIR, and our proposed method MMA-RAG, which adaptively determines whether to apply RIR based on a classifier under the few-shot setting.


### III-A Datasets and Knowledge Bases

Visual question answering tasks that require external knowledge integration face unique challenges, as answers depend on information beyond image content.
Foundational benchmarks such as OK-VQA [[18](#bib.bib27 "Ok-vqa: a visual question answering benchmark requiring external knowledge")] exemplify this paradigm, featuring more than 14000 questions that demand reasoning with common sense and domain-specific knowledge.
In this dataset, there have already been many excellent studies.
Methods for extracting relevant knowledge from noisy sources [[30](#bib.bib28 "Multi-modal answer validation for knowledge-based vqa")], cross-modal retrieval frameworks [[3](#bib.bib26 "Murag: multimodal retrieval-augmented generator for open question answering over images and text")], and hybrid graph representations [[39](#bib.bib29 "Mucko: multi-layer cross-modal knowledge reasoning for fact-based visual question answering")] demonstrate improved performance in knowledge-intensive VQA tasks, highlighting the necessity of unified visual-textual knowledge architectures.


To address the limitations of existing datasets in the evaluation of deep visual knowledge integration, Encyclopedic-VQA introduces a comprehensive benchmark featuring 221K unique question-answer pairs, each linked to up to 5 images [[20](#bib.bib30 "Encyclopedic vqa: visual questions about detailed properties of fine-grained categories")].
These images are derived from iNaturalist 2021 [[26](#bib.bib31 "Benchmarking representation learning for natural world image collections")] and Google Landmarks Dataset V2 [[29](#bib.bib32 "Google landmarks dataset v2-a large-scale benchmark for instance-level recognition and retrieval")].
Unlike previous VQA tasks that focus on understanding a generic scene, this dataset emphasizes fine-grained instance-level attributes that require encyclopedic knowledge of specific entities, artifacts, or biological species.
Encyclopedic-VQA has become a standard testbed for evaluating knowledge-intensive vision systems.
Recent works like EchoSight [[33](#bib.bib33 "Echosight: advancing visual-language models with wiki knowledge")] achieve a more effective fusion of multimodal knowledge, validating the practicality of the Encyclopedic VQA data set to advance the synthesis of multimodal knowledge.


InfoSeek [[4](#bib.bib34 "Can pre-trained vision and language models answer visual information-seeking questions?")] establishes a large-scale benchmark for visual information search queries, consisting of approximately 1.3 million questions grounded in over 11000 visual entities derived from the Open Visual Entity Nexus dataset [[6](#bib.bib35 "Open-domain visual entity recognition: towards recognizing millions of wikipedia entities")].
These entities span diverse domains, including cultural landmarks, rare species, and technological artifacts, ensuring broad coverage of knowledge-intensive topics.
The dataset combines 8.9K human-annotated QA pairs with 1.3M automatically generated questions, using templated parsing of Wikipedia infoboxes and entity-attribute relationships.


Our dataset is randomly sampled from the aforementioned public datasets, followed by obtaining screenshots to prepare the data for training and evaluation.
The numbers of training and evaluation samples for the three datasets are summarized in Table 1.


TABLE I: Training and evaluation sample sizes of the three datasets.

| Dataset | Infoseek | OK-VQA | E-VQA |
| --- | --- | --- | --- |
| Training | 3740 | 1000 | 1000 |
| Evaluation | 1646 | 4989 | 3345 |


### III-B Backbone Models and Metrics

Idefics2-8B is an open source vision language model introduced by Hugging Face, featuring 8 billion parameters [[13](#bib.bib36 "What matters when building vision-language models?")].
This model can process arbitrary sequences of text and image input to generate textual outputs.
In multiple visual question answering benchmarks, Idefics-2 ranks among the top models of similar scale, with performance comparable to larger models.


Idefics3-8B is a vision language model developed by the research team at Hugging Face, designed to process image and text input while generating textual output [[12](#bib.bib37 "Building and better understanding vision-language models: insights and future directions")].
Architecturally, Idefics3-8B uses a cross-attention mechanism, which allows an effective integration of visual and linguistic information, thus achieving strong performance in multimodal tasks.


The Qwen2-VL series introduces several key advancements in vision language models to enhance the model’s ability to process images of varying resolutions [[27](#bib.bib38 "Qwen2-vl: enhancing vision-language model’s perception of the world at any resolution")].
Qwen2.5-VL extends these capabilities by introducing a unified vision framework for both images and videos, multimodal rotary position embedding (M-RoPE) for improved cross-modal alignment, and a Naive Dynamic Resolution mechanism for adaptive visual tokenization based on image resolution.


We employ Qwen2.5-Instruct as an automatic evaluator to assess the correctness of the generated responses, based on which we compute the accuracy for each dataset.


### III-C Baselines

RIR generates responses by incorporating reverse image retrieval [[32](#bib.bib19 "Reverse image retrieval cues parametric memory in multimodal llms")].
Building upon RIR, we further consider several baselines that make adaptive decisions about whether to use RIR.
Chain-of-Thought (CoT) encourages the model to produce explicit intermediate reasoning steps [[28](#bib.bib45 "Chain-of-thought prompting elicits reasoning in large language models")].
P(true) serves as a standard confidence-based baseline, which estimates correctness by calculating the probability that the model affirms its own answer [[11](#bib.bib22 "Language models (mostly) know what they know")].
CLIP measures image–text semantic alignment using joint embeddings [[24](#bib.bib23 "Learning transferable visual models from natural language supervision")].


## IV Results

### IV-A Main Results

We trained the classifier for Multimodal Adaptive Retrieval Augmented Generation using visual features and textual hidden states and conducted experiments on three datasets: InfoSeek, OK-VQA, and Encyclopedic-VQA.
Given the varying class distributions across datasets, we apply dataset-specific class weights to balance the contribution of different classes during training, thereby ensuring the reliability and validity of the experiments.
The retrieval process involved obtaining similar images through Google search, capturing screenshots and using these screenshots alongside the original images and questions as input to generate final answers.
The classifier’s role was to assess the utility of such retrievals; if beneficial for generating correct answers, external retrieval was employed; otherwise, it was omitted.
During the testing process, Qwen2.5-instruct was used to evaluate whether the answers were correct and the final evaluation metric was accuracy.
Tab. [II](#S4.T2 "TABLE II ‣ IV-A Main Results ‣ IV Results ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning") shows the main experimental results.


TABLE II: Accuracy scores on the InfoSeek, E-VQA, and OK-VQA test datasets, using Idefics2-8B, Idefics3-8B, and Qwen2VL-7B as backbones, with Qwen25\_Instruct used to judge answer correctness. Bold indicates state-of-the-art performance

| Model | Method | InfoSeek | OK-VQA | E-VQA |
| --- | --- | --- | --- | --- |
|  | Zero shot | 19.9 | 58.7 | 10.0 |
|  | Few shot | 15.9 | 58.5 | 14.1 |
|  | RIR | 23.3 | 62.2 | 19.8 |
| Qwen2VL | CoT | 20.4 | 46.7 | 14.4 |
|  | P(true) | 22.6 | 53.3 | 19.1 |
|  | CLIP | 20.9 | 56.0 | 18.5 |
|  | MMA-RAG | 23.9 | 62.4 | 20.0 |
|  | Zero shot | 15.1 | 53.8 | 13.0 |
|  | Few shot | 14.2 | 58.7 | 12.7 |
|  | RIR | 17.2 | 56.7 | 14.1 |
| Idefics2 | CoT | 15.5 | 49.0 | 14.1 |
|  | P(true) | 14.1 | 49.1 | 13.8 |
|  | CLIP | 16.4 | 54.8 | 14.3 |
|  | MMA-RAG | 20.3 | 60.1 | 14.6 |
|  | Zero shot | 12.5 | 43.2 | 11.9 |
|  | Few shot | 8.5 | 46.0 | 6.3 |
|  | RIR | 17.5 | 56.6 | 12.5 |
| Idefics3 | CoT | 14.3 | 43.7 | 10.9 |
|  | P(true) | 12.9 | 43.7 | 11.0 |
|  | CLIP | 12.6 | 50.0 | 10.0 |
|  | MMA-RAG | 18.1 | 58.3 | 12.6 |


Across different models and datasets, the performance of zero-shot and few-shot prompts is inconsistent, primarily due to variations in dataset types and task characteristics.
To ensure experimental validity, we adopted the few-shot prompt in both the RIR and MMA-RAG experiments.
Reverse Image Retrieval systems that use Google to retrieve similar images have significantly improved the accuracy of the generated answers.
However, in certain cases, the use of retrieved external images leads to a degradation in response quality, producing incorrect responses that would not have occurred if the model had relied solely on the original image and question.
We refer to such instances as ”harmful samples”, where the introduction of additional visual information inadvertently misguides the model.
Compared with the original Reverse Image Retrieval model, the multimodal adaptive RAG model achieves varying degrees of improvement in three datasets.
Compared with the confidence-based P(true) baseline, the auxiliary-model-based CLIP approach, and reasoning-based CoT method, MMA-RAG consistently delivers substantial and statistically significant performance gains across all evaluated datasets.
This improvement is primarily attributed to the model’s ability to predict and suppress the influence of harmful samples, thus avoiding unnecessary or detrimental retrievals and preserving the model’s ability to generate correct answers.


For certain model configurations on OK-VQA and E-VQA, MMA-RAG yields only marginal improvements, likely because the proportion of harmful samples introduced by RIR is low, leaving limited room for denoising.
This also indicates that MMA-RAG is relatively safe: when retrieval is beneficial, the gating mechanism tends to keep retrieval enabled and is unlikely to introduce adverse effects.


### IV-B Feature Robustness

We further conducted exploratory experiments on the extraction of visual features, including selecting patch embeddings from different Transformer layers and applying various pooling strategies, such as average pooling and max pooling to obtain global image representations.


![Refer to caption](05_Multimodal-Adaptive-RAG_images/vision_features_of_different_layers.png)

Figure 3: Impact of Transformer Layer Selection on Visual Feature Representation 


As illustrated in Fig. [3](#S4.F3 "Figure 3 ‣ IV-B Feature Robustness ‣ IV Results ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning"), the vertical axis represents the response accuracy and the horizontal axis corresponds to the layer index.
Despite differences in feature extraction strategies, overall performance variation remained relatively small.
This limited variance can be attributed to the robustness of the downstream classifier to feature extraction strategies.
Since the middle and later layers of vision transformers already produce stable semantic representations, and pooling differences are largely smoothed out by the subsequent MLP, the overall performance is less sensitive to such variations.


### IV-C Ablation Study

The consistent improvements observed across the datasets can be attributed to the effective exploitation of internal hidden states by the multimodal adaptive RAG model.
In particular, both the textual hidden states, which encapsulate the semantic representation and contextual reasoning derived from the question and the generated response, and the visual features, which encode salient spatial and perceptual information from the input image, play a pivotal role in informing adaptive retrieval decisions.
To better understand the individual and combined contributions of these features, we conducted a series of ablation studies, analyzing their respective impacts on the classifier’s performance within the MMA-RAG framework.


In this experiment, we trained the classifier on three datasets using only hidden textual states or only visual features, while keeping all other settings unchanged.


TABLE III: Performance of ablation study on Multimodal adaptive RAG with idefics2-8B. The performance of the model is jointly influenced by textual and visual features.

| Method | Infoseek | OK-VQA | E-VQA |
| --- | --- | --- | --- |
| RIR | 17.2 | 56.7 | 14.1 |
| wo-text | 17.9 | 58.6 | 14.2 |
| wo-vision | 19.3 | 59.9 | 14.2 |
| MM Adaptive RAG | 20.3 | 60.1 | 14.6 |


As shown in the results of Tab. [III](#S4.T3 "TABLE III ‣ IV-C Ablation Study ‣ IV Results ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning") , given the limited accuracy of large models in generating answers, classifiers that incorporate visual features outperform those relying solely on hidden text states.
This suggests that in VQA tasks, extracting and utilizing visual features may enhance the ability to determine the effectiveness of external retrievals.
Similarly, classifiers that lack textual hidden states have also exhibited a decline in performance.
This has demonstrated that hidden textual states inherently contain implicit cues regarding the accuracy of the generated answers.
Therefore, by effectively integrating hidden text states with visual features, we were able to extract maximally valuable information, thus enhancing our understanding of the model decision-making process.


![Refer to caption](05_Multimodal-Adaptive-RAG_images/x3.png)

Figure 4: Layer-wise Comparison of Classifier Accuracy Using Textual and Multimodal Features on the InfoSeek, OK-VQA, and E-VQA Datasets


Additionally, we conduct a layer-wise comparative study using the Idefics2-8B model to compare classifiers that rely solely on textual features or visual features with those that jointly incorporate textual and visual features.
As shown in Fig. [4](#S4.F4 "Figure 4 ‣ IV-C Ablation Study ‣ IV Results ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning") and Fig. [5](#S4.F5 "Figure 5 ‣ IV-C Ablation Study ‣ IV Results ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning"), the vertical axis represents the classifier accuracy, while the horizontal axis corresponds to different network layers.
The combined use of textual and visual features consistently leads to improved accuracy across datasets, and both modalities are indispensable across all layers. These results further validate the effectiveness of our approach.


![Refer to caption](05_Multimodal-Adaptive-RAG_images/x4.png)

Figure 5: Layer-wise Comparison of Classifier Accuracy with Textual and Visual Features on OK-VQA Dataset


### IV-D RIR Strategy Comparison

To further examine the impact of retrieval trigger policies, we conduct a comparative study between RIR-Optimistic and RIR-Pessimistic strategies.
Experiments are performed with Idefics2-8B and Idefics3-8B on three datasets: InfoSeek, OK-VQA, and E-VQA.
For each dataset, we report the response accuracy under both strategies while keeping all other settings fixed.


![Refer to caption](05_Multimodal-Adaptive-RAG_images/x5.png)

Figure 6: Performance Comparison between RIR-Optimistic and RIR-Pessimistic Strategies


As shown in Fig. [6](#S4.F6 "Figure 6 ‣ IV-D RIR Strategy Comparison ‣ IV Results ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning"), a clear dataset-dependent preference emerges between the two retrieval strategies, where the vertical axis denotes the response accuracy and the horizontal axis corresponds to the layer index.
On OK-VQA, the RIR-Pessimistic Strategy consistently outperforms the RIR-Optimistic Strategy, whereas on InfoSeek and E-VQA the opposite trend is observed.


This divergence can be attributed to the differing roles that external visual information plays across datasets. OK-VQA primarily focuses on common-sense reasoning and world knowledge, where the answer often depends less on fine-grained visual cues and more on abstract or factual understanding. In such cases, reverse image retrieval is prone to introducing visually similar yet semantically irrelevant evidence, making a pessimistic, retrieval-averse strategy more robust. By contrast, InfoSeek and E-VQA emphasize instance-level recognition and encyclopedic knowledge, where additional visual context retrieved from external sources can provide complementary cues that help disambiguate entities or attributes, thereby improving answer accuracy.
Consistent trends across the backbone indicate that strategy preference is driven by dataset characteristics, highlighting the need for adaptive retrieval policies.


## V Conclusion

In this paper, we propose MMA-RAG, a multimodal adaptive retrieval-augmented generation framework that regulates external retrieval based on internal visual and textual representations of multimodal large language models.
By predicting the utility of reverse image retrieval, MMA-RAG selectively incorporates external visual evidence only when it is likely to improve response correctness, thereby mitigating harmful retrieval in visual question answering tasks.
A layer-wise analysis of multimodal internal representations reveals that the evolution of visual and textual confidence signals provides reliable cues to detect misleading evidence, motivating the use of an internal representation-based retrieval utility classifier.
Extensive experiments on multiple knowledge-intensive VQA benchmarks with diverse vision–language backbone models demonstrate that MMA-RAG improves both accuracy and inference robustness over standard retrieval-based approaches and existing baselines.


## VI References


## References

- [1]
  S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh (2015)
  
  Vqa: visual question answering.
  
  In Proceedings of the IEEE international conference on computer vision,
  
   pp. 2425–2433.
  
  Cited by: [§I](#S1.p3.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [2]
  B. Chen, A. Khare, G. Kumar, A. Akula, and P. Narayana (2025)
  
  Seeing beyond: enhancing visual question answering with multi-modal retrieval.
  
  In Proceedings of the 31st International Conference on Computational Linguistics: Industry Track,
  
   pp. 410–421.
  
  Cited by: [§I](#S1.p3.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [3]
  W. Chen, H. Hu, X. Chen, P. Verga, and W. W. Cohen (2022)
  
  Murag: multimodal retrieval-augmented generator for open question answering over images and text.
  
  arXiv preprint arXiv:2210.02928.
  
  Cited by: [§I](#S1.p3.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning"),
  [§III-A](#S3.SS1.p1.1 "III-A Datasets and Knowledge Bases ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [4]
  Y. Chen, H. Hu, Y. Luan, H. Sun, S. Changpinyo, A. Ritter, and M. Chang (2023)
  
  Can pre-trained vision and language models answer visual information-seeking questions?.
  
  arXiv preprint arXiv:2302.11713.
  
  Cited by: [§III-A](#S3.SS1.p3.1 "III-A Datasets and Knowledge Bases ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [5]
  Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, H. Wang, and H. Wang (2023)
  
  Retrieval-augmented generation for large language models: a survey.
  
  arXiv preprint arXiv:2312.10997 2,  pp. 1.
  
  Cited by: [§I](#S1.p2.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [6]
  H. Hu, Y. Luan, Y. Chen, U. Khandelwal, M. Joshi, K. Lee, K. Toutanova, and M. Chang (2023)
  
  Open-domain visual entity recognition: towards recognizing millions of wikipedia entities.
  
  In Proceedings of the IEEE/CVF International Conference on Computer Vision,
  
   pp. 12065–12075.
  
  Cited by: [§III-A](#S3.SS1.p3.1 "III-A Datasets and Knowledge Bases ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [7]
  Z. Hu, A. Iscen, C. Sun, Z. Wang, K. Chang, Y. Sun, C. Schmid, D. A. Ross, and A. Fathi (2023)
  
  Reveal: retrieval-augmented visual-language pre-training with multi-source multimodal knowledge memory.
  
  In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
  
   pp. 23369–23379.
  
  Cited by: [§I](#S1.p3.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [8]
  L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin, et al. (2025)
  
  A survey on hallucination in large language models: principles, taxonomy, challenges, and open questions.
  
  ACM Transactions on Information Systems 43 (2),  pp. 1–55.
  
  Cited by: [§I](#S1.p1.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [9]
  Y. Huang and J. Huang (2024)
  
  A survey on retrieval-augmented text generation for large language models.
  
  arXiv preprint arXiv:2404.10981.
  
  Cited by: [§I](#S1.p2.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [10]
  Z. Jiang, F. F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-Yu, Y. Yang, J. Callan, and G. Neubig (2023)
  
  Active retrieval augmented generation.
  
  In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing,
  
   pp. 7969–7992.
  
  Cited by: [§I](#S1.p6.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [11]
  S. Kadavath, T. Conerly, A. Askell, T. Henighan, D. Drain, E. Perez, N. Schiefer, Z. Hatfield-Dodds, N. DasSarma, E. Tran-Johnson, et al. (2022)
  
  Language models (mostly) know what they know.
  
  arXiv preprint arXiv:2207.05221.
  
  Cited by: [§III-C](#S3.SS3.p1.1 "III-C Baselines ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [12]
  H. Laurençon, A. Marafioti, V. Sanh, and L. Tronchon (2024)
  
  Building and better understanding vision-language models: insights and future directions.
  
  In Workshop on Responsibly Building the Next Generation of Multimodal Foundational Models,
  
  Cited by: [§III-B](#S3.SS2.p2.1 "III-B Backbone Models and Metrics ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [13]
  H. Laurençon, L. Tronchon, M. Cord, and V. Sanh (2024)
  
  What matters when building vision-language models?.
  
  Advances in Neural Information Processing Systems 37,  pp. 87874–87907.
  
  Cited by: [§III-B](#S3.SS2.p1.1 "III-B Backbone Models and Metrics ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [14]
  P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, et al. (2020)
  
  Retrieval-augmented generation for knowledge-intensive nlp tasks.
  
  Advances in neural information processing systems 33,  pp. 9459–9474.
  
  Cited by: [§I](#S1.p2.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [15]
  S. Li, Y. Yu, H. Yang, Y. Zhou, and C. Ji (2025)
  
  HIM-rag: a heuristic framework for iterative multi-source retrieval-augmented generation.
  
  In 2025 International Joint Conference on Neural Networks (IJCNN),
  
   pp. 1–8.
  
  Cited by: [§I](#S1.p1.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [16]
  Y. Li, Y. Li, X. Wang, Y. Jiang, Z. Zhang, X. Zheng, H. Wang, H. Zheng, F. Huang, J. Zhou, et al. (2024)
  
  Benchmarking multimodal retrieval augmented generation with dynamic vqa dataset and self-adaptive planning agent.
  
  arXiv preprint arXiv:2411.02937.
  
  Cited by: [§I](#S1.p6.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [17]
  Y. Lin, Y. Xie, D. Chen, Y. Xu, C. Zhu, and L. Yuan (2022)
  
  Revive: regional visual representation matters in knowledge-based visual question answering.
  
  Advances in neural information processing systems 35,  pp. 10560–10571.
  
  Cited by: [§I](#S1.p3.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [18]
  K. Marino, M. Rastegari, A. Farhadi, and R. Mottaghi (2019)
  
  Ok-vqa: a visual question answering benchmark requiring external knowledge.
  
  In Proceedings of the IEEE/cvf conference on computer vision and pattern recognition,
  
   pp. 3195–3204.
  
  Cited by: [§III-A](#S3.SS1.p1.1 "III-A Datasets and Knowledge Bases ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [19]
  L. Mei, S. Mo, Z. Yang, and C. Chen (2025)
  
  A survey of multimodal retrieval-augmented generation.
  
  arXiv preprint arXiv:2504.08748.
  
  Cited by: [§I](#S1.p6.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [20]
  T. Mensink, J. Uijlings, L. Castrejon, A. Goel, F. Cadar, H. Zhou, F. Sha, A. Araujo, and V. Ferrari (2023)
  
  Encyclopedic vqa: visual questions about detailed properties of fine-grained categories.
  
  In Proceedings of the IEEE/CVF International Conference on Computer Vision,
  
   pp. 3113–3124.
  
  Cited by: [§III-A](#S3.SS1.p2.1 "III-A Datasets and Knowledge Bases ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [21]
  H. Orgad, M. Toker, Z. Gekhman, R. Reichart, I. Szpektor, H. Kotek, and Y. Belinkov (2024)
  
  Llms know more than they show: on the intrinsic representation of llm hallucinations.
  
  arXiv preprint arXiv:2410.02707.
  
  Cited by: [§II](#S2.p5.1 "II Methodology ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [22]
  D. Park, Z. Qian, G. Han, and S. Lim (2024)
  
  Mitigating dialogue hallucination for large vision language models via adversarial instruction tuning.
  
  arXiv preprint arXiv:2403.10492.
  
  Cited by: [§I](#S1.p1.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [23]
  S. Peng, W. Zhang, and D. Qu (2025)
  
  Think before retrieving: query-response relevance retrieval augmented generation.
  
  In 2025 International Joint Conference on Neural Networks (IJCNN),
  
  Vol. ,  pp. 1–8.
  
  External Links: [Document](https://dx.doi.org/10.1109/IJCNN64981.2025.11227249)
  
  Cited by: [§I](#S1.p2.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [24]
  A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. (2021)
  
  Learning transferable visual models from natural language supervision.
  
  In International conference on machine learning,
  
   pp. 8748–8763.
  
  Cited by: [§III-C](#S3.SS3.p1.1 "III-C Baselines ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [25]
  M. Shao, A. Basit, R. Karri, and M. Shafique (2024)
  
  Survey of different large language model architectures: trends, benchmarks, and challenges.
  
  IEEE Access.
  
  Cited by: [§I](#S1.p1.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [26]
  G. Van Horn, E. Cole, S. Beery, K. Wilber, S. Belongie, and O. Mac Aodha (2021)
  
  Benchmarking representation learning for natural world image collections.
  
  In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
  
   pp. 12884–12893.
  
  Cited by: [§III-A](#S3.SS1.p2.1 "III-A Datasets and Knowledge Bases ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [27]
  P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, et al. (2024)
  
  Qwen2-vl: enhancing vision-language model’s perception of the world at any resolution.
  
  arXiv preprint arXiv:2409.12191.
  
  Cited by: [§III-B](#S3.SS2.p3.1 "III-B Backbone Models and Metrics ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [28]
  J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al. (2022)
  
  Chain-of-thought prompting elicits reasoning in large language models.
  
  Advances in neural information processing systems 35,  pp. 24824–24837.
  
  Cited by: [§III-C](#S3.SS3.p1.1 "III-C Baselines ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [29]
  T. Weyand, A. Araujo, B. Cao, and J. Sim (2020)
  
  Google landmarks dataset v2-a large-scale benchmark for instance-level recognition and retrieval.
  
  In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition,
  
   pp. 2575–2584.
  
  Cited by: [§III-A](#S3.SS1.p2.1 "III-A Datasets and Knowledge Bases ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [30]
  J. Wu, J. Lu, A. Sabharwal, and R. Mottaghi (2022)
  
  Multi-modal answer validation for knowledge-based vqa.
  
  In Proceedings of the AAAI conference on artificial intelligence,
  
  Vol. 36,  pp. 2712–2721.
  
  Cited by: [§III-A](#S3.SS1.p1.1 "III-A Datasets and Knowledge Bases ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [31]
  S. Wu, Y. Xiong, Y. Cui, H. Wu, C. Chen, Y. Yuan, L. Huang, X. Liu, T. Kuo, N. Guan, et al. (2024)
  
  Retrieval-augmented generation for natural language processing: a survey.
  
  arXiv preprint arXiv:2407.13193.
  
  Cited by: [§I](#S1.p2.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [32]
  J. Xu, M. Moor, and J. Leskovec (2024)
  
  Reverse image retrieval cues parametric memory in multimodal llms.
  
  arXiv preprint arXiv:2405.18740.
  
  Cited by: [§I](#S1.p4.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning"),
  [§III-C](#S3.SS3.p1.1 "III-C Baselines ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [33]
  Y. Yan and W. Xie (2024)
  
  Echosight: advancing visual-language models with wiki knowledge.
  
  arXiv preprint arXiv:2407.12735.
  
  Cited by: [§III-A](#S3.SS1.p2.1 "III-A Datasets and Knowledge Bases ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [34]
  M. Yasunaga, A. Aghajanyan, W. Shi, R. James, J. Leskovec, P. Liang, M. Lewis, L. Zettlemoyer, and W. Yih (2022)
  
  Retrieval-augmented multimodal language modeling.
  
  arXiv preprint arXiv:2211.12561.
  
  Cited by: [§I](#S1.p3.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [35]
  Q. Yu, Z. Xiao, B. Li, Z. Wang, C. Chen, and W. Zhang (2025)
  
  MRAMG-bench: a beyondtext benchmark for multimodal retrieval-augmented multimodal generation.
  
  arXiv preprint arXiv:2502.04176.
  
  Cited by: [§I](#S1.p3.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [36]
  W. Zhai (2025)
  
  SAM-rag: an self-adaptive framework for multimodal retrieval-augmented generation.
  
  In 2025 International Joint Conference on Neural Networks (IJCNN),
  
   pp. 1–8.
  
  Cited by: [§I](#S1.p3.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [37]
  P. Zhao, H. Zhang, Q. Yu, Z. Wang, Y. Geng, F. Fu, L. Yang, W. Zhang, J. Jiang, and B. Cui (2024)
  
  Retrieval-augmented generation for ai-generated content: a survey.
  
  arXiv preprint arXiv:2402.19473.
  
  Cited by: [§I](#S1.p2.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [38]
  R. Zhao, H. Chen, W. Wang, F. Jiao, X. L. Do, C. Qin, B. Ding, X. Guo, M. Li, X. Li, et al. (2023)
  
  Retrieving multimodal information for augmented generation: a survey.
  
  arXiv preprint arXiv:2303.10868.
  
  Cited by: [§I](#S1.p3.1 "I Introduction ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").
- [39]
  Z. Zhu, J. Yu, Y. Wang, Y. Sun, Y. Hu, and Q. Wu (2020)
  
  Mucko: multi-layer cross-modal knowledge reasoning for fact-based visual question answering.
  
  arXiv preprint arXiv:2006.09073.
  
  Cited by: [§III-A](#S3.SS1.p1.1 "III-A Datasets and Knowledge Bases ‣ III Experiments ‣ Multimodal Adaptive Retrieval Augmented Generation through Internal Representation Learning").

