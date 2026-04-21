# Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction


\MakePerPage

[2]footnote


# Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction

Liang Wang1 2 Qiang Liu1 2  Shaozhen Liu1 Xin Sun3 Shu Wu1 2 Liang Wang1 2 3
  
1New Laboratory of Pattern Recognition (NLPR)
  
State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS)
  
Institute of Automation, Chinese Academy of Sciences (CASIA)
  
2 School of Artificial Intelligence, University of Chinese Academy of Sciences
  
3 University of Science and Technology of China
  

liang.wang@cripac.ia.ac.cn, qiang.liu@nlpr.ia.ac.cn, liushaozhen2025@ia.ac.cn
  
sunxin000@mail.ustc.edu.cn, {shu.wu, wangliang}@nlpr.ia.ac.cn
Corresponding author

###### Abstract

Molecular property prediction (MPP) is integral to drug discovery and material science,
but often faces the challenge of data scarcity in real-world scenarios.
Addressing this, few-shot molecular property prediction (FSMPP) has been developed. Unlike other few-shot tasks, FSMPP typically employs a pre-trained molecular encoder and a context-aware classifier, benefiting from molecular pre-training and molecular context information.
Despite these advancements, existing methods struggle with the ineffective fine-tuning of pre-trained encoders.
We attribute this issue to the imbalance between the abundance of tunable parameters and the scarcity of labeled molecules, and the lack of contextual perceptiveness in the encoders.
To overcome this hurdle, we propose a parameter-efficient in-context tuning method, named Pin-Tuning.
Specifically, we propose a lightweight adapter for pre-trained message passing layers (MP-Adapter) and Bayesian weight consolidation for pre-trained atom/bond embedding layers (Emb-BWC), to achieve parameter-efficient tuning while preventing over-fitting and catastrophic forgetting.
Additionally, we enhance the MP-Adapters with contextual perceptiveness. This innovation allows for in-context tuning of the pre-trained encoder, thereby improving its adaptability for specific FSMPP tasks.
When evaluated on public datasets, our method demonstrates superior tuning with fewer trainable parameters, improving few-shot predictive performance.\*\*\*Code is available at: <https://github.com/CRIPAC-DIG/Pin-Tuning>


## 1 Introduction

In the field of drug discovery and material science, molecular property prediction (MPP) stands as a pivotal task [[5](https://arxiv.org/html/2411.01158v1#bib.bib5), [9](https://arxiv.org/html/2411.01158v1#bib.bib9), [63](https://arxiv.org/html/2411.01158v1#bib.bib63)]. MPP involves the prediction of molecular properties like solubility and toxicity, based on their structural and physicochemical characteristics, which is integral to the development of new pharmaceuticals and materials. However, a major challenge encountered in real-world MPP scenarios is data scarcity. Obtaining extensive molecular data with well-characterized properties can be time-consuming and expensive.
To address this, few-shot molecular property prediction (FSMPP) has emerged as a crucial approach, enabling predictions with limited labeled molecules [[1](https://arxiv.org/html/2411.01158v1#bib.bib1), [41](https://arxiv.org/html/2411.01158v1#bib.bib41), [4](https://arxiv.org/html/2411.01158v1#bib.bib4)].


The methodology for general MPP typically adheres to an encoder-classifier framework [[71](https://arxiv.org/html/2411.01158v1#bib.bib71), [23](https://arxiv.org/html/2411.01158v1#bib.bib23), [27](https://arxiv.org/html/2411.01158v1#bib.bib27), [56](https://arxiv.org/html/2411.01158v1#bib.bib56)], as illustrated in [Figure 2](https://arxiv.org/html/2411.01158v1#S4.F2 "In 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction")(a). In this streamlined framework, the encoder converts molecular structures into vectorized representations [[12](https://arxiv.org/html/2411.01158v1#bib.bib12), [28](https://arxiv.org/html/2411.01158v1#bib.bib28), [50](https://arxiv.org/html/2411.01158v1#bib.bib50), [67](https://arxiv.org/html/2411.01158v1#bib.bib67), [2](https://arxiv.org/html/2411.01158v1#bib.bib2)], and then the classifier uses these representations to predict molecular properties.
In the context of few-shot scenarios, two significant discoveries have been instrumental in advancing this task.
Firstly, pre-trained molecular encoders have demonstrated consistent effectiveness in FSMPP tasks [[20](https://arxiv.org/html/2411.01158v1#bib.bib20), [14](https://arxiv.org/html/2411.01158v1#bib.bib14), [58](https://arxiv.org/html/2411.01158v1#bib.bib58)]. This indicates the utility of leveraging pre-acquired knowledge in dealing with data-limited scenarios.
Secondly, unlike typical few-shot tasks such as image classification [[57](https://arxiv.org/html/2411.01158v1#bib.bib57), [48](https://arxiv.org/html/2411.01158v1#bib.bib48)], FSMPP tasks greatly benefits from molecular context information. This involves comprehending the seen many-to-many relationships between molecules and properties [[58](https://arxiv.org/html/2411.01158v1#bib.bib58), [45](https://arxiv.org/html/2411.01158v1#bib.bib45), [73](https://arxiv.org/html/2411.01158v1#bib.bib73)], as molecules are multi-labeled by various properties.
These two discoveries have collectively led to the development of the widely used FSMPP framework that utilizes a pre-trained encoder followed by a context-aware classifier, as shown in [Figure 2](https://arxiv.org/html/2411.01158v1#S4.F2 "In 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction")(b).


![Refer to caption](09_Pin-Tuning_images/x1.png)

Figure 1: Comparison of molecular encoders trained via different paradigms: train-from-scratch, pretrain-then-freeze, and pretrain-then-finetune.
The evaluation is conducted across two datasets and three encoder architectures [[20](https://arxiv.org/html/2411.01158v1#bib.bib20), [47](https://arxiv.org/html/2411.01158v1#bib.bib47), [66](https://arxiv.org/html/2411.01158v1#bib.bib66)].
The results consistently demonstrate that while pretraining outperforms training from scratch, the current methods do not yet effectively facilitate finetuning.


Despite the progress, there are observed limitations in the current approaches to FSMPP. Notably, while using a pre-trained molecular encoder generally outperforms training from scratch, fine-tuning the pre-trained encoder often leads to inferior results compared to keeping it frozen,
which can be observed in [Figure 1](https://arxiv.org/html/2411.01158v1#S1.F1 "In 1 Introduction ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").


The observed ineffective fine-tuning can be attributed to two primary factors:
(i) Imbalance between the abundance of tunable parameters and the scarcity of labeled molecules: fine-tuning all parameters of a pre-trained encoder with few labeled molecules leads to a disproportionate ratio of tunable parameters to available data. This imbalance often results in over-fitting and catastrophic forgetting [[7](https://arxiv.org/html/2411.01158v1#bib.bib7), [6](https://arxiv.org/html/2411.01158v1#bib.bib6)].
(ii) Limited contextual perceptiveness in the encoder: while molecular context is leveraged to enhance the classifier [[58](https://arxiv.org/html/2411.01158v1#bib.bib58), [73](https://arxiv.org/html/2411.01158v1#bib.bib73)], the encoder typically lacks the explicit capability to perceive this context, relying instead on implicit gradient-based optimization.
This leads to the encoder not directly engaging with the nuanced molecular context information that is critical in FSMPP tasks.
In summary, while significant strides have been made, the challenges of imbalance between the number of parameters and labeled data, along with the need for contextual perceptiveness in the encoder, necessitate more sophisticated methodologies in this domain.


Based on the aforementioned analysis, we propose the parameter-efficient in-context tuning method, named Pin-Tuning, to address the two primary challenges in FSMPP.
To overcome the parameter-data imbalance, we propose a parameter-efficient chemical knowledge adaptation approach for pre-trained molecular encoders. A lightweight adapters (MP-Adapter) are designed to tune the pre-trained message passing layers efficiently.
Additionally, we impose a Bayesian weight consolidation (Emb-BWC) on the pre-trained embedding layers to prevent aggressive parameter updates, thereby mitigating the risk of over-fitting and catastrophic forgetting.
To address the second challenge, we further endow the MP-Adapter with the capability to perceive context. This innovation allows for in-context tuning of the pre-trained molecular encoders, enabling them to adapt more effectively to specific downstream tasks.
Our approach is rigorously evaluated on public datasets. The experimental results demonstrate that our method achieves superior tuning performance with fewer trainable parameters, leading to enhanced performance in few-shot molecular property prediction.


The main contributions of our work are summarized as follows:

- •
  
  We analyze the deficiencies of existing FSMPP approaches regarding the adaptation of pre-trained molecular encoders. The key issues include an imbalance between the number of tunable parameters and labeled molecules, as well as a lack of contextual perceptiveness in the encoders.
- •
  
  We propose Pin-Tuning to adapt the pre-trained molecular encoders for FSMPP tasks. This includes the MP-Adapter for message passing layers and the Emb-BWC for embedding layers,
  facilitating parameter-efficient tuning of pre-trained molecular encoders.
- •
  
  We further endow the MP-Adapter with the capability to perceive context to allows for in-context tuning, which provides more meaningful adaptation guidance during the tuning process.
- •
  
  We conduct extensive experiments on benchmark datasets, which show that Pin-Tuning outperforms state-of-the-art methods on FSMPP by effectively tuning pre-trained molecular encoders.


## 2 Related work

Few-shot molecular property prediction.
Few-shot molecular property prediction aims to accurately predict the properties of new molecules with limited training data [[49](https://arxiv.org/html/2411.01158v1#bib.bib49)].
Early research applied general few-shot techniques to FSMPP.
IterRefLSTM [[1](https://arxiv.org/html/2411.01158v1#bib.bib1)] is the pioneer work to leverage metric learning to solve FSMPP problem. Following this, Meta-GGNN [[41](https://arxiv.org/html/2411.01158v1#bib.bib41)] and Meta-MGNN [[14](https://arxiv.org/html/2411.01158v1#bib.bib14)] introduce meta-learning with graph neural networks, setting a foundational framework that subsequent studies have continued to build upon [[39](https://arxiv.org/html/2411.01158v1#bib.bib39), [40](https://arxiv.org/html/2411.01158v1#bib.bib40), [4](https://arxiv.org/html/2411.01158v1#bib.bib4)].
It is noteworthy that Meta-MGNN employs a pre-trained molecular encoder [[20](https://arxiv.org/html/2411.01158v1#bib.bib20)] and achieves superior results through fine-tuning in the meta-learning process compared to training from scratch.
In fact, pre-trained graph neural networks [[64](https://arxiv.org/html/2411.01158v1#bib.bib64), [36](https://arxiv.org/html/2411.01158v1#bib.bib36), [17](https://arxiv.org/html/2411.01158v1#bib.bib17), [54](https://arxiv.org/html/2411.01158v1#bib.bib54), [37](https://arxiv.org/html/2411.01158v1#bib.bib37)] have shown promise in enhancing various graph-based downstream tasks [[52](https://arxiv.org/html/2411.01158v1#bib.bib52), [13](https://arxiv.org/html/2411.01158v1#bib.bib13)], including molecular property prediction [[60](https://arxiv.org/html/2411.01158v1#bib.bib60), [62](https://arxiv.org/html/2411.01158v1#bib.bib62), [38](https://arxiv.org/html/2411.01158v1#bib.bib38), [72](https://arxiv.org/html/2411.01158v1#bib.bib72)].
Recent efforts have shifted towards leveraging unique nature in FSMPP, such as the many-to-many relationships between molecules and properties arising from the multi-labeled nature of molecules, often referred to as the molecular context.
PAR [[58](https://arxiv.org/html/2411.01158v1#bib.bib58)] initially employs graph structure learning [[32](https://arxiv.org/html/2411.01158v1#bib.bib32), [55](https://arxiv.org/html/2411.01158v1#bib.bib55)] to connect similar molecules through a homogeneous context graph.
MHNfs [[45](https://arxiv.org/html/2411.01158v1#bib.bib45)] introduces a large-scale external molecular library as context to augment the limited known information.
GS-Meta [[73](https://arxiv.org/html/2411.01158v1#bib.bib73)] further incorporates auxiliary task to depict the many-to-many relationships.


Parameter-efficient tuning.
As pre-training techniques have advanced, tuning of pre-trained models has become increasingly crucial.
Traditional full fine-tuning approaches updates all parameters, often leading to high computational costs and the risk of over-fitting, especially when available data for downstream tasks are limited [[33](https://arxiv.org/html/2411.01158v1#bib.bib33), [15](https://arxiv.org/html/2411.01158v1#bib.bib15)]. This challenge has led to the emergence of parameter-efficient tuning [[26](https://arxiv.org/html/2411.01158v1#bib.bib26), [29](https://arxiv.org/html/2411.01158v1#bib.bib29), [34](https://arxiv.org/html/2411.01158v1#bib.bib34)].
The philosophy of parameter-efficient tuning is to optimize a small subset of parameters, reducing the computational costs while retaining or even improving performance on downstream tasks [[19](https://arxiv.org/html/2411.01158v1#bib.bib19), [69](https://arxiv.org/html/2411.01158v1#bib.bib69)].
Among the various strategies, the adapters [[18](https://arxiv.org/html/2411.01158v1#bib.bib18), [42](https://arxiv.org/html/2411.01158v1#bib.bib42), [59](https://arxiv.org/html/2411.01158v1#bib.bib59)] have gained prominence. Adapters are small modules inserted between the pre-trained layers. During the tuning process, only the parameters of these adapters are updated while the rest remains frozen, which not only improves tuning efficiency but also offers an elegant solution to the generalization [[70](https://arxiv.org/html/2411.01158v1#bib.bib70), [30](https://arxiv.org/html/2411.01158v1#bib.bib30), [8](https://arxiv.org/html/2411.01158v1#bib.bib8)].
By keeping the majority of the pre-trained parameters intact, adapters preserve the rich pre-trained knowledge.
This attribute is particularly valuable in many real-world applications including FSMPP.


## 3 Preliminaries

### 3.1 Problem formulation

Let {𝒯}𝒯\{\mathcal{T}\}{ caligraphic\_T } be a collection of tasks, where each task 𝒯𝒯\mathcal{T}caligraphic\_T involves the prediction of a property p𝑝pitalic\_p. The training set comprising multiple tasks {𝒯train}subscript𝒯train\{\mathcal{T}\_{\text{train}}\}{ caligraphic\_T start\_POSTSUBSCRIPT train end\_POSTSUBSCRIPT }, is represented as 𝒟train={(mi,yi,t)|t∈{𝒯train}}subscript𝒟trainconditional-setsubscript𝑚𝑖subscript𝑦

𝑖𝑡𝑡subscript𝒯train\mathcal{D}\_{\text{train}}=\{(m\_{i},y\_{i,t})|t\in\{\mathcal{T}\_{\text{train}}\}\}caligraphic\_D start\_POSTSUBSCRIPT train end\_POSTSUBSCRIPT = { ( italic\_m start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT , italic\_y start\_POSTSUBSCRIPT italic\_i , italic\_t end\_POSTSUBSCRIPT ) | italic\_t ∈ { caligraphic\_T start\_POSTSUBSCRIPT train end\_POSTSUBSCRIPT } }, with misubscript𝑚𝑖m\_{i}italic\_m start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT indicating a molecule and yi,tsubscript𝑦

𝑖𝑡y\_{i,t}italic\_y start\_POSTSUBSCRIPT italic\_i , italic\_t end\_POSTSUBSCRIPT its associated label for task t𝑡titalic\_t. Correspondingly, the test set 𝒟testsubscript𝒟test\mathcal{D}\_{\text{test}}caligraphic\_D start\_POSTSUBSCRIPT test end\_POSTSUBSCRIPT, formed by tasks {𝒯test}subscript𝒯test\{\mathcal{T}\_{\text{test}}\}{ caligraphic\_T start\_POSTSUBSCRIPT test end\_POSTSUBSCRIPT }, ensures a separation of properties between training and testing phases, as the property sets {ptrain}subscript𝑝train\{p\_{\text{train}}\}{ italic\_p start\_POSTSUBSCRIPT train end\_POSTSUBSCRIPT } and {ptest}subscript𝑝test\{p\_{\text{test}}\}{ italic\_p start\_POSTSUBSCRIPT test end\_POSTSUBSCRIPT } are disjoint ({ptrain}∩{ptest}=∅subscript𝑝trainsubscript𝑝test\{p\_{\text{train}}\}\cap\{p\_{\text{test}}\}=\emptyset{ italic\_p start\_POSTSUBSCRIPT train end\_POSTSUBSCRIPT } ∩ { italic\_p start\_POSTSUBSCRIPT test end\_POSTSUBSCRIPT } = ∅).


The goal of FSMPP is to train a model using 𝒟trainsubscript𝒟train\mathcal{D}\_{\text{train}}caligraphic\_D start\_POSTSUBSCRIPT train end\_POSTSUBSCRIPT that can accurately infer new properties from a limited number of labeled molecules in 𝒟testsubscript𝒟test\mathcal{D}\_{\text{test}}caligraphic\_D start\_POSTSUBSCRIPT test end\_POSTSUBSCRIPT.
Episodic training has emerged as a promising strategy in meta-learning [[10](https://arxiv.org/html/2411.01158v1#bib.bib10), [16](https://arxiv.org/html/2411.01158v1#bib.bib16)] to deal with few-shot problem. Instead of retaining all {𝒯train}subscript𝒯train\{\mathcal{T}\_{\text{train}}\}{ caligraphic\_T start\_POSTSUBSCRIPT train end\_POSTSUBSCRIPT } tasks in memory, episodes {Et}t=1Bsubscriptsuperscriptsubscript𝐸𝑡𝐵𝑡1\{E\_{t}\}^{B}\_{t=1}{ italic\_E start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT } start\_POSTSUPERSCRIPT italic\_B end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_t = 1 end\_POSTSUBSCRIPT are iteratively sampled throughout the training process. For each episode Etsubscript𝐸𝑡E\_{t}italic\_E start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT, a particular task 𝒯tsubscript𝒯𝑡\mathcal{T}\_{t}caligraphic\_T start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT is selected from the training set, along with corresponding support set 𝒮tsubscript𝒮𝑡\mathcal{S}\_{t}caligraphic\_S start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT and query set 𝒬tsubscript𝒬𝑡\mathcal{Q}\_{t}caligraphic\_Q start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT. Typically, the prediction task involves classifying molecules into two classes: positive (y=1𝑦1y=1italic\_y = 1) or negative (y=0𝑦0y=0italic\_y = 0). Then a 2-way K𝐾Kitalic\_K-shot episode Et=(𝒮t,𝒬t)subscript𝐸𝑡subscript𝒮𝑡subscript𝒬𝑡E\_{t}=(\mathcal{S}\_{t},\mathcal{Q}\_{t})italic\_E start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT = ( caligraphic\_S start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT , caligraphic\_Q start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT ) is constructed. The support set 𝒮t={(mis,yi,ts)}i=12⁢Ksubscript𝒮𝑡subscriptsuperscriptsubscriptsuperscript𝑚𝑠𝑖subscriptsuperscript𝑦𝑠

𝑖𝑡2𝐾𝑖1\mathcal{S}\_{t}=\{(m^{s}\_{i},y^{s}\_{i,t})\}^{2K}\_{i=1}caligraphic\_S start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT = { ( italic\_m start\_POSTSUPERSCRIPT italic\_s end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT , italic\_y start\_POSTSUPERSCRIPT italic\_s end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i , italic\_t end\_POSTSUBSCRIPT ) } start\_POSTSUPERSCRIPT 2 italic\_K end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT includes 2⁢K2𝐾2K2 italic\_K examples, each class contributing K𝐾Kitalic\_K molecules. The query set containing M𝑀Mitalic\_M molecules is denoted as 𝒬t={(miq,yi,tq)}i=1Msubscript𝒬𝑡subscriptsuperscriptsubscriptsuperscript𝑚𝑞𝑖subscriptsuperscript𝑦𝑞

𝑖𝑡𝑀𝑖1\mathcal{Q}\_{t}=\{(m^{q}\_{i},y^{q}\_{i,t})\}^{M}\_{i=1}caligraphic\_Q start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT = { ( italic\_m start\_POSTSUPERSCRIPT italic\_q end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT , italic\_y start\_POSTSUPERSCRIPT italic\_q end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i , italic\_t end\_POSTSUBSCRIPT ) } start\_POSTSUPERSCRIPT italic\_M end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT.


### 3.2 Encoder-classifier framework for FSMPP

Encoder-classifier framework is widely adopted in FSMPP methods. As illustrated in [Figure 2](https://arxiv.org/html/2411.01158v1#S4.F2 "In 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction")(a), given a molecule m𝑚mitalic\_m whose property need to be predicted, a molecular encoder f⁢(⋅)𝑓⋅f(\cdot)italic\_f ( ⋅ ) first learns the molecule’s representation based on its structure, i.e., 𝒉m=f⁢(m)∈ℝdsubscript𝒉𝑚𝑓𝑚superscriptℝ𝑑\boldsymbol{h}\_{m}=f(m)\in\mathbb{R}^{d}bold\_italic\_h start\_POSTSUBSCRIPT italic\_m end\_POSTSUBSCRIPT = italic\_f ( italic\_m ) ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT. The molecule m𝑚mitalic\_m is generally represented as a graph m=(𝒱,𝐀,𝐗,𝐄)𝑚𝒱𝐀𝐗𝐄m=(\mathcal{V},\mathbf{A},\mathbf{X},\mathbf{E})italic\_m = ( caligraphic\_V , bold\_A , bold\_X , bold\_E ), where 𝒱𝒱\mathcal{V}caligraphic\_V denotes the nodes (atoms), 𝐀𝐀\mathbf{A}bold\_A represents the adjacent matrix defined by edges (chemical bonds), and 𝐗,𝐄

𝐗𝐄\mathbf{X},\mathbf{E}bold\_X , bold\_E denote the original feature of atoms and bonds, then graph neural networks (GNNs) are employed as the molecular encoders [[44](https://arxiv.org/html/2411.01158v1#bib.bib44), [51](https://arxiv.org/html/2411.01158v1#bib.bib51), [21](https://arxiv.org/html/2411.01158v1#bib.bib21)]. Subsequently, the learned molecular representation is fed into a classifier g⁢(⋅)𝑔⋅g(\cdot)italic\_g ( ⋅ ) to obtain the prediction y^=g⁢(𝒉m)^𝑦𝑔subscript𝒉𝑚\hat{y}=g(\boldsymbol{h}\_{m})over^ start\_ARG italic\_y end\_ARG = italic\_g ( bold\_italic\_h start\_POSTSUBSCRIPT italic\_m end\_POSTSUBSCRIPT ). The model is trained by minimizing the discrepancy between y^^𝑦\hat{y}over^ start\_ARG italic\_y end\_ARG and the ground truth y𝑦yitalic\_y.


Further, two key discoveries have been pivotal for FSMPP. The first is the proven effectiveness of pre-trained molecular encoders,
while the second is the significant advantage gained from molecular context.
Together, these discoveries have further reshaped the widely adopted FSMPP framework, which combines a pre-trained encoder followed by a context-aware classifier, as shown in [Figure 2](https://arxiv.org/html/2411.01158v1#S4.F2 "In 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction")(b).


### 3.3 Pre-trained molecular encoders (PMEs)

Due to the scarcity of labeled data in molecular tasks, molecular pre-training has emerged as a crucial area, which involves training encoders on extensive molecular datasets to extract informative representations.
Pre-GNN [[20](https://arxiv.org/html/2411.01158v1#bib.bib20)] is a classic pre-trained molecular encoder that has been widely used in addressing FSMPP tasks [[14](https://arxiv.org/html/2411.01158v1#bib.bib14), [58](https://arxiv.org/html/2411.01158v1#bib.bib58), [73](https://arxiv.org/html/2411.01158v1#bib.bib73)]. The backbone of Pre-GNN is a modified version of Graph Isomorphism Network (GIN) [[65](https://arxiv.org/html/2411.01158v1#bib.bib65)] tailored to molecules, which we call GIN-Mol, consisting of multiple atom/bond embedding layers and message passing layers.


Atom/Bond embedding layers.
The raw atom features and bond features are both categorical vectors, denoted as (iv,1,iv,2,…,iv,|En|)subscript𝑖

𝑣1subscript𝑖

𝑣2…subscript𝑖

𝑣subscript𝐸𝑛(i\_{v,1},i\_{v,2},\ldots,i\_{v,|E\_{n}|})( italic\_i start\_POSTSUBSCRIPT italic\_v , 1 end\_POSTSUBSCRIPT , italic\_i start\_POSTSUBSCRIPT italic\_v , 2 end\_POSTSUBSCRIPT , … , italic\_i start\_POSTSUBSCRIPT italic\_v , | italic\_E start\_POSTSUBSCRIPT italic\_n end\_POSTSUBSCRIPT | end\_POSTSUBSCRIPT ) and (je,1,je,2,…,je,|Ee|)subscript𝑗

𝑒1subscript𝑗

𝑒2…subscript𝑗

𝑒subscript𝐸𝑒(j\_{e,1},j\_{e,2},\ldots,j\_{e,|E\_{e}|})( italic\_j start\_POSTSUBSCRIPT italic\_e , 1 end\_POSTSUBSCRIPT , italic\_j start\_POSTSUBSCRIPT italic\_e , 2 end\_POSTSUBSCRIPT , … , italic\_j start\_POSTSUBSCRIPT italic\_e , | italic\_E start\_POSTSUBSCRIPT italic\_e end\_POSTSUBSCRIPT | end\_POSTSUBSCRIPT ) for atom v𝑣vitalic\_v and bond e𝑒eitalic\_e, respectively.
These categorical features are embedded as:

|  | 𝒉v(0)=∑a=1|En|EmbAtoma⁢(iv,a),𝒉e(l)=∑b=1|Ee|EmbBondb(l)⁢(je,b),formulae-sequencesuperscriptsubscript𝒉𝑣0superscriptsubscript𝑎1subscript𝐸𝑛subscriptEmbAtom𝑎subscript𝑖  𝑣𝑎superscriptsubscript𝒉𝑒𝑙superscriptsubscript𝑏1subscript𝐸𝑒superscriptsubscriptEmbBond𝑏𝑙subscript𝑗  𝑒𝑏\boldsymbol{h}\_{v}^{(0)}=\sum\nolimits\_{a=1}^{|E\_{n}|}\texttt{EmbAtom}\_{a}(i\_{% v,a}),\quad\boldsymbol{h}\_{e}^{(l)}=\sum\nolimits\_{b=1}^{|E\_{e}|}\texttt{% EmbBond}\_{b}^{(l)}(j\_{e,b}),bold\_italic\_h start\_POSTSUBSCRIPT italic\_v end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( 0 ) end\_POSTSUPERSCRIPT = ∑ start\_POSTSUBSCRIPT italic\_a = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT | italic\_E start\_POSTSUBSCRIPT italic\_n end\_POSTSUBSCRIPT | end\_POSTSUPERSCRIPT EmbAtom start\_POSTSUBSCRIPT italic\_a end\_POSTSUBSCRIPT ( italic\_i start\_POSTSUBSCRIPT italic\_v , italic\_a end\_POSTSUBSCRIPT ) , bold\_italic\_h start\_POSTSUBSCRIPT italic\_e end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT = ∑ start\_POSTSUBSCRIPT italic\_b = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT | italic\_E start\_POSTSUBSCRIPT italic\_e end\_POSTSUBSCRIPT | end\_POSTSUPERSCRIPT EmbBond start\_POSTSUBSCRIPT italic\_b end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT ( italic\_j start\_POSTSUBSCRIPT italic\_e , italic\_b end\_POSTSUBSCRIPT ) , |  | (1) |
| --- | --- | --- | --- |

where EmbAtoma⁢(⋅)a∈{1,…,|En|}subscriptEmbAtom𝑎subscript⋅𝑎1…subscript𝐸𝑛{\texttt{EmbAtom}\_{a}(\cdot)}\_{a\in\{1,\ldots,|E\_{n}|\}}EmbAtom start\_POSTSUBSCRIPT italic\_a end\_POSTSUBSCRIPT ( ⋅ ) start\_POSTSUBSCRIPT italic\_a ∈ { 1 , … , | italic\_E start\_POSTSUBSCRIPT italic\_n end\_POSTSUBSCRIPT | } end\_POSTSUBSCRIPT and EmbBondb⁢(⋅)b∈{1,…,|Ee|}subscriptEmbBond𝑏subscript⋅𝑏1…subscript𝐸𝑒\texttt{EmbBond}\_{b}(\cdot)\_{b\in\{1,\ldots,|E\_{e}|\}}EmbBond start\_POSTSUBSCRIPT italic\_b end\_POSTSUBSCRIPT ( ⋅ ) start\_POSTSUBSCRIPT italic\_b ∈ { 1 , … , | italic\_E start\_POSTSUBSCRIPT italic\_e end\_POSTSUBSCRIPT | } end\_POSTSUBSCRIPT represent embedding operations that map integer indices to d𝑑ditalic\_d-dimensional real vectors, i.e., 𝒉v(0),𝒉e(l)∈ℝd

superscriptsubscript𝒉𝑣0superscriptsubscript𝒉𝑒𝑙
superscriptℝ𝑑\boldsymbol{h}\_{v}^{(0)},\boldsymbol{h}\_{e}^{(l)}\in\mathbb{R}^{d}bold\_italic\_h start\_POSTSUBSCRIPT italic\_v end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( 0 ) end\_POSTSUPERSCRIPT , bold\_italic\_h start\_POSTSUBSCRIPT italic\_e end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT, l∈{0,1,…,L−1}𝑙01…𝐿1l\in\{0,1,\ldots,L-1\}italic\_l ∈ { 0 , 1 , … , italic\_L - 1 } represents the index of encoder layers, and L𝐿Litalic\_L is the number of encoder layers.
The atom embedding layer is present only in the first encoder layer, while an bond embedding layer exists in each layer.


Message passing layers.
At the l𝑙litalic\_l-th encoder layer, atom representations are updated by aggregating the features of neighboring atoms and chemical bonds:

|  | 𝒉v(l)=ReLU⁢(MLP(l)⁢(∑u𝒉u(l−1)+∑e=(v,u)𝒉e(l−1))),superscriptsubscript𝒉𝑣𝑙ReLUsuperscriptMLP𝑙subscript𝑢superscriptsubscript𝒉𝑢𝑙1subscript  𝑒𝑣𝑢superscriptsubscript𝒉𝑒𝑙1\boldsymbol{h}\_{v}^{(l)}=\texttt{ReLU}\left(\texttt{MLP}^{(l)}\left(\sum% \nolimits\_{u}\boldsymbol{h}\_{u}^{(l-1)}+\sum\nolimits\_{\begin{subarray}{c}e=(v% ,u)\end{subarray}}\boldsymbol{h}\_{e}^{(l-1)}\right)\right),bold\_italic\_h start\_POSTSUBSCRIPT italic\_v end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT = ReLU ( MLP start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT ( ∑ start\_POSTSUBSCRIPT italic\_u end\_POSTSUBSCRIPT bold\_italic\_h start\_POSTSUBSCRIPT italic\_u end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( italic\_l - 1 ) end\_POSTSUPERSCRIPT + ∑ start\_POSTSUBSCRIPT start\_ARG start\_ROW start\_CELL italic\_e = ( italic\_v , italic\_u ) end\_CELL end\_ROW end\_ARG end\_POSTSUBSCRIPT bold\_italic\_h start\_POSTSUBSCRIPT italic\_e end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( italic\_l - 1 ) end\_POSTSUPERSCRIPT ) ) , |  | (2) |
| --- | --- | --- | --- |

where u∈𝒩⁢(v)∪{v}𝑢𝒩𝑣𝑣u\in\mathcal{N}(v)\cup\{v\}italic\_u ∈ caligraphic\_N ( italic\_v ) ∪ { italic\_v } is the set of atoms connected to v𝑣vitalic\_v,
and 𝒉v(l)∈ℝdsuperscriptsubscript𝒉𝑣𝑙superscriptℝ𝑑\boldsymbol{h}\_{v}^{(l)}\in\mathbb{R}^{d}bold\_italic\_h start\_POSTSUBSCRIPT italic\_v end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT is the learned representation of atom v𝑣vitalic\_v at the l𝑙litalic\_l-th layer.
MLP⁢(⋅)MLP⋅\texttt{MLP}(\cdot)MLP ( ⋅ ) is implemented by 2-layer neural networks, in which the hidden dimension is d1subscript𝑑1d\_{1}italic\_d start\_POSTSUBSCRIPT 1 end\_POSTSUBSCRIPT.
After MLP, batch normalization is applied right before the ReLU.
The molecule-level representation 𝒉m∈ℝdsubscript𝒉𝑚superscriptℝ𝑑\boldsymbol{h}\_{m}\in\mathbb{R}^{d}bold\_italic\_h start\_POSTSUBSCRIPT italic\_m end\_POSTSUBSCRIPT ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT is obtained by averaging the atom representations at the final layer.


## 4 The proposed Pin-Tuning method

This section delves into our motivation and proposed method. Our framework for FSMPP is depicted in [Figure 2](https://arxiv.org/html/2411.01158v1#S4.F2 "In 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction")(c). The details of our principal design, Pin-Tuning for PMEs, is present in [Figure 2](https://arxiv.org/html/2411.01158v1#S4.F2 "In 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction")(d).


As shown in [Figure 1](https://arxiv.org/html/2411.01158v1#S1.F1 "In 1 Introduction ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction"), pretraining then finetuning molecular encoders is a common approach. However, fully fine-tuning yields results inferior to simply freezing them.
Thus, the following question arises:


How to effectively adapt pre-trained molecular encoders to downstream tasks, especially in few-shot scenarios?


We analyze the reasons of observed ineffective fine-tuning issue,
and attribute it to two primary factors:
(i) imbalance between the abundance of tunable parameters and the scarcity of labeled molecules, and
(ii) limited contextual perceptiveness in the encoder.


![Refer to caption](09_Pin-Tuning_images/x2.png)

Figure 2: (a) The vanilla encoder-classifier framework for MPP. (b) The framework widely adopted by existing FSMPP methods, which contains a pre-trained molecular encoder and a context-aware property classifier. (c) Our proposed framework for FSMPP, in which we introduce a Pin-Tuning method to update the pre-trained molecular encoder followed by the property classifier. (d) The details of our proposed Pin-Tuning method for pre-trained molecular encoders.
In (b) and (c), we use the property names like SR-HSE to denote the molecular context in episodes.


### 4.1 Parameter-efficient tuning for PMEs

To address the first cause of observed ineffective tuning, we reform the tuning method for PMEs. Instead of conducting full fine-tuning for all parameters, we propose tuning strategies specifically tailored to the message passing layers and embedding layers in PMEs, respectively.


#### 4.1.1 MP-Adapter: message passing layer-oriented adapter

For message passing layers in PMEs, the number of parameters is disproportionately large compared to the training samples. To mitigate this imbalance, we design a lightweight adapter targeted at the message passing layers, called MP-Adapter.
The pre-trained parameters in each message passing layer include parameters in the MLP and the following batch normalization.
We freeze all pre-trained parameters in message passing layers and add a lightweight trainable adapter after MLP in each message passing layer.
Formally, the adapter module for l𝑙litalic\_l-th layer can be represented as:

|  | 𝒛v(l)=FeedForwarddown⁢(𝒉v(l))∈ℝd2,superscriptsubscript𝒛𝑣𝑙subscriptFeedForwarddownsuperscriptsubscript𝒉𝑣𝑙superscriptℝsubscript𝑑2\displaystyle\boldsymbol{z}\_{v}^{(l)}=\texttt{FeedForward}\_{\texttt{down}}(% \boldsymbol{h}\_{v}^{(l)})\in\mathbb{R}^{d\_{2}},bold\_italic\_z start\_POSTSUBSCRIPT italic\_v end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT = FeedForward start\_POSTSUBSCRIPT down end\_POSTSUBSCRIPT ( bold\_italic\_h start\_POSTSUBSCRIPT italic\_v end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT ) ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT end\_POSTSUPERSCRIPT , |  | (3) |
| --- | --- | --- | --- |
|  | Δ⁢𝒉v(l)=FeedForwardup⁢(ϕ⁢(𝒛v(l)))∈ℝd,Δsuperscriptsubscript𝒉𝑣𝑙subscriptFeedForwardupitalic-ϕsuperscriptsubscript𝒛𝑣𝑙superscriptℝ𝑑\displaystyle\Delta\boldsymbol{h}\_{v}^{(l)}=\texttt{FeedForward}\_{\texttt{up}}% (\phi(\boldsymbol{z}\_{v}^{(l)}))\in\mathbb{R}^{d},roman\_Δ bold\_italic\_h start\_POSTSUBSCRIPT italic\_v end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT = FeedForward start\_POSTSUBSCRIPT up end\_POSTSUBSCRIPT ( italic\_ϕ ( bold\_italic\_z start\_POSTSUBSCRIPT italic\_v end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT ) ) ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT , |  | (4) |
| --- | --- | --- | --- |
|  | 𝒉~v(l)=LayerNorm⁢(𝒉v(l)+Δ⁢𝒉v(l))∈ℝd,superscriptsubscript~𝒉𝑣𝑙LayerNormsuperscriptsubscript𝒉𝑣𝑙Δsuperscriptsubscript𝒉𝑣𝑙superscriptℝ𝑑\displaystyle\tilde{\boldsymbol{h}}\_{v}^{(l)}=\texttt{LayerNorm}(\boldsymbol{h% }\_{v}^{(l)}+\Delta\boldsymbol{h}\_{v}^{(l)})\in\mathbb{R}^{d},over~ start\_ARG bold\_italic\_h end\_ARG start\_POSTSUBSCRIPT italic\_v end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT = LayerNorm ( bold\_italic\_h start\_POSTSUBSCRIPT italic\_v end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT + roman\_Δ bold\_italic\_h start\_POSTSUBSCRIPT italic\_v end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT ) ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT , |  | (5) |
| --- | --- | --- | --- |

where FeedForward⁢(⋅)FeedForward⋅\texttt{FeedForward}(\cdot)FeedForward ( ⋅ ) denotes feed forward layer and LayerNorm⁢(⋅)LayerNorm⋅\texttt{LayerNorm}(\cdot)LayerNorm ( ⋅ ) denotes layer normalization.
To limit the number of parameters, we introduce a bottleneck architecture. The adapters downscale the original features from d𝑑ditalic\_d dimensions to a smaller dimension d2subscript𝑑2d\_{2}italic\_d start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT, apply nonlinearity ϕitalic-ϕ\phiitalic\_ϕ, then upscale back to d𝑑ditalic\_d dimensions. By setting d2subscript𝑑2d\_{2}italic\_d start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT smaller than d𝑑ditalic\_d, we can limit the number of parameters added.
The adapter module has a skip-connection internally. With the skip-connection, we adopt the near-zero initialization for parameters in the adapter modules, so that the modules are initialized to approximate identity functions.
Therefore, the encoder with initialized adapters is equivalent to the pre-trained encoder.
Furthermore, we add a layer normalization after skip-connection for training stability.


#### 4.1.2 Emb-BWC: embedding layer-oriented Bayesian weight consolidation

Unlike message passing layers, embedding layers contain fewer parameters. Therefore,
we directly fine-tune the parameters of the embedding layers, but impose a constraint to limit the magnitude of parameter updates, preventing aggressive optimization and catastrophic forgetting.


The parameters in an embedding layer consist of an embedding matrix used for lookups based on the indices of the original features.
We stack the embedding matrices of all embedding layers to form Φ∈ℝE×dΦsuperscriptℝ𝐸𝑑\Phi\in\mathbb{R}^{E\times d}roman\_Φ ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_E × italic\_d end\_POSTSUPERSCRIPT, where E𝐸Eitalic\_E represents the total number of lookup entries.
Further, Φi∈ℝdsubscriptΦ𝑖superscriptℝ𝑑\Phi\_{i}\in\mathbb{R}^{d}roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT denotes the i𝑖iitalic\_i-th row’s embedding vector, and Φi,j∈ℝsubscriptΦ

𝑖𝑗ℝ\Phi\_{i,j}\in\mathbb{R}roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT ∈ blackboard\_R represents the j𝑗jitalic\_j-th dimensional value of ΦisubscriptΦ𝑖\Phi\_{i}roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT.


To avoid aggressive optimization of ΦΦ\Phiroman\_Φ, we derive a Bayesian weight consolidation framework tailored for embedding layers, called Emb-BWC,
by applying Bayesian learning theory [[3](https://arxiv.org/html/2411.01158v1#bib.bib3)] to fine-tuning.


Proposition 1: (Emb-BWC ensures an appropriate stability-plasticity trade-off for pre-trained embedding layers.) Let Φ∈ℝE×dΦsuperscriptℝ𝐸𝑑\Phi\in\mathbb{R}^{E\times d}roman\_Φ ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_E × italic\_d end\_POSTSUPERSCRIPT be the pre-trained embeddings before fine-tuning, and Φ′∈ℝE×dsuperscriptΦ′superscriptℝ𝐸𝑑\Phi^{\prime}\in\mathbb{R}^{E\times d}roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_E × italic\_d end\_POSTSUPERSCRIPT be the fine-tuned embeddings. Then, the embeddings can both retain the atom and bond properties obtained from pre-training and be appropriately updated to adapt to downstream FSMPP tasks, by introducing the following Emb-BWC loss into objective during the fine-tuning process:

|  | ℒEmb-BWC=−12⁢∑i=1E(Φi′−Φi)⊤⁢𝐇⁢(𝒟𝒫,Φi)⁢(Φi′−Φi),subscriptℒEmb-BWC12superscriptsubscript𝑖1𝐸superscriptsubscriptsuperscriptΦ′𝑖subscriptΦ𝑖top𝐇subscript𝒟𝒫subscriptΦ𝑖subscriptsuperscriptΦ′𝑖subscriptΦ𝑖\mathcal{L}\_{\textrm{Emb-BWC}}=-\frac{1}{2}\sum\_{i=1}^{E}(\Phi^{\prime}\_{i}-% \Phi\_{i})^{\top}\mathbf{H}(\mathcal{D}\_{\mathcal{P}},\Phi\_{i})(\Phi^{\prime}\_{% i}-\Phi\_{i}),caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT = - divide start\_ARG 1 end\_ARG start\_ARG 2 end\_ARG ∑ start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_E end\_POSTSUPERSCRIPT ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT ⊤ end\_POSTSUPERSCRIPT bold\_H ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) , |  | (6) |
| --- | --- | --- | --- |

where 𝐇⁢(𝒟𝒫,Φi)∈ℝd×d𝐇subscript𝒟𝒫subscriptΦ𝑖superscriptℝ𝑑𝑑\mathbf{H}(\mathcal{D}\_{\mathcal{P}},\Phi\_{i})\in\mathbb{R}^{d\times d}bold\_H ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d × italic\_d end\_POSTSUPERSCRIPT is the Hessian of the log likelihood ℒ𝒫subscriptℒ𝒫\mathcal{L}\_{\mathcal{P}}caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT of pre-training dataset 𝒟𝒫subscript𝒟𝒫\mathcal{D}\_{\mathcal{P}}caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT at ΦisubscriptΦ𝑖\Phi\_{i}roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT.


Details on the theoretical derivation of [Eq. 6](https://arxiv.org/html/2411.01158v1#S4.E6 "In 4.1.2 Emb-BWC: embedding layer-oriented Bayesian weight consolidation ‣ 4.1 Parameter-efficient tuning for PMEs ‣ 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") are given in [Appendix A](https://arxiv.org/html/2411.01158v1#A1 "Appendix A Derivation of Emb-BWC regularization ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").
Since 𝐇⁢(𝒟𝒫,Φi)𝐇subscript𝒟𝒫subscriptΦ𝑖\mathbf{H}(\mathcal{D}\_{\mathcal{P}},\Phi\_{i})bold\_H ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) is intractable to compute due to the great dimensionality of ΦΦ\Phiroman\_Φ, we adopt the diagonal approximation of Hessian.
By approximating 𝐇𝐇\mathbf{H}bold\_H as a diagonal matrix, the j𝑗jitalic\_j-th value on the diagonal of 𝐇𝐇\mathbf{H}bold\_H can be considered as the importance of the parameter Φi,jsubscriptΦ

𝑖𝑗\Phi\_{i,j}roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT.
The following three choices are considered.


Identity matrix. When using the identity matrix to approximate the negation of 𝐇𝐇\mathbf{H}bold\_H, [Eq. 6](https://arxiv.org/html/2411.01158v1#S4.E6 "In 4.1.2 Emb-BWC: embedding layer-oriented Bayesian weight consolidation ‣ 4.1 Parameter-efficient tuning for PMEs ‣ 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") is simplified to ℒEmb-BWCIM=12⁢∑i=1E∑j=1d(Φi,j′−Φi,j)2superscriptsubscriptℒEmb-BWCIM12superscriptsubscript𝑖1𝐸superscriptsubscript𝑗1𝑑superscriptsubscriptsuperscriptΦ′

𝑖𝑗subscriptΦ

𝑖𝑗2\mathcal{L}\_{\textrm{Emb-BWC}}^{\textrm{IM}}=\frac{1}{2}\sum\_{i=1}^{E}\sum\_{j=%
1}^{d}(\Phi^{\prime}\_{i,j}-\Phi\_{i,j})^{2}caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT IM end\_POSTSUPERSCRIPT = divide start\_ARG 1 end\_ARG start\_ARG 2 end\_ARG ∑ start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_E end\_POSTSUPERSCRIPT ∑ start\_POSTSUBSCRIPT italic\_j = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT, assigning equal importance to each parameter. This loss function is also known as L2 penalty with pre-trained model as the starting point (L2-SP) [[31](https://arxiv.org/html/2411.01158v1#bib.bib31)].


Diagonal of Fisher information matrix. The Fisher information matrix (FIM) 𝐅𝐅\mathbf{F}bold\_F is the negation of the expectation of the Hessian over the data distribution, i.e., 𝐅=−𝔼𝒟𝒫⁢[𝐇]𝐅subscript𝔼subscript𝒟𝒫delimited-[]𝐇\mathbf{F}=-\mathbb{E}\_{\mathcal{D}\_{\mathcal{P}}}[\mathbf{H}]bold\_F = - blackboard\_E start\_POSTSUBSCRIPT caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_POSTSUBSCRIPT [ bold\_H ], and the FIM can be further simplified with a diagonal approximation. Then, the [Eq. 6](https://arxiv.org/html/2411.01158v1#S4.E6 "In 4.1.2 Emb-BWC: embedding layer-oriented Bayesian weight consolidation ‣ 4.1 Parameter-efficient tuning for PMEs ‣ 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") is simplified to ℒEmb-BWCFIM=12⁢∑i=1E𝐅^i⁢(Φi′−Φi)2superscriptsubscriptℒEmb-BWCFIM12superscriptsubscript𝑖1𝐸subscript^𝐅𝑖superscriptsubscriptsuperscriptΦ′𝑖subscriptΦ𝑖2\mathcal{L}\_{\textrm{Emb-BWC}}^{\textrm{FIM}}=\frac{1}{2}\sum\_{i=1}^{E}\hat{%
\mathbf{F}}\_{i}(\Phi^{\prime}\_{i}-\Phi\_{i})^{2}caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT FIM end\_POSTSUPERSCRIPT = divide start\_ARG 1 end\_ARG start\_ARG 2 end\_ARG ∑ start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_E end\_POSTSUPERSCRIPT over^ start\_ARG bold\_F end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT, where 𝐅^i∈ℝdsubscript^𝐅𝑖superscriptℝ𝑑\hat{\mathbf{F}}\_{i}\in\mathbb{R}^{d}over^ start\_ARG bold\_F end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT is the diagonal of 𝐅⁢(𝒟𝒫,Φi)∈ℝd×d𝐅subscript𝒟𝒫subscriptΦ𝑖superscriptℝ𝑑𝑑\mathbf{F}(\mathcal{D}\_{\mathcal{P}},\Phi\_{i})\in\mathbb{R}^{d\times d}bold\_F ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d × italic\_d end\_POSTSUPERSCRIPT and the j𝑗jitalic\_j-th value in 𝐅^isubscript^𝐅𝑖\hat{\mathbf{F}}\_{i}over^ start\_ARG bold\_F end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT is computed as 𝔼𝒟𝒫⁢(∂ℒ𝒫/∂Φi,j)2subscript𝔼subscript𝒟𝒫superscriptsubscriptℒ𝒫subscriptΦ

𝑖𝑗2\mathbb{E}\_{\mathcal{D}\_{\mathcal{P}}}(\partial\mathcal{L}\_{\mathcal{P}}/%
\partial\Phi\_{i,j})^{2}blackboard\_E start\_POSTSUBSCRIPT caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_POSTSUBSCRIPT ( ∂ caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT / ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT. This is equivalent to elastic weight consolidation (EWC) [[24](https://arxiv.org/html/2411.01158v1#bib.bib24)].


Diagonal of embedding-wise Fisher information matrix. In different property prediction tasks, the impact of the same atoms and inter-atomic interactions may be significant or negligible. Therefore, we propose this choice to assign importance to parameters based on different embeddings, rather than treating each parameter individually. By defining Φ~i=∑jΦi,jsubscript~Φ𝑖subscript𝑗subscriptΦ

𝑖𝑗\tilde{\Phi}\_{i}=\sum\_{j}\Phi\_{i,j}over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT = ∑ start\_POSTSUBSCRIPT italic\_j end\_POSTSUBSCRIPT roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT, the total update of the embedding ΦisubscriptΦ𝑖\Phi\_{i}roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT can be represented as Δ⁢Φi=Φ~i′−Φ~i=∑j(Φi,j′−Φi,j)ΔsubscriptΦ𝑖superscriptsubscript~Φ𝑖′subscript~Φ𝑖subscript𝑗superscriptsubscriptΦ

𝑖𝑗′subscriptΦ

𝑖𝑗\Delta\Phi\_{i}=\tilde{\Phi}\_{i}^{\prime}-\tilde{\Phi}\_{i}=\sum\_{j}(\Phi\_{i,j}^%
{\prime}-\Phi\_{i,j})roman\_Δ roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT = over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT - over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT = ∑ start\_POSTSUBSCRIPT italic\_j end\_POSTSUBSCRIPT ( roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT ). Then, the [Eq. 6](https://arxiv.org/html/2411.01158v1#S4.E6 "In 4.1.2 Emb-BWC: embedding layer-oriented Bayesian weight consolidation ‣ 4.1 Parameter-efficient tuning for PMEs ‣ 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") is reformulated to
ℒEmb-EWCEFIM=12⁢∑i=1E𝐅~i⁢(Φ~i′−Φ~i)2superscriptsubscriptℒEmb-EWCEFIM12superscriptsubscript𝑖1𝐸subscript~𝐅𝑖superscriptsubscriptsuperscript~Φ′𝑖subscript~Φ𝑖2\mathcal{L}\_{\textrm{Emb-EWC}}^{\textrm{EFIM}}=\frac{1}{2}\sum\_{i=1}^{E}\tilde%
{\mathbf{F}}\_{i}(\tilde{\Phi}^{\prime}\_{i}-\tilde{\Phi}\_{i})^{2}caligraphic\_L start\_POSTSUBSCRIPT Emb-EWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT EFIM end\_POSTSUPERSCRIPT = divide start\_ARG 1 end\_ARG start\_ARG 2 end\_ARG ∑ start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_E end\_POSTSUPERSCRIPT over~ start\_ARG bold\_F end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ( over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT - over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT,
where
𝐅~i=∑j𝔼𝒟𝒫⁢(∂ℒ𝒫/∂Φi,j)2subscript~𝐅𝑖subscript𝑗subscript𝔼subscript𝒟𝒫superscriptsubscriptℒ𝒫subscriptΦ

𝑖𝑗2\tilde{\mathbf{F}}\_{i}=\sum\_{j}\mathbb{E}\_{\mathcal{D}\_{\mathcal{P}}}(\partial%
\mathcal{L}\_{\mathcal{P}}/{\partial\Phi\_{i,j}})^{2}over~ start\_ARG bold\_F end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT = ∑ start\_POSTSUBSCRIPT italic\_j end\_POSTSUBSCRIPT blackboard\_E start\_POSTSUBSCRIPT caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_POSTSUBSCRIPT ( ∂ caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT / ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT.


Detailed derivation is given in [Appendix A](https://arxiv.org/html/2411.01158v1#A1 "Appendix A Derivation of Emb-BWC regularization ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").
Intuitively, these three approximations employ different methods to assign importance to parameters. ℒEmb-BWCIMsuperscriptsubscriptℒEmb-BWCIM\mathcal{L}\_{\textrm{Emb-BWC}}^{\textrm{IM}}caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT IM end\_POSTSUPERSCRIPT assigns the same importance to each parameter, ℒEmb-BWCFIMsuperscriptsubscriptℒEmb-BWCFIM\mathcal{L}\_{\textrm{Emb-BWC}}^{\textrm{FIM}}caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT FIM end\_POSTSUPERSCRIPT assigns individual importance to each parameter, and ℒEmb-BWCEFIMsuperscriptsubscriptℒEmb-BWCEFIM\mathcal{L}\_{\textrm{Emb-BWC}}^{\textrm{EFIM}}caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT EFIM end\_POSTSUPERSCRIPT assigns the same importance to parameters within the same embedding vector.


### 4.2 Enabling contextual perceptiveness in MP-Adapter

For different property prediction tasks, the decisive substructures vary. As shown in [Figure 2](https://arxiv.org/html/2411.01158v1#S4.F2 "In 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction"), the ester group in the given molecule determines the property SR-HSE, while the carbon-carbon triple bond determines the property SR-MMP. If fine-tuning can be guided by molecular context, encoding context-specific molecular representations allows for dynamic representations of molecules tailored to specific tasks and enables the modeling of the context-specific significance of substructures.


![Refer to caption](09_Pin-Tuning_images/x3.png)

Figure 3: Convert the context information of a 2-shot episode into a context graph.


Extracting molecular context information.
In each episode, we consider the labels of the support molecules on the target property and seen properties, as well as the labels of the query molecules on seen properties, as the context of this episode.
We adopt the form of a graph to describe the context.
[Figure 3](https://arxiv.org/html/2411.01158v1#S4.F3 "In 4.2 Enabling contextual perceptiveness in MP-Adapter ‣ 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") demonstrates the transformation from original context data to a context graph. In the left table, the labels of molecules m1q,m22

superscriptsubscript𝑚1𝑞superscriptsubscript𝑚22m\_{1}^{q},m\_{2}^{2}italic\_m start\_POSTSUBSCRIPT 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_q end\_POSTSUPERSCRIPT , italic\_m start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT for property ptsubscript𝑝𝑡p\_{t}italic\_p start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT are the prediction targets, and the other shaded values are the available context. The right side shows the context graph constructed based on the available context.
Specifically, we construct context graph 𝒢t=(𝒱t,𝐀t,𝐗t)subscript𝒢𝑡subscript𝒱𝑡subscript𝐀𝑡subscript𝐗𝑡\mathcal{G}\_{t}=(\mathcal{V}\_{t},\mathbf{A}\_{t},\mathbf{X}\_{t})caligraphic\_G start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT = ( caligraphic\_V start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT , bold\_A start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT , bold\_X start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT ) for episode Etsubscript𝐸𝑡E\_{t}italic\_E start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT. It contains M𝑀Mitalic\_M molecule nodes {m}𝑚\{m\}{ italic\_m } and P𝑃Pitalic\_P property nodes {p}𝑝\{p\}{ italic\_p }.
Three types of edges indicate different relationships between molecules and properties.


Then we employ a GNN-based context encoder: 𝐂=ContextEncoder⁢(𝒱t,𝐀t,𝐗t)𝐂ContextEncodersubscript𝒱𝑡subscript𝐀𝑡subscript𝐗𝑡\mathbf{C}=\texttt{ContextEncoder}(\mathcal{V}\_{t},\mathbf{A}\_{t},\mathbf{X}\_{%
t})bold\_C = ContextEncoder ( caligraphic\_V start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT , bold\_A start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT , bold\_X start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT ),
where 𝐂∈ℝ(M+P)×d2𝐂superscriptℝ𝑀𝑃subscript𝑑2\mathbf{C}\in\mathbb{R}^{(M+P)\times d\_{2}}bold\_C ∈ blackboard\_R start\_POSTSUPERSCRIPT ( italic\_M + italic\_P ) × italic\_d start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT end\_POSTSUPERSCRIPT denotes the learned context representation matrix for Etsubscript𝐸𝑡E\_{t}italic\_E start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT. 𝒱tsubscript𝒱𝑡\mathcal{V}\_{t}caligraphic\_V start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT and 𝐀tsubscript𝐀𝑡\mathbf{A}\_{t}bold\_A start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT denote the node set and the adjacent matrix of the context graph, respectively, and 𝐗tsubscript𝐗𝑡\mathbf{X}\_{t}bold\_X start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT denotes the initial features of nodes.
The features of molecule nodes are initialized with a pre-trained molecular encoder. The property nodes are randomly initialized.
When we make the prediction of molecule m𝑚mitalic\_m’s target property p𝑝pitalic\_p, we take the learned representations of the this molecule 𝒄msubscript𝒄𝑚\boldsymbol{c}\_{m}bold\_italic\_c start\_POSTSUBSCRIPT italic\_m end\_POSTSUBSCRIPT and of the target property 𝒄psubscript𝒄𝑝\boldsymbol{c}\_{p}bold\_italic\_c start\_POSTSUBSCRIPT italic\_p end\_POSTSUBSCRIPT as the context vectors.
Details about the context encoder are provided in [Section F.2](https://arxiv.org/html/2411.01158v1#A6.SS2 "F.2 Model configuration ‣ Appendix F Implementation details ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").


In-context tuning with molecular context information.
After obtaining the context vectors, we consider enabling the molecular encoder to use the context as a condition, achieving conditional molecular encoding.
To achieve this, we further refine our adapter module.
While neural conditional encoding has been explored in some domains, such as cross-attention [[43](https://arxiv.org/html/2411.01158v1#bib.bib43)] and ControlNet [[68](https://arxiv.org/html/2411.01158v1#bib.bib68)] for conditional image generation, these methods often come with a significant increase in the number of parameters.
This contradicts our motivation of parameter-efficient tuning for few-shot tasks.
In this work, we adopt a simple yet effective method. We directly concatenate the context with the output of the message passing layer, and feed them into the downscaling feed-forward layer in the MP-Adapter. Formally, the downscaling process defined in [Eq. 3](https://arxiv.org/html/2411.01158v1#S4.E3 "In 4.1.1 MP-Adapter: message passing layer-oriented adapter ‣ 4.1 Parameter-efficient tuning for PMEs ‣ 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") is reformulated as:

|  | 𝒛(l)=FeedForwarddown⁢(𝒉v(l)⁢‖𝒄m‖⁢𝒄p),superscript𝒛𝑙subscriptFeedForwarddownsuperscriptsubscript𝒉𝑣𝑙normsubscript𝒄𝑚subscript𝒄𝑝\boldsymbol{z}^{(l)}=\texttt{FeedForward}\_{\texttt{down}}(\boldsymbol{h}\_{v}^{% (l)}\|\boldsymbol{c}\_{m}\|\boldsymbol{c}\_{p}),bold\_italic\_z start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT = FeedForward start\_POSTSUBSCRIPT down end\_POSTSUBSCRIPT ( bold\_italic\_h start\_POSTSUBSCRIPT italic\_v end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ( italic\_l ) end\_POSTSUPERSCRIPT ∥ bold\_italic\_c start\_POSTSUBSCRIPT italic\_m end\_POSTSUBSCRIPT ∥ bold\_italic\_c start\_POSTSUBSCRIPT italic\_p end\_POSTSUBSCRIPT ) , |  | (7) |
| --- | --- | --- | --- |

where ∥∥\|∥ denotes concatenation.
Such learned molecular representations are more easily predicted on specific properties, verified in [Section 5.5](https://arxiv.org/html/2411.01158v1#S5.SS5 "5.5 Case study ‣ 5 Experiments ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") and [Appendix G](https://arxiv.org/html/2411.01158v1#A7 "Appendix G More experimental results and discussions ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").


### 4.3 Optimization

Following MAML [[10](https://arxiv.org/html/2411.01158v1#bib.bib10)], a gradient descent strategy is adopted.
Firstly, B𝐵Bitalic\_B episodes {Et}t=1Bsuperscriptsubscriptsubscript𝐸𝑡𝑡1𝐵\{E\_{t}\}\_{t=1}^{B}{ italic\_E start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT } start\_POSTSUBSCRIPT italic\_t = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_B end\_POSTSUPERSCRIPT are randomly sampled.
For each episode, in the inner-loop optimization, the loss on the support set is computed as ℒt,𝒮c⁢l⁢s⁢(fθ)subscriptsuperscriptℒ𝑐𝑙𝑠

𝑡𝒮subscript𝑓𝜃\mathcal{L}^{cls}\_{t,\mathcal{S}}(f\_{\theta})caligraphic\_L start\_POSTSUPERSCRIPT italic\_c italic\_l italic\_s end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_t , caligraphic\_S end\_POSTSUBSCRIPT ( italic\_f start\_POSTSUBSCRIPT italic\_θ end\_POSTSUBSCRIPT ) and the parameters θ𝜃\thetaitalic\_θ are updated by gradient descent:

|  | ℒt,𝒮c⁢l⁢s⁢(fθ)subscriptsuperscriptℒ𝑐𝑙𝑠  𝑡𝒮subscript𝑓𝜃\displaystyle\mathcal{L}^{cls}\_{t,\mathcal{S}}(f\_{\theta})caligraphic\_L start\_POSTSUPERSCRIPT italic\_c italic\_l italic\_s end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_t , caligraphic\_S end\_POSTSUBSCRIPT ( italic\_f start\_POSTSUBSCRIPT italic\_θ end\_POSTSUBSCRIPT ) | =−∑𝒮t(y⁢log⁡(y^)+(1−y)⁢log⁡(1−y^)),absentsubscriptsubscript𝒮𝑡𝑦^𝑦1𝑦1^𝑦\displaystyle=-\sum\nolimits\_{\mathcal{S}\_{t}}(y\log(\hat{y})+(1-y)\log(1-\hat% {y})),= - ∑ start\_POSTSUBSCRIPT caligraphic\_S start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT end\_POSTSUBSCRIPT ( italic\_y roman\_log ( over^ start\_ARG italic\_y end\_ARG ) + ( 1 - italic\_y ) roman\_log ( 1 - over^ start\_ARG italic\_y end\_ARG ) ) , |  | (8) |
| --- | --- | --- | --- | --- |
|  |  | θ′←θ−αi⁢n⁢n⁢e⁢r⁢∇θℒt,𝒮c⁢l⁢s⁢(fθ),←superscript𝜃′𝜃subscript𝛼𝑖𝑛𝑛𝑒𝑟subscript∇𝜃subscriptsuperscriptℒ𝑐𝑙𝑠  𝑡𝒮subscript𝑓𝜃\displaystyle\theta^{\prime}\leftarrow\theta-\alpha\_{inner}\nabla\_{\theta}% \mathcal{L}^{cls}\_{t,\mathcal{S}}(f\_{\theta}),italic\_θ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ← italic\_θ - italic\_α start\_POSTSUBSCRIPT italic\_i italic\_n italic\_n italic\_e italic\_r end\_POSTSUBSCRIPT ∇ start\_POSTSUBSCRIPT italic\_θ end\_POSTSUBSCRIPT caligraphic\_L start\_POSTSUPERSCRIPT italic\_c italic\_l italic\_s end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_t , caligraphic\_S end\_POSTSUBSCRIPT ( italic\_f start\_POSTSUBSCRIPT italic\_θ end\_POSTSUBSCRIPT ) , |  | (9) |
| --- | --- | --- | --- | --- |

where αi⁢n⁢n⁢e⁢rsubscript𝛼𝑖𝑛𝑛𝑒𝑟\alpha\_{inner}italic\_α start\_POSTSUBSCRIPT italic\_i italic\_n italic\_n italic\_e italic\_r end\_POSTSUBSCRIPT is the learning rate.
In the outer loop, the classification loss of query set is denoted as ℒt,Qc⁢l⁢ssubscriptsuperscriptℒ𝑐𝑙𝑠

𝑡𝑄\mathcal{L}^{cls}\_{t,Q}caligraphic\_L start\_POSTSUPERSCRIPT italic\_c italic\_l italic\_s end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_t , italic\_Q end\_POSTSUBSCRIPT.
Together with our Emb-BWC regularizer, the meta-training loss ℒ⁢(fθ′)ℒsubscript𝑓superscript𝜃′\mathcal{L}(f\_{\theta^{\prime}})caligraphic\_L ( italic\_f start\_POSTSUBSCRIPT italic\_θ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT end\_POSTSUBSCRIPT ) is computed and we do an outer-loop optimization with learning rate αo⁢u⁢t⁢e⁢rsubscript𝛼𝑜𝑢𝑡𝑒𝑟\alpha\_{outer}italic\_α start\_POSTSUBSCRIPT italic\_o italic\_u italic\_t italic\_e italic\_r end\_POSTSUBSCRIPT across the mini-batch:

|  | ℒ⁢(fθ′)ℒsubscript𝑓superscript𝜃′\displaystyle\mathcal{L}(f\_{\theta^{\prime}})caligraphic\_L ( italic\_f start\_POSTSUBSCRIPT italic\_θ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT end\_POSTSUBSCRIPT ) | =1B⁢∑t=1Bℒt,𝒬c⁢l⁢s⁢(fθ′)+λ⁢ℒEmb-BWC,absent1𝐵superscriptsubscript𝑡1𝐵subscriptsuperscriptℒ𝑐𝑙𝑠  𝑡𝒬subscript𝑓superscript𝜃′𝜆subscriptℒEmb-BWC\displaystyle=\frac{1}{B}\sum\nolimits\_{t=1}^{B}\mathcal{L}^{cls}\_{t,\mathcal{% Q}}(f\_{\theta^{\prime}})+\lambda\mathcal{L}\_{\textrm{Emb-BWC}},= divide start\_ARG 1 end\_ARG start\_ARG italic\_B end\_ARG ∑ start\_POSTSUBSCRIPT italic\_t = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_B end\_POSTSUPERSCRIPT caligraphic\_L start\_POSTSUPERSCRIPT italic\_c italic\_l italic\_s end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_t , caligraphic\_Q end\_POSTSUBSCRIPT ( italic\_f start\_POSTSUBSCRIPT italic\_θ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT end\_POSTSUBSCRIPT ) + italic\_λ caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT , |  | (10) |
| --- | --- | --- | --- | --- |
|  |  | θ←θ−αo⁢u⁢t⁢e⁢r⁢∇θℒ⁢(fθ′),←𝜃𝜃subscript𝛼𝑜𝑢𝑡𝑒𝑟subscript∇𝜃ℒsubscript𝑓superscript𝜃′\displaystyle\theta\leftarrow\theta-\alpha\_{outer}\nabla\_{\theta}\mathcal{L}(f% \_{\theta^{\prime}}),italic\_θ ← italic\_θ - italic\_α start\_POSTSUBSCRIPT italic\_o italic\_u italic\_t italic\_e italic\_r end\_POSTSUBSCRIPT ∇ start\_POSTSUBSCRIPT italic\_θ end\_POSTSUBSCRIPT caligraphic\_L ( italic\_f start\_POSTSUBSCRIPT italic\_θ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT end\_POSTSUBSCRIPT ) , |  | (11) |
| --- | --- | --- | --- | --- |

where λ𝜆\lambdaitalic\_λ is the weight of Emb-BWC regularizer. The pseudo-code is provided in [Appendix B](https://arxiv.org/html/2411.01158v1#A2 "Appendix B Pseudo-code of training process ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").
We also provide more discussion of tunable parameter size and total model size in  [Appendix C](https://arxiv.org/html/2411.01158v1#A3 "Appendix C Discussion of tunable parameter size and total model size ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").


## 5 Experiments

### 5.1 Evaluation setups

Datasets. We use five common few-shot molecular property prediction datasets from the MoleculeNet [[61](https://arxiv.org/html/2411.01158v1#bib.bib61)]: Tox21, SIDER, MUV, ToxCast, and PCBA.
Standard data splits for FSMPP are adopted.
Dataset statistics and more details of datasets can be found in [Appendix D](https://arxiv.org/html/2411.01158v1#A4 "Appendix D Details of datasets ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").


Baselines. For a comprehensive comparison, we adopt two types of baselines: (1) methods with molecular encoders trained from scratch, including Siamese Network [[25](https://arxiv.org/html/2411.01158v1#bib.bib25)], ProtoNet [[46](https://arxiv.org/html/2411.01158v1#bib.bib46)], MAML [[10](https://arxiv.org/html/2411.01158v1#bib.bib10)], TPN [[35](https://arxiv.org/html/2411.01158v1#bib.bib35)], EGNN [[22](https://arxiv.org/html/2411.01158v1#bib.bib22)], and IterRefLSTM [[1](https://arxiv.org/html/2411.01158v1#bib.bib1)]; and (2) methods which leverage pre-trained molecular encoders, including Pre-GNN [[20](https://arxiv.org/html/2411.01158v1#bib.bib20)], Meta-MGNN [[14](https://arxiv.org/html/2411.01158v1#bib.bib14)], PAR [[58](https://arxiv.org/html/2411.01158v1#bib.bib58)], and GS-Meta [[73](https://arxiv.org/html/2411.01158v1#bib.bib73)]. More details about these baselines are in [Appendix E](https://arxiv.org/html/2411.01158v1#A5 "Appendix E Details of baselines ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").


Metrics. Following prior works [[1](https://arxiv.org/html/2411.01158v1#bib.bib1), [58](https://arxiv.org/html/2411.01158v1#bib.bib58)], ROC-AUC scores are calculated on the query set for each meta-testing task, to evaluate the performance of FSMPP. We run experiments 10 times with different random seeds and report the mean and standard deviations.


Table 1: ROC-AUC scores (%) on benchmark datasets, compared with methods trained from scratch (first group) and methods that leverage pre-trained molecular encoder (second group). The best is marked with boldface and the second best is with underline. ΔΔ\Deltaroman\_Δ*Improve.* indicates the relative improvements over the baseline models in percentage.


| Model | Tox21 | | SIDER | | MUV | | ToxCast | | PCBA | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10-shot | 5-shot | 10-shot | 5-shot | 10-shot | 5-shot | 10-shot | 5-shot | 10-shot | 5-shot |
| Siamese | 80.40(0.35) | - | 71.10(4.32) | - | 59.96(5.13) | - | - | - | - | - |
| ProtoNet | 74.98(0.32) | 72.78(3.93) | 64.54(0.89) | 64.09(2.37) | 65.88(4.11) | 64.86(2.31) | 68.87(0.43) | 66.26(1.49) | 64.93(1.94) | 62.29(2.12) |
| MAML | 80.21(0.24) | 69.17(1.34) | 70.43(0.76) | 60.92(0.65) | 63.90(2.28) | 63.00(0.61) | 68.30(0.59) | 67.56(1.53) | 66.22(1.31) | 65.25(0.75) |
| TPN | 76.05(0.24) | 75.45(0.95) | 67.84(0.95) | 66.52(1.28) | 65.22(5.82) | 65.13(0.23) | 69.47(0.71) | 66.04(1.14) | 67.61(0.33) | 63.66(1.64) |
| EGNN | 81.21(0.16) | 76.80(2.62) | 72.87(0.73) | 60.61(1.06) | 65.20(2.08) | 63.46(2.58) | 74.02(1.11) | 67.13(0.50) | 69.92(1.85) | 67.71(3.67) |
| IterRefLSTM | 81.10(0.17) | - | 69.63(0.31) | - | 49.56(5.12) | - | - | - | - | - |
| Pre-GNN | 82.14(0.08) | 82.04(0.30) | 73.96(0.08) | 76.76(0.53) | 67.14(1.58) | 70.23(1.40) | 75.31(0.95) | 74.43(0.47) | 76.79(0.45) | 75.27(0.49) |
| Meta-MGNN | 82.97(0.10) | 76.12(0.23) | 75.43(0.21) | 66.60(0.38) | 68.99(1.84) | 64.07(0.56) | 76.27(0.56) | 75.26(0.43) | 72.58(0.34) | 72.51(0.52) |
| PAR | 84.93(0.11) | 83.95(0.15) | 78.08(0.16) | 77.70(0.34) | 69.96(1.37) | 68.08(2.42) | 79.41(0.08) | 76.89(0.32) | 73.71(0.61) | 72.79(0.98) |
| GS-Meta | 86.67(0.41) | 86.43(0.02) | 84.36(0.54) | 84.57(0.01) | 66.08(1.25) | 64.50(0.20) | 83.81(0.16) | 82.65(0.35) | 79.40(0.43) | 77.47(0.29) |
| Pin-Tuning | 91.56(2.57) | 90.95(2.33) | 93.41(3.52) | 92.02(3.01) | 73.33(2.00) | 70.71(1.42) | 84.94(1.09) | 83.71(0.93) | 81.26(0.46) | 79.23(0.52) |
| ΔΔ\Deltaroman\_Δ*Improve.* | 5.64% | 5.23% | 10.73% | 8.81% | 4.82% | 3.86% | 1.35% | 1.28% | 2.34% | 2.27% |


### 5.2 Performance comparison

We compare Pin-Tuning with the baselines and the results are summarized in [Table 1](https://arxiv.org/html/2411.01158v1#S5.T1 "In 5.1 Evaluation setups ‣ 5 Experiments ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction"), [Table 7](https://arxiv.org/html/2411.01158v1#A6.T7 "In F.2 Model configuration ‣ Appendix F Implementation details ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction"), and [Table 8](https://arxiv.org/html/2411.01158v1#A6.T8 "In F.2 Model configuration ‣ Appendix F Implementation details ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").
Our method significantly outperforms all baseline models under both the 10-shot and 5-shot settings, demonstrating the effectiveness and superiority of our approach.


Across all datasets, our method provides greater improvement in the 10-shot scenario than in the 5-shot scenario. This is attributed to the molecular context constructed based on support molecules. When there are more molecules in the support set, the uncertainty in the context is reduced, providing more effective adaptation guidance for our parameter-efficient tuning.


Among benchmark datasets, our method shows significant improvement on the SIDER dataset, increasing by 10.73% in the 10-shot scenario and by 8.81% in the 5-shot scenario. We consider this is related to the relatively balanced ratio of positive to negative samples, as well as the absence of missing labels in the SIDER dataset ([Table 5](https://arxiv.org/html/2411.01158v1#A4.T5 "In Appendix D Details of datasets ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction")). A balanced and low-uncertainty distribution can better benefit addressing the FSMPP task from our method.


We also observe that the standard deviations of our method’s results under 10 seeds are slightly higher than that of baseline models. However, our worst-case results are still better than the best baseline model. For example, in 10-shot experiments on the Tox21 dataset, the performance of our method is 91.56±2.57plus-or-minus91.562.5791.56\pm 2.5791.56 ± 2.57.
However, our 10 runs yield specific results
with the worst-case ROC-AUC reaching 88.02, which is also better than the best baseline model GS-Meta’s result of 86.67±0.41plus-or-minus86.670.4186.67\pm 0.4186.67 ± 0.41. Therefore, a high standard deviation does not mean our method is inferior to baseline models.


### 5.3 Ablation study

For MP-Adapter, the main components consist of:
(i) bottleneck adapter module (Adapter), (ii) introducing molecular context to adatpers (Context), and
(iii) layer normalization (LayerNorm).
The results of ablation experiments are summarized in [Table 2](https://arxiv.org/html/2411.01158v1#S5.T2 "In 5.3 Ablation study ‣ 5 Experiments ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction"). The bottleneck adapter and the modeling of molecular context are the most critical, having the most significant impact on performance. Removing them leads to a noticeable decline, which underscores the importance of parameter-efficient tuning and context perceptiveness in FSMPP tasks.
Layer normalization is used to normalize the resulting representations, which is also important for improving the optimization effect and stability.


Table 2: Ablation analysis on the MP-Adapter, in which we drop different components to form variants. We report ROC-AUC scores (%), and the best performance is highlighted in bold.


| Model | Component | | | Tox21 | | SIDER | | MUV | | ToxCast | | PCBA | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adapter | Context | LayerNorm | 10-shot | 5-shot | 10-shot | 5-shot | 10-shot | 5-shot | 10-shot | 5-shot | 10-shot | 5-shot |
| Pin-Tuning | ✓ | ✓ | ✓ | 91.56 | 90.95 | 93.41 | 92.02 | 73.33 | 70.71 | 84.94 | 83.71 | 81.26 | 79.23 |
| w/o Adapter | - | - | ✓ | 79.72 | 78.49 | 74.04 | 72.94 | 66.06 | 62.88 | 80.06 | 78.70 | 73.85 | 72.02 |
| w/o Context | ✓ | - | ✓ | 81.42 | 79.34 | 74.68 | 72.86 | 68.70 | 66.12 | 81.49 | 79.85 | 74.69 | 72.46 |
| w/o LayerNorm | ✓ | ✓ | - | 86.71 | 84.93 | 91.50 | 90.76 | 70.26 | 67.42 | 83.52 | 82.55 | 80.07 | 78.23 |


Table 3: Ablation analysis on the Emb-BWC.


| Fine-tune | Regularizer | Tox21 | SIDER | MUV | PCBA |
| --- | --- | --- | --- | --- | --- |
| - | - | 89.70 | 90.12 | 70.76 | 80.24 |
| ✓ | - | 90.17 | 92.06 | 72.37 | 80.74 |
| ✓ | ℒEmb-BWCIMsuperscriptsubscriptℒEmb-BWCIM\mathcal{L}\_{\textrm{Emb-BWC}}^{\textrm{IM}}caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT IM end\_POSTSUPERSCRIPT | 91.56 | 93.41 | 73.22 | 81.26 |
| ✓ | ℒEmb-BWCFIMsuperscriptsubscriptℒEmb-BWCFIM\mathcal{L}\_{\textrm{Emb-BWC}}^{\textrm{FIM}}caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT FIM end\_POSTSUPERSCRIPT | 90.93 | 90.09 | 72.17 | 80.78 |
| ✓ | ℒEmb-BWCEFIMsuperscriptsubscriptℒEmb-BWCEFIM\mathcal{L}\_{\textrm{Emb-BWC}}^{\textrm{EFIM}}caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT EFIM end\_POSTSUPERSCRIPT | 91.32 | 90.31 | 72.78 | 81.22 |


For Emb-BWC, we verify the effectiveness of fine-tuning the embedding layers and regularizing them with different approximations of ℒEmb-BWCsubscriptℒEmb-BWC\mathcal{L}\_{\textrm{Emb-BWC}}caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT ([Table 3](https://arxiv.org/html/2411.01158v1#S5.T3 "In 5.3 Ablation study ‣ 5 Experiments ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction")). Since the embedding layers have relatively few parameters, direct fine-tuning can also enhance performance. Applying our proposed regularizers to fine-tuning can further improve the effects. Among the three regularizers, the ℒEmb-BWCIMsuperscriptsubscriptℒEmb-BWCIM\mathcal{L}\_{\textrm{Emb-BWC}}^{\textrm{IM}}caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT IM end\_POSTSUPERSCRIPT is the most effective. This indicates that keeping pre-trained parameters to some extent can better utilize pre-trained knowledge, but the parameters worth keeping in fine-tuning and the important parameters in pre-training revealed by Fisher information matrix are not completely consistent.


### 5.4 Sensitivity analysis

![Refer to caption](09_Pin-Tuning_images/x4.png)

Figure 4: Effect of different hyper-parameters. The y-axis represents ROC-AUC scores (%) and the x-axis is the different hyper-parameters.


![Refer to caption](09_Pin-Tuning_images/x5.png)

Figure 5: ROC-AUC (%) and number of trainable parameters of Pin-Tuning with varied value of d2subscript𝑑2d\_{2}italic\_d start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT and full Fine-Tuning method (e.g., GS-Meta) on the Tox21 dataset.


Effect of weight of Emb-BWC regularizer λ𝜆\lambdaitalic\_λ.
Emb-BWC is applied on the embedding layers to limit the magnitude of parameter updates during fine-tuning.
We vary the weight of this regularization λ𝜆\lambdaitalic\_λ from {0.01,0.1,1,10}0.010.1110\{0.01,0.1,1,10\}{ 0.01 , 0.1 , 1 , 10 }. The first subfigure in [Figure 5](https://arxiv.org/html/2411.01158v1#S5.F5 "In 5.4 Sensitivity analysis ‣ 5 Experiments ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") shows that the performance is best when λ=0.1𝜆0.1\lambda=0.1italic\_λ = 0.1 or 1111. When λ𝜆\lambdaitalic\_λ is too small, the parameters undergo too large updates on few-shot downstream datasets, leading to over-fitting and ineffectively utilizing the pre-trained knowledge. Too large λ𝜆\lambdaitalic\_λ causes the parameters of the embedding layers to be nearly frozen, which prevents effective adaptation.


Effect of hidden dimension of MP-Adapter d2subscript𝑑2d\_{2}italic\_d start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT.
The results corresponding to different values of d2subscript𝑑2d\_{2}italic\_d start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT from {25,50,75,100,150}255075100150\{25,50,75,100,150\}{ 25 , 50 , 75 , 100 , 150 } are presented in the second subfigure of [Figure 5](https://arxiv.org/html/2411.01158v1#S5.F5 "In 5.4 Sensitivity analysis ‣ 5 Experiments ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").
On the Tox21 dataset, we further analyze the impact of this hyper-parameter on the number of trainable parameters. As shown in [Figure 5](https://arxiv.org/html/2411.01158v1#S5.F5 "In 5.4 Sensitivity analysis ‣ 5 Experiments ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction"), the number of parameters that our method needs to train is significantly less than that required by the full fine-tuning method, such as GS-Meta, while our method also performs better in terms of ROC-AUC performance due to solving over-fitting and context perceptiveness issues. When d=50𝑑50d=50italic\_d = 50, Pin-Tuning performs best on Tox21, and the number of parameters that need to train is only 14.2% of that required by traditional fine-tuning methods.


### 5.5 Case study

![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_GS-Meta-SR-MMP_markersize24.png)


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_GS-Meta-SR-p53_markersize24.png)


Figure 6: Molecular representations encoded by GS-Meta [[73](https://arxiv.org/html/2411.01158v1#bib.bib73)].


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_Pin-Tuning-SR-MMP_markersize24.png)


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_Pin-Tuning-SR-p53_markersize24.png)


Figure 7: Molecular representations encoded by Pin-Tuning.


We visualized the molecular representations learned by the GS-Meta and our Pin-Tuning’s encoders in the 10-shot setting, respectively. As shown in [Figure 7](https://arxiv.org/html/2411.01158v1#S5.F7 "In 5.5 Case study ‣ 5 Experiments ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") and [7](https://arxiv.org/html/2411.01158v1#S5.F7 "Figure 7 ‣ 5.5 Case study ‣ 5 Experiments ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction"), Pin-Tuning can effectively adapt to different downstream tasks based on context information, generating property-specific molecular representations. Across different tasks, our method is more effective in encoding representations that facilitate the prediction of the current property, reducing the difficulty of property prediction from the encoding representation aspect.
More case studies are provided in [Appendix G](https://arxiv.org/html/2411.01158v1#A7 "Appendix G More experimental results and discussions ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").


## 6 Conclusion

In this work, we propose a tuning method, Pin-Tuning, to address the ineffective fine-tuning of pre-trained molecular encoders in FSMPP tasks.
Through the innovative parameter-efficient tuning and in-context tuning for pre-trained molecular encoders, our approach not only mitigates the issues of parameter-data imbalance but also enhances contextual perceptiveness.
The promising results on public datasets underscore the potential of Pin-Tuning to advance this field, offering valuable insights for future research in drug discovery and material science.


## Acknowledgments

This work is jointly supported by National Science and Technology Major Project (2023ZD0120901), National Natural Science Foundation of China (62372454, 62236010) and the Excellent Youth Program of State Key Laboratory of Multimodal Artificial Intelligence Systems.


## References

- Altae-Tran et al. [2017]
  Han Altae-Tran, Bharath Ramsundar, Aneesh S Pappu, and Vijay Pande.
  
  Low data drug discovery with one-shot learning.
  
  *ACS Central Science*, 3(4):283–293, 2017.
- Chen et al. [2023a]
  Dingshuo Chen, Yanqiao Zhu, Jieyu Zhang, Yuanqi Du, Zhixun Li, Q. Liu, Shu Wu, and Liang Wang.
  
  Uncovering neural scaling laws in molecular representation learning.
  
  In *NeurIPS*, 2023a.
- Chen and Garner [2024]
  Haolin Chen and Philip N. Garner.
  
  Bayesian parameter-efficient fine-tuning for overcoming catastrophic forgetting.
  
  *arXiv*, abs/2402.12220, 2024.
- Chen et al. [2023b]
  Wenlin Chen, Austin Tripp, and José Miguel Hernández-Lobato.
  
  Meta-learning adaptive deep kernel gaussian processes for molecular property prediction.
  
  In *ICLR*, 2023b.
- Deng et al. [2023]
  Jianyuan Deng, Zhibo Yang, Hehe Wang, Iwao Ojima, Dimitris Samaras, and Fusheng Wang.
  
  A systematic study of key elements underlying molecular property prediction.
  
  *Nature Communications*, 14, 2023.
- Ding et al. [2022]
  Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, Jing Yi, Weilin Zhao, Xiaozhi Wang, Zhiyuan Liu, Hai-Tao Zheng, Jianfei Chen, Yang Liu, Jie Tang, Juanzi Li, and Maosong Sun.
  
  Delta tuning: A comprehensive study of parameter efficient methods for pre-trained language models.
  
  *arXiv*, abs/2203.06904, 2022.
- Ding et al. [2023]
  Ning Ding, Yujia Qin, Guang Yang, Fu Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, Jing Yi, Weilin Zhao, Xiaozhi Wang, Zhiyuan Liu, Haitao Zheng, Jianfei Chen, Y. Liu, Jie Tang, Juanzi Li, and Maosong Sun.
  
  Parameter-efficient fine-tuning of large-scale pre-trained language models.
  
  *Nature Machine Intelligence*, 5:220–235, 2023.
- Dong et al. [2023]
  Wei Dong, Dawei Yan, Zhijun Lin, and Peng Wang.
  
  Efficient adaptation of large vision transformer via adapter re-composing.
  
  2023.
- Fang et al. [2021]
  Xiaomin Fang, Lihang Liu, Jieqiong Lei, Donglong He, Shanzhuo Zhang, Jingbo Zhou, Fan Wang, Hua Wu, and Haifeng Wang.
  
  Geometry-enhanced molecular representation learning for property prediction.
  
  *Nature Machine Intelligence*, 4:127 – 134, 2021.
- Finn et al. [2017]
  Chelsea Finn, Pieter Abbeel, and Sergey Levine.
  
  Model-agnostic meta-learning for fast adaptation of deep networks.
  
  In *ICML*, 2017.
- Gilmer et al. [2017]
  Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl.
  
  Neural message passing for quantum chemistry.
  
  In *ICML*, 2017.
- Godwin et al. [2022]
  Jonathan Godwin, Michael Schaarschmidt, Alexander L. Gaunt, Alvaro Sanchez-Gonzalez, Yulia Rubanova, Petar Velickovic, James Kirkpatrick, and Peter W. Battaglia.
  
  Simple GNN regularisation for 3d molecular property prediction and beyond.
  
  In *ICLR*, 2022.
- Guan et al. [2024]
  Renxiang Guan, Zihao Li, Wenxuan Tu, Jun Wang, Yue Liu, Xianju Li, Chang Tang, and Ruyi Feng.
  
  Contrastive multiview subspace clustering of hyperspectral images based on graph convolutional networks.
  
  *IEEE Transactions on Geoscience and Remote Sensing*, 2024.
- Guo et al. [2021]
  Zhichun Guo, Chuxu Zhang, Wenhao Yu, John Herr, Olaf Wiest, Meng Jiang, and Nitesh V. Chawla.
  
  Few-shot graph learning for molecular property prediction.
  
  In *WWW*, 2021.
- He et al. [2022]
  Junxian He, Chunting Zhou, Xuezhe Ma, Taylor Berg-Kirkpatrick, and Graham Neubig.
  
  Towards a unified view of parameter-efficient transfer learning.
  
  In *ICLR*, 2022.
- Hospedales et al. [2020]
  Timothy M. Hospedales, Antreas Antoniou, Paul Micaelli, and Amos J. Storkey.
  
  Meta-learning in neural networks: A survey.
  
  *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 44:5149–5169, 2020.
- Hou et al. [2022]
  Zhenyu Hou, Xiao Liu, Yukuo Cen, Yuxiao Dong, Hongxia Yang, Chunjie Wang, and Jie Tang.
  
  Graphmae: Self-supervised masked graph autoencoders.
  
  In *KDD*, 2022.
- Houlsby et al. [2019]
  Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly.
  
  Parameter-efficient transfer learning for NLP.
  
  In *ICML*, 2019.
- Hu et al. [2022]
  Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.
  
  Lora: Low-rank adaptation of large language models.
  
  In *ICLR*, 2022.
- Hu et al. [2020]
  Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay S. Pande, and Jure Leskovec.
  
  Strategies for pre-training graph neural networks.
  
  In *ICLR*, 2020.
- Kim et al. [2022]
  Dongki Kim, Jinheon Baek, and Sung Ju Hwang.
  
  Graph self-supervised learning with accurate discrepancy learning.
  
  In *NeurIPS*, 2022.
- Kim et al. [2019]
  Jongmin Kim, Taesup Kim, Sungwoong Kim, and Chang D. Yoo.
  
  Edge-labeling graph neural network for few-shot learning.
  
  In *CVPR*, 2019.
- Kim et al. [2023]
  Suyeon Kim, Dongha Lee, SeongKu Kang, Seonghyeon Lee, and Hwanjo Yu.
  
  Learning topology-specific experts for molecular property prediction.
  
  In *AAAI*, pages 8291–8299, 2023.
- Kirkpatrick et al. [117]
  James Kirkpatrick, Razvan Pascanu, Neil C. Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell.
  
  Overcoming catastrophic forgetting in neural networks.
  
  *Proceedings of the National Academy of Sciences*, 114, 117.
- Koch et al. [2015]
  Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov.
  
  Siamese neural networks for one-shot image recognition.
  
  In *ICML deep learning workshop*, 2015.
- Lester et al. [2021]
  Brian Lester, Rami Al-Rfou, and Noah Constant.
  
  The power of scale for parameter-efficient prompt tuning.
  
  In *EMNLP (1)*, pages 3045–3059. Association for Computational Linguistics, 2021.
- Li et al. [2022a]
  Han Li, Dan Zhao, and Jianyang Zeng.
  
  KPGT: knowledge-guided pre-training of graph transformer for molecular property prediction.
  
  In *KDD*, pages 857–867, 2022a.
- Li et al. [2022b]
  Shuangli Li, Jingbo Zhou, Tong Xu, Dejing Dou, and Hui Xiong.
  
  Geomgcl: Geometric graph contrastive learning for molecular property prediction.
  
  In *AAAI*, pages 4541–4549, 2022b.
- Li and Liang [2021]
  Xiang Lisa Li and Percy Liang.
  
  Prefix-tuning: Optimizing continuous prompts for generation.
  
  In *ACL*, 2021.
- Li et al. [2023a]
  Xin Li, Dongze Lian, Zhihe Lu, Jiawang Bai, Zhibo Chen, and Xinchao Wang.
  
  Graphadapter: Tuning vision-language models with dual knowledge graph.
  
  In *NeurIPS*, 2023a.
- Li et al. [2018]
  Xuhong Li, Yves Grandvalet, and Franck Davoine.
  
  Explicit inductive bias for transfer learning with convolutional networks.
  
  In *ICML*, 2018.
- Li et al. [2023b]
  Zhixun Li, Liang Wang, Xin Sun, Yifan Luo, Yanqiao Zhu, Dingshuo Chen, Yingtao Luo, Xiangxin Zhou, Qiang Liu, Shu Wu, Liang Wang, and Jeffrey Xu Yu.
  
  GSLB: the graph structure learning benchmark.
  
  In *NeurIPS*, 2023b.
- Lialin et al. [2023]
  Vladislav Lialin, Vijeta Deshpande, and Anna Rumshisky.
  
  Scaling down to scale up: A guide to parameter-efficient fine-tuning.
  
  *arXiv*, abs/2303.15647, 2023.
- Liu et al. [2021]
  Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang.
  
  GPT understands, too.
  
  *arXiv*, abs/2103.10385, 2021.
- Liu et al. [2019]
  Yanbin Liu, Juho Lee, Minseop Park, Saehoon Kim, Eunho Yang, Sung Ju Hwang, and Yi Yang.
  
  Learning to propagate labels: Transductive propagation network for few-shot learning.
  
  In *ICLR*, 2019.
- Liu et al. [2022]
  Yue Liu, Jun Xia, Sihang Zhou, Siwei Wang, Xifeng Guo, Xihong Yang, Ke Liang, Wenxuan Tu, Stan Z. Li, and Xinwang Liu.
  
  A survey of deep graph clustering: Taxonomy, challenge, and application.
  
  *arXiv*, abs/2211.12875, 2022.
- Liu et al. [2023a]
  Yue Liu, Ke Liang, Jun Xia, Sihang Zhou, Xihong Yang, Xinwang Liu, and Stan Z. Li.
  
  Dink-net: Neural clustering on large graphs.
  
  In *ICML*, 2023a.
- Liu et al. [2023b]
  Zhiyuan Liu, Yaorui Shi, An Zhang, Enzhi Zhang, Kenji Kawaguchi, Xiang Wang, and Tat-Seng Chua.
  
  Rethinking tokenizer and decoder in masked graph modeling for molecules.
  
  In *NeurIPS*, 2023b.
- Lv et al. [2023]
  Qiujie Lv, Guanxing Chen, Ziduo Yang, Weihe Zhong, and Calvin Yu-Chian Chen.
  
  Meta learning with graph attention networks for low-data drug discovery.
  
  *IEEE transactions on neural networks and learning systems*, 2023.
- Meng et al. [2023]
  Ziqiao Meng, Yaoman Li, Peilin Zhao, Yang Yu, and Irwin King.
  
  Meta-learning with motif-based task augmentation for few-shot molecular property prediction.
  
  In *SDM*, 2023.
- Nguyen et al. [2020]
  Cuong Q. Nguyen, Constantine Kreatsoulas, and Kim Branson.
  
  Meta-learning gnn initializations for low-resource molecular property prediction.
  
  In *ICML Workshop on Graph Representation Learning and Beyond*, 2020.
- Pfeiffer et al. [2021]
  Jonas Pfeiffer, Aishwarya Kamath, Andreas Rücklé, Kyunghyun Cho, and Iryna Gurevych.
  
  Adapterfusion: Non-destructive task composition for transfer learning.
  
  In *EACL*, 2021.
- Rombach et al. [2022]
  Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.
  
  High-resolution image synthesis with latent diffusion models.
  
  In *CVPR*, pages 10674–10685. IEEE, 2022.
- Rong et al. [2020]
  Yu Rong, Yatao Bian, Tingyang Xu, Weiyang Xie, Ying Wei, Wenbing Huang, and Junzhou Huang.
  
  Self-supervised graph transformer on large-scale molecular data.
  
  In *NeurIPS*, 2020.
- Schimunek et al. [2023]
  Johannes Schimunek, Philipp Seidl, Lukas Friedrich, Daniel Kuhn, Friedrich Rippmann, Sepp Hochreiter, and Günter Klambauer.
  
  Context-enriched molecule representations improve few-shot drug discovery.
  
  In *ICLR*, 2023.
- Snell et al. [2017]
  Jake Snell, Kevin Swersky, and Richard S. Zemel.
  
  Prototypical networks for few-shot learning.
  
  In *NeurIPS*, 2017.
- Song et al. [2020]
  Ying Song, Shuangjia Zheng, Zhangming Niu, Zhang-Hua Fu, Yutong Lu, and Yuedong Yang.
  
  Communicative representation learning on attributed molecular graphs.
  
  In *IJCAI*, 2020.
- Song et al. [2023]
  Yisheng Song, Ting Wang, Puyu Cai, Subrota K. Mondal, and Jyoti Prakash Sahoo.
  
  A comprehensive survey of few-shot learning: Evolution, applications, challenges, and opportunities.
  
  *ACM Computing Surveys*, 2023.
- Stanley et al. [2021]
  Megan Stanley, John Bronskill, Krzysztof Maziarz, Hubert Misztela, Jessica Lanini, Marwin H. S. Segler, Nadine Schneider, and Marc Brockschmidt.
  
  Fs-mol: A few-shot learning dataset of molecules.
  
  In *NeurIPS Datasets and Benchmarks*, 2021.
- Stärk et al. [2022]
  Hannes Stärk, Dominique Beaini, Gabriele Corso, Prudencio Tossou, Christian Dallago, Stephan Günnemann, and Pietro Lió.
  
  3d infomax improves gnns for molecular property prediction.
  
  In *ICML*, pages 20479–20502, 2022.
- Sun et al. [2021]
  Mengying Sun, Jing Xing, Huijun Wang, Bin Chen, and Jiayu Zhou.
  
  Mocl: Data-driven molecular fingerprint via knowledge-aware contrastive learning from molecular graph.
  
  In *KDD*, 2021.
- Sun et al. [2024]
  Xin Sun, Liang Wang, Qiang Liu, Shu Wu, Zilei Wang, and Liang Wang.
  
  DIVE: subgraph disagreement for graph out-of-distribution generalization.
  
  In *KDD*, 2024.
- Vinyals et al. [2016]
  Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra.
  
  Matching networks for one shot learning.
  
  In *NeurIPS*, page 3637–3645, 2016.
- Wang et al. [2024a]
  Liang Wang, Xiang Tao, Qiang Liu, Shu Wu, and Liang Wang.
  
  Rethinking graph masked autoencoders through alignment and uniformity.
  
  In *AAAI*, 2024a.
- Wang et al. [2024b]
  Liang Wang, Shu Wu, Q. Liu, Yanqiao Zhu, Xiang Tao, and Mengdi Zhang.
  
  Bi-level graph structure learning for next poi recommendation.
  
  *IEEE Transactions on Knowledge and Data Engineering*, 36:5695–5708, 2024b.
- Wang et al. [2023]
  Xu Wang, Huan Zhao, Wei-Wei Tu, and Quanming Yao.
  
  Automated 3d pre-training for molecular property prediction.
  
  In *KDD*, pages 2419–2430, 2023.
- Wang et al. [2020]
  Yaqing Wang, Quanming Yao, James T. Kwok, and Lionel M. Ni.
  
  Generalizing from a few examples: A survey on few-shot learning.
  
  *ACM Computing Surveys*, 2020.
- Wang et al. [2021]
  Yaqing Wang, Abulikemu Abuduweili, Quanming Yao, and Dejing Dou.
  
  Property-aware relation networks for few-shot molecular property prediction.
  
  In *NeurIPS*, 2021.
- Wang et al. [2022a]
  Yaqing Wang, Sahaj Agarwal, Subhabrata Mukherjee, Xiaodong Liu, Jing Gao, Ahmed Hassan Awadallah, and Jianfeng Gao.
  
  Adamix: Mixture-of-adaptations for parameter-efficient model tuning.
  
  In *EMNLP*, 2022a.
- Wang et al. [2022b]
  Yuyang Wang, Jianren Wang, Zhonglin Cao, and Amir Barati Farimani.
  
  Molecular contrastive learning of representations via graph neural networks.
  
  *Nature Machine Intelligence*, 2022b.
- Wu et al. [2017]
  Zhenqin Wu, Bharath Ramsundar, Evan N. Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S. Pappu, Karl Leswing, and Vijay S. Pande.
  
  Moleculenet: A benchmark for molecular machine learning.
  
  *arXiv*, abs/1703.00564, 2017.
- Xia et al. [2023]
  Jun Xia, Chengshuai Zhao, Bozhen Hu, Zhangyang Gao, Cheng Tan, Yue Liu, Siyuan Li, and Stan Z. Li.
  
  Mole-bert: Rethinking pre-training graph neural networks for molecules.
  
  In *ICLR*, 2023.
- Xiao et al. [2024]
  Yi Xiao, Xiangxin Zhou, Qiang Liu, and Liang Wang.
  
  Bridging text and molecule: A survey on multimodal frameworks for molecule.
  
  *arXiv*, abs/2403.13830, 2024.
- Xie et al. [2022]
  Yaochen Xie, Zhao Xu, Jingtun Zhang, Zhengyang Wang, and Shuiwang Ji.
  
  Self-supervised learning of graph neural networks: A unified review.
  
  *TPAMI*, 2022.
- Xu et al. [2019]
  Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka.
  
  How powerful are graph neural networks?
  
  In *ICLR*, 2019.
- Ying et al. [2021]
  Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, and Tie-Yan Liu.
  
  Do transformers really perform badly for graph representation?
  
  In *NeurIPS*, 2021.
- Zaidi et al. [2023]
  Sheheryar Zaidi, Michael Schaarschmidt, James Martens, Hyunjik Kim, Yee Whye Teh, Alvaro Sanchez-Gonzalez, Peter W. Battaglia, Razvan Pascanu, and Jonathan Godwin.
  
  Pre-training via denoising for molecular property prediction.
  
  In *ICLR*, 2023.
- Zhang et al. [2023a]
  Lvmin Zhang, Anyi Rao, and Maneesh Agrawala.
  
  Adding conditional control to text-to-image diffusion models.
  
  In *ICCV*, pages 3813–3824. IEEE, 2023a.
- Zhang et al. [2023b]
  Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao.
  
  Adaptive budget allocation for parameter-efficient fine-tuning.
  
  In *ICLR*, 2023b.
- Zhang et al. [2024]
  Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Jiao Qiao.
  
  Llama-adapter: Efficient fine-tuning of language models with zero-init attention.
  
  In *ICLR*, 2024.
- Zhang et al. [2021]
  Zaixi Zhang, Qi Liu, Hao Wang, Chengqiang Lu, and Chee-Kong Lee.
  
  Motif-based graph self-supervised learning for molecular property prediction.
  
  In *NeurIPS*, pages 15870–15882, 2021.
- Zhu et al. [2024]
  Yanqiao Zhu, Dingshuo Chen, Yuanqi Du, Yingze Wang, Q. Liu, and Shu Wu.
  
  Molecular contrastive pretraining with collaborative featurizations.
  
  *Journal of Chemical Information and Modeling*, 2024.
- Zhuang et al. [2023]
  Xiang Zhuang, Qiang Zhang, Bin Wu, Keyan Ding, Yin Fang, and Huajun Chen.
  
  Graph sampling-based meta-learning for molecular property prediction.
  
  In *IJCAI*, 2023.


Appendix


The organization of the appendix is as follows:

- •
  
  Appendix A: Derivation of Emb-BWC regularization;
- •
  
  Appendix B: Pseudo-code of training process;
- •
  
  Appendix C: Discussion of tunable parameter size and total model size;
- •
  
  Appendix D: Details of datasets;
- •
  
  Appendix E: Details of baselines;
- •
  
  Appendix F: Implementation details;
- •
  
  Appendix G: More experimental results and discussions;
- •
  
  Appendix H: Limitations and future directions.

## Appendix A Derivation of Emb-BWC regularization

### A.1 Derivation of ℒEmb-BWCsubscriptℒEmb-BWC\mathcal{L}\_{\textrm{Emb-BWC}}caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT

Let Φ∈ℝE×dΦsuperscriptℝ𝐸𝑑\Phi\in\mathbb{R}^{E\times d}roman\_Φ ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_E × italic\_d end\_POSTSUPERSCRIPT be the pre-trained embeddings before fine-tuning, and Φ′∈ℝE×dsuperscriptΦ′superscriptℝ𝐸𝑑\Phi^{\prime}\in\mathbb{R}^{E\times d}roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_E × italic\_d end\_POSTSUPERSCRIPT be the fine-tuned embeddings.
Further, Φi∈ℝdsubscriptΦ𝑖superscriptℝ𝑑\Phi\_{i}\in\mathbb{R}^{d}roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT denotes the i𝑖iitalic\_i-th row’s embedding vector in ΦΦ\Phiroman\_Φ, and Φi,j∈ℝsubscriptΦ

𝑖𝑗ℝ\Phi\_{i,j}\in\mathbb{R}roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT ∈ blackboard\_R represents the j𝑗jitalic\_j-th dimensional value of ΦisubscriptΦ𝑖\Phi\_{i}roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT.


The optimization of embedding layers can be interpreted as performing a maximum a posterior (MAP) estimation of the parameters Φ′superscriptΦ′\Phi^{\prime}roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT given the pre-training data and training data of downstream FSMPP task, which is formulated in a Bayesian framework.


In the FSMPP setting, the molecular encoder has been pre-trained on the pre-training task 𝒫𝒫\mathcal{P}caligraphic\_P using data 𝒟𝒫subscript𝒟𝒫\mathcal{D}\_{\mathcal{P}}caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT, and is then fine-tuned on a downstream FSMPP task ℱℱ\mathcal{F}caligraphic\_F using data 𝒟ℱsubscript𝒟ℱ\mathcal{D}\_{\mathcal{F}}caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT.
The overall objective is to find the optimal parameters on task ℱℱ\mathcal{F}caligraphic\_F while preserving the prior knowledge obtained in pre-training on task ℱℱ\mathcal{F}caligraphic\_F.
Based on a prior p⁢(Φ′)𝑝superscriptΦ′p(\Phi^{\prime})italic\_p ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ) of the embedding parameters, the posterior after observing the FSMPP task ℱℱ\mathcal{F}caligraphic\_F can be computed with Bayes’ rule:

|  | p⁢(Φ′|𝒟𝒫,𝒟ℱ)𝑝conditionalsuperscriptΦ′  subscript𝒟𝒫subscript𝒟ℱ\displaystyle p(\Phi^{\prime}|\mathcal{D}\_{\mathcal{P}},\mathcal{D}\_{\mathcal{% F}})italic\_p ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT | caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT ) | =p⁢(𝒟ℱ|Φ′,𝒟𝒫)⁢p⁢(Φ′|𝒟𝒫)p⁢(𝒟ℱ|𝒟𝒫)absent𝑝conditionalsubscript𝒟ℱ  superscriptΦ′subscript𝒟𝒫𝑝conditionalsuperscriptΦ′subscript𝒟𝒫𝑝conditionalsubscript𝒟ℱsubscript𝒟𝒫\displaystyle=\frac{p(\mathcal{D}\_{\mathcal{F}}|\Phi^{\prime},\mathcal{D}\_{% \mathcal{P}})p(\Phi^{\prime}|\mathcal{D}\_{\mathcal{P}})}{p(\mathcal{D}\_{% \mathcal{F}}|\mathcal{D}\_{\mathcal{P}})}= divide start\_ARG italic\_p ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT | roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT , caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT ) italic\_p ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT | caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT ) end\_ARG start\_ARG italic\_p ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT | caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT ) end\_ARG |  | (12) |
| --- | --- | --- | --- | --- |
|  |  | =p⁢(𝒟ℱ|Φ′)⁢p⁢(Φ′|𝒟𝒫)p⁢(𝒟ℱ),absent𝑝conditionalsubscript𝒟ℱsuperscriptΦ′𝑝conditionalsuperscriptΦ′subscript𝒟𝒫𝑝subscript𝒟ℱ\displaystyle=\frac{p(\mathcal{D}\_{\mathcal{F}}|\Phi^{\prime})p(\Phi^{\prime}|% \mathcal{D\_{\mathcal{P}}})}{p(\mathcal{D}\_{\mathcal{F}})},= divide start\_ARG italic\_p ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT | roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ) italic\_p ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT | caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT ) end\_ARG start\_ARG italic\_p ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT ) end\_ARG , |  |

where 𝒟ℱsubscript𝒟ℱ\mathcal{D}\_{\mathcal{F}}caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT is assumed to be independent of 𝒟𝒫subscript𝒟𝒫\mathcal{D}\_{\mathcal{P}}caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT. Taking a logarithm of the posterior, the MAP objective is therefore:

|  | Φ′⁣∗superscriptΦ  ′\displaystyle\Phi^{\prime\*}roman\_Φ start\_POSTSUPERSCRIPT ′ ∗ end\_POSTSUPERSCRIPT | =arg⁢maxΦ′⁢log⁡p⁢(Φ′|𝒟𝒫,𝒟ℱ)absentsuperscriptΦ′argmax𝑝conditionalsuperscriptΦ′  subscript𝒟𝒫subscript𝒟ℱ\displaystyle=\underset{\Phi^{\prime}}{\operatorname\*{arg\,max}}\log p(\Phi^{% \prime}|\mathcal{D}\_{\mathcal{P}},\mathcal{D}\_{\mathcal{F}})= start\_UNDERACCENT roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT end\_UNDERACCENT start\_ARG roman\_arg roman\_max end\_ARG roman\_log italic\_p ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT | caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT ) |  | (13) |
| --- | --- | --- | --- | --- |
|  |  | =arg⁢maxΦ′⁢log⁡p⁢(𝒟ℱ|Φ′)+log⁡p⁢(Φ′|𝒟𝒫).absentsuperscriptΦ′argmax𝑝conditionalsubscript𝒟ℱsuperscriptΦ′𝑝conditionalsuperscriptΦ′subscript𝒟𝒫\displaystyle=\underset{\Phi^{\prime}}{\operatorname\*{arg\,max}}\log p(% \mathcal{D}\_{\mathcal{F}}|\Phi^{\prime})+\log p(\Phi^{\prime}|\mathcal{D}\_{% \mathcal{P}}).= start\_UNDERACCENT roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT end\_UNDERACCENT start\_ARG roman\_arg roman\_max end\_ARG roman\_log italic\_p ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT | roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ) + roman\_log italic\_p ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT | caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT ) . |  |

The first term log⁡p⁢(𝒟ℱ|Φ′)𝑝conditionalsubscript𝒟ℱsuperscriptΦ′\log p(\mathcal{D}\_{\mathcal{F}}|\Phi^{\prime})roman\_log italic\_p ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT | roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ) is the log likelihood of the data 𝒟ℱsubscript𝒟ℱ\mathcal{D}\_{\mathcal{F}}caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT given the parameters Φ′superscriptΦ′\Phi^{\prime}roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT, which can be expressed as the training loss function on task ℱ=−log⁡p⁢(𝒟ℱ|Φ′)ℱ𝑝conditionalsubscript𝒟ℱsuperscriptΦ′\mathcal{F}=-\log p(\mathcal{D}\_{\mathcal{F}}|\Phi^{\prime})caligraphic\_F = - roman\_log italic\_p ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT | roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ), denoted as ℒℱ⁢(Φ′)subscriptℒℱsuperscriptΦ′\mathcal{L}\_{\mathcal{F}}(\Phi^{\prime})caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ). The second term p⁢(Φ′|𝒟𝒫)𝑝conditionalsuperscriptΦ′subscript𝒟𝒫p(\Phi^{\prime}|\mathcal{D}\_{\mathcal{P}})italic\_p ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT | caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT ) is the posterior of the parameters given the pre-training dataset 𝒟𝒫subscript𝒟𝒫\mathcal{D}\_{\mathcal{P}}caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT.
Since Φ′=[Φ1′⁣⊤,Φ2′⁣⊤,…,ΦE′⁣⊤]⊤superscriptΦ′superscript

superscriptsubscriptΦ1

′topsuperscriptsubscriptΦ2

′top…superscriptsubscriptΦ𝐸

′top
top\Phi^{\prime}=[\Phi\_{1}^{\prime\top},\Phi\_{2}^{\prime\top},\ldots,\Phi\_{E}^{%
\prime\top}]^{\top}roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT = [ roman\_Φ start\_POSTSUBSCRIPT 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ ⊤ end\_POSTSUPERSCRIPT , roman\_Φ start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ ⊤ end\_POSTSUPERSCRIPT , … , roman\_Φ start\_POSTSUBSCRIPT italic\_E end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ ⊤ end\_POSTSUPERSCRIPT ] start\_POSTSUPERSCRIPT ⊤ end\_POSTSUPERSCRIPT, and Φi′superscriptsubscriptΦ𝑖′\Phi\_{i}^{\prime}roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT is conditionally independent of Φj′superscriptsubscriptΦ𝑗′\Phi\_{j}^{\prime}roman\_Φ start\_POSTSUBSCRIPT italic\_j end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT for i,j={1,…,E}

𝑖𝑗
1…𝐸i,j=\{1,\ldots,E\}italic\_i , italic\_j = { 1 , … , italic\_E } and i≠j𝑖𝑗i\neq jitalic\_i ≠ italic\_j given condition 𝒟𝒫subscript𝒟𝒫\mathcal{D}\_{\mathcal{P}}caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT, we have p⁢(Φ′|𝒟𝒫)=∏i=1Ep⁢(Φi′|𝒟𝒫)𝑝conditionalsuperscriptΦ′subscript𝒟𝒫superscriptsubscriptproduct𝑖1𝐸𝑝conditionalsuperscriptsubscriptΦ𝑖′subscript𝒟𝒫p(\Phi^{\prime}|\mathcal{D}\_{\mathcal{P}})=\prod\_{i=1}^{E}p(\Phi\_{i}^{\prime}|%
\mathcal{D}\_{\mathcal{P}})italic\_p ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT | caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT ) = ∏ start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_E end\_POSTSUPERSCRIPT italic\_p ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT | caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT ). Thus, log⁡p⁢(Φ′|𝒟𝒫)=∑i=1Elog⁡p⁢(Φi′|𝒟𝒫)𝑝conditionalsuperscriptΦ′subscript𝒟𝒫superscriptsubscript𝑖1𝐸𝑝conditionalsuperscriptsubscriptΦ𝑖′subscript𝒟𝒫\log p(\Phi^{\prime}|\mathcal{D}\_{\mathcal{P}})=\sum\_{i=1}^{E}\log p(\Phi\_{i}^%
{\prime}|\mathcal{D}\_{\mathcal{P}})roman\_log italic\_p ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT | caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT ) = ∑ start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_E end\_POSTSUPERSCRIPT roman\_log italic\_p ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT | caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT )


For adapting pre-trained molecular embedding layers to downstream FMSPP tasks, this posterior must encompass the prior knowledge of the pre-trained embedding layers to reflect which parameters are important for pre-training task 𝒫𝒫\mathcal{P}caligraphic\_P. Despite the true posterior being intractable, log⁡p⁢(Φi′|𝒟𝒫)𝑝conditionalsuperscriptsubscriptΦ𝑖′subscript𝒟𝒫\log p(\Phi\_{i}^{\prime}|\mathcal{D}\_{\mathcal{P}})roman\_log italic\_p ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT | caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT ) can be defined as a function f⁢(Φi′)𝑓superscriptsubscriptΦ𝑖′f(\Phi\_{i}^{\prime})italic\_f ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ) and approximated around the optimum point f⁢(Φi)𝑓subscriptΦ𝑖f(\Phi\_{i})italic\_f ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ), where f⁢(Φi)𝑓subscriptΦ𝑖f(\Phi\_{i})italic\_f ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) is the pre-trained values and ∇f⁢(Φi)=0∇𝑓subscriptΦ𝑖0\nabla f(\Phi\_{i})=0∇ italic\_f ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) = 0. Performing a second-order Taylor expansion on f⁢(Φi′)𝑓superscriptsubscriptΦ𝑖′f(\Phi\_{i}^{\prime})italic\_f ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ) around ΦisubscriptΦ𝑖\Phi\_{i}roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT gives:

|  | log⁡p⁢(Φi′∣𝒟𝒫)𝑝conditionalsuperscriptsubscriptΦ𝑖′subscript𝒟𝒫\displaystyle\log p\left(\Phi\_{i}^{\prime}\mid\mathcal{D}\_{\mathcal{P}}\right)roman\_log italic\_p ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ∣ caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT ) | ≈f⁢(Φi)+12⁢(Φi′−Φi)T⁢∇2f⁢(Φi)⁢(Φi′−Φi)absent𝑓subscriptΦ𝑖12superscriptsuperscriptsubscriptΦ𝑖′subscriptΦ𝑖𝑇superscript∇2𝑓subscriptΦ𝑖superscriptsubscriptΦ𝑖′subscriptΦ𝑖\displaystyle\approx f\left(\Phi\_{i}\right)+\frac{1}{2}\left(\Phi\_{i}^{\prime}% -\Phi\_{i}\right)^{T}\nabla^{2}f\left(\Phi\_{i}\right)\left(\Phi\_{i}^{\prime}-% \Phi\_{i}\right)≈ italic\_f ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) + divide start\_ARG 1 end\_ARG start\_ARG 2 end\_ARG ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT italic\_T end\_POSTSUPERSCRIPT ∇ start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT italic\_f ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) |  | (14) |
| --- | --- | --- | --- | --- |
|  |  | =f⁢(Φi)+12⁢(Φi′−Φi)T⁢𝐇⁢(𝒟𝒫,Φi)⁢(Φi′−Φi),absent𝑓subscriptΦ𝑖12superscriptsuperscriptsubscriptΦ𝑖′subscriptΦ𝑖𝑇𝐇subscript𝒟𝒫subscriptΦ𝑖superscriptsubscriptΦ𝑖′subscriptΦ𝑖\displaystyle=f\left(\Phi\_{i}\right)+\frac{1}{2}\left(\Phi\_{i}^{\prime}-\Phi\_{% i}\right)^{T}\mathbf{H}(\mathcal{D}\_{\mathcal{P}},\Phi\_{i})\left(\Phi\_{i}^{% \prime}-\Phi\_{i}\right),= italic\_f ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) + divide start\_ARG 1 end\_ARG start\_ARG 2 end\_ARG ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT italic\_T end\_POSTSUPERSCRIPT bold\_H ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) , |  |

where 𝐇⁢(𝒟𝒫,Φi)∈ℝd×d𝐇subscript𝒟𝒫subscriptΦ𝑖superscriptℝ𝑑𝑑\mathbf{H}(\mathcal{D}\_{\mathcal{P}},\Phi\_{i})\in\mathbb{R}^{d\times d}bold\_H ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d × italic\_d end\_POSTSUPERSCRIPT is the Hessian matrix of f⁢(Φi′)𝑓superscriptsubscriptΦ𝑖′f(\Phi\_{i}^{\prime})italic\_f ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ) at ΦisubscriptΦ𝑖\Phi\_{i}roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT. The second term suggests that the posterior of the parameters on the pre-training data can be approximated by a Gaussian distribution with mean Φi′superscriptsubscriptΦ𝑖′\Phi\_{i}^{\prime}roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT and covariance 𝐇⁢(𝒟𝒫,Φi)−1𝐇superscriptsubscript𝒟𝒫subscriptΦ𝑖1\mathbf{H}(\mathcal{D}\_{\mathcal{P}},\Phi\_{i})^{-1}bold\_H ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT - 1 end\_POSTSUPERSCRIPT.
Following [Eq. 13](https://arxiv.org/html/2411.01158v1#A1.E13 "In A.1 Derivation of ℒ_\"Emb-BWC\" ‣ Appendix A Derivation of Emb-BWC regularization ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction"), the training objective becomes:

|  | Φ′⁣∗superscriptΦ  ′\displaystyle\Phi^{\prime\*}roman\_Φ start\_POSTSUPERSCRIPT ′ ∗ end\_POSTSUPERSCRIPT | =arg⁢maxΦ′⁢log⁡p⁢(𝒟ℱ|Φ′)+log⁡p⁢(Φ′|𝒟𝒫)absentsuperscriptΦ′argmax𝑝conditionalsubscript𝒟ℱsuperscriptΦ′𝑝conditionalsuperscriptΦ′subscript𝒟𝒫\displaystyle=\underset{\Phi^{\prime}}{\operatorname\*{arg\,max}}\log p(% \mathcal{D}\_{\mathcal{F}}|\Phi^{\prime})+\log p(\Phi^{\prime}|\mathcal{D}\_{% \mathcal{P}})= start\_UNDERACCENT roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT end\_UNDERACCENT start\_ARG roman\_arg roman\_max end\_ARG roman\_log italic\_p ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT | roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ) + roman\_log italic\_p ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT | caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT ) |  | (15) |
| --- | --- | --- | --- | --- |
|  |  | =arg⁢maxΦ′⁢log⁡p⁢(𝒟ℱ|Φ′)+∑i=1Elog⁡p⁢(Φi′|𝒟𝒫)absentsuperscriptΦ′argmax𝑝conditionalsubscript𝒟ℱsuperscriptΦ′superscriptsubscript𝑖1𝐸𝑝conditionalsuperscriptsubscriptΦ𝑖′subscript𝒟𝒫\displaystyle=\underset{\Phi^{\prime}}{\operatorname\*{arg\,max}}\log p(% \mathcal{D}\_{\mathcal{F}}|\Phi^{\prime})+\sum\_{i=1}^{E}\log p(\Phi\_{i}^{\prime% }|\mathcal{D}\_{\mathcal{P}})= start\_UNDERACCENT roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT end\_UNDERACCENT start\_ARG roman\_arg roman\_max end\_ARG roman\_log italic\_p ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT | roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ) + ∑ start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_E end\_POSTSUPERSCRIPT roman\_log italic\_p ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT | caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT ) |  |
|  |  | =arg⁢minΦ′⁢ℒℱ⁢(Φ′)−∑i=1Ef⁢(Φi)−12⁢∑i=1E(Φi′−Φi)T⁢𝐇⁢(𝒟𝒫,Φi)⁢(Φi′−Φi)absentsuperscriptΦ′argminsubscriptℒℱsuperscriptΦ′superscriptsubscript𝑖1𝐸𝑓subscriptΦ𝑖12superscriptsubscript𝑖1𝐸superscriptsuperscriptsubscriptΦ𝑖′subscriptΦ𝑖𝑇𝐇subscript𝒟𝒫subscriptΦ𝑖superscriptsubscriptΦ𝑖′subscriptΦ𝑖\displaystyle=\underset{\Phi^{\prime}}{\operatorname\*{arg\,min}}\mathcal{L}\_{% \mathcal{F}}(\Phi^{\prime})-\sum\_{i=1}^{E}f(\Phi\_{i})-\frac{1}{2}\sum\_{i=1}^{E% }\left(\Phi\_{i}^{\prime}-\Phi\_{i}\right)^{T}\mathbf{H}(\mathcal{D}\_{\mathcal{P% }},\Phi\_{i})\left(\Phi\_{i}^{\prime}-\Phi\_{i}\right)= start\_UNDERACCENT roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT end\_UNDERACCENT start\_ARG roman\_arg roman\_min end\_ARG caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ) - ∑ start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_E end\_POSTSUPERSCRIPT italic\_f ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) - divide start\_ARG 1 end\_ARG start\_ARG 2 end\_ARG ∑ start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_E end\_POSTSUPERSCRIPT ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT italic\_T end\_POSTSUPERSCRIPT bold\_H ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) |  |
|  |  | =arg⁢minΦ′⁢ℒℱ⁢(Φ′)−12⁢∑i=1E(Φi′−Φi)T⁢𝐇⁢(𝒟𝒫,Φi)⁢(Φi′−Φi).absentsuperscriptΦ′argminsubscriptℒℱsuperscriptΦ′12superscriptsubscript𝑖1𝐸superscriptsuperscriptsubscriptΦ𝑖′subscriptΦ𝑖𝑇𝐇subscript𝒟𝒫subscriptΦ𝑖superscriptsubscriptΦ𝑖′subscriptΦ𝑖\displaystyle=\underset{\Phi^{\prime}}{\operatorname\*{arg\,min}}\mathcal{L}\_{% \mathcal{F}}(\Phi^{\prime})-\frac{1}{2}\sum\_{i=1}^{E}\left(\Phi\_{i}^{\prime}-% \Phi\_{i}\right)^{T}\mathbf{H}(\mathcal{D}\_{\mathcal{P}},\Phi\_{i})\left(\Phi\_{i% }^{\prime}-\Phi\_{i}\right).= start\_UNDERACCENT roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT end\_UNDERACCENT start\_ARG roman\_arg roman\_min end\_ARG caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_F end\_POSTSUBSCRIPT ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ) - divide start\_ARG 1 end\_ARG start\_ARG 2 end\_ARG ∑ start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_E end\_POSTSUPERSCRIPT ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT italic\_T end\_POSTSUPERSCRIPT bold\_H ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) ( roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) . |  |

We define the second term as our Emb-BWC regularization objective:

|  | ℒEmb-BWC=−12⁢∑i=1E(Φi′−Φi)⊤⁢𝐇⁢(𝒟𝒫,Φi)⁢(Φi′−Φi).subscriptℒEmb-BWC12superscriptsubscript𝑖1𝐸superscriptsubscriptsuperscriptΦ′𝑖subscriptΦ𝑖top𝐇subscript𝒟𝒫subscriptΦ𝑖subscriptsuperscriptΦ′𝑖subscriptΦ𝑖\mathcal{L}\_{\textrm{Emb-BWC}}=-\frac{1}{2}\sum\_{i=1}^{E}(\Phi^{\prime}\_{i}-% \Phi\_{i})^{\top}\mathbf{H}(\mathcal{D}\_{\mathcal{P}},\Phi\_{i})(\Phi^{\prime}\_{% i}-\Phi\_{i}).caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT = - divide start\_ARG 1 end\_ARG start\_ARG 2 end\_ARG ∑ start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_E end\_POSTSUPERSCRIPT ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT ⊤ end\_POSTSUPERSCRIPT bold\_H ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) . |  | (16) |
| --- | --- | --- | --- |


### A.2 Derivation of ℒEmb-BWCFIMsuperscriptsubscriptℒEmb-BWCFIM\mathcal{L}\_{\textrm{Emb-BWC}}^{\textrm{FIM}}caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT FIM end\_POSTSUPERSCRIPT

Since the Fisher information matrix (FIM) 𝐅𝐅\mathbf{F}bold\_F is the negation of the expectation of the Hessian over the data distribution, i.e., 𝐅=−𝔼𝒟𝒫⁢[𝐇]𝐅subscript𝔼subscript𝒟𝒫delimited-[]𝐇\mathbf{F}=-\mathbb{E}\_{\mathcal{D}\_{\mathcal{P}}}[\mathbf{H}]bold\_F = - blackboard\_E start\_POSTSUBSCRIPT caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_POSTSUBSCRIPT [ bold\_H ], the objective can be reformulated as:

|  | ℒEmb-BWCFIM=12⁢∑i=1E(Φi′−Φi)⊤⁢𝐅⁢(𝒟𝒫,Φi)⁢(Φi′−Φi),superscriptsubscriptℒEmb-BWCFIM12superscriptsubscript𝑖1𝐸superscriptsubscriptsuperscriptΦ′𝑖subscriptΦ𝑖top𝐅subscript𝒟𝒫subscriptΦ𝑖subscriptsuperscriptΦ′𝑖subscriptΦ𝑖\mathcal{L}\_{\textrm{Emb-BWC}}^{\textrm{FIM}}=\frac{1}{2}\sum\_{i=1}^{E}(\Phi^{% \prime}\_{i}-\Phi\_{i})^{\top}\mathbf{F}(\mathcal{D}\_{\mathcal{P}},\Phi\_{i})(% \Phi^{\prime}\_{i}-\Phi\_{i}),caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT FIM end\_POSTSUPERSCRIPT = divide start\_ARG 1 end\_ARG start\_ARG 2 end\_ARG ∑ start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_E end\_POSTSUPERSCRIPT ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT ⊤ end\_POSTSUPERSCRIPT bold\_F ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) , |  | (17) |
| --- | --- | --- | --- |

where 𝐅⁢(𝒟𝒫,Φi)∈ℝd×d𝐅subscript𝒟𝒫subscriptΦ𝑖superscriptℝ𝑑𝑑\mathbf{F}(\mathcal{D}\_{\mathcal{P}},\Phi\_{i})\in\mathbb{R}^{d\times d}bold\_F ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d × italic\_d end\_POSTSUPERSCRIPT is the corresponding Fisher information matrix of 𝐇⁢(𝒟𝒫,Φi)𝐇subscript𝒟𝒫subscriptΦ𝑖\mathbf{H}(\mathcal{D}\_{\mathcal{P}},\Phi\_{i})bold\_H ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ).
Further, the Fisher information matrix can be further simplified with a diagonal approximation. Then, the objective is simplified to:

|  | ℒEmb-BWCFIM≈12⁢∑i=1E𝐅^i⁢(Φi′−Φi)2,superscriptsubscriptℒEmb-BWCFIM12superscriptsubscript𝑖1𝐸subscript^𝐅𝑖superscriptsubscriptsuperscriptΦ′𝑖subscriptΦ𝑖2\mathcal{L}\_{\textrm{Emb-BWC}}^{\textrm{FIM}}\approx\frac{1}{2}\sum\_{i=1}^{E}% \hat{\mathbf{F}}\_{i}(\Phi^{\prime}\_{i}-\Phi\_{i})^{2},caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT FIM end\_POSTSUPERSCRIPT ≈ divide start\_ARG 1 end\_ARG start\_ARG 2 end\_ARG ∑ start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_E end\_POSTSUPERSCRIPT over^ start\_ARG bold\_F end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ( roman\_Φ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT , |  | (18) |
| --- | --- | --- | --- |

where 𝐅^i∈ℝdsubscript^𝐅𝑖superscriptℝ𝑑\hat{\mathbf{F}}\_{i}\in\mathbb{R}^{d}over^ start\_ARG bold\_F end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ∈ blackboard\_R start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT is the diagonal of 𝐅⁢(𝒟𝒫,Φi)𝐅subscript𝒟𝒫subscriptΦ𝑖\mathbf{F}(\mathcal{D}\_{\mathcal{P}},\Phi\_{i})bold\_F ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ). According to the definition of the Fisher information matrix, the j𝑗jitalic\_j-th value in 𝐅^isubscript^𝐅𝑖\hat{\mathbf{F}}\_{i}over^ start\_ARG bold\_F end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT is computed as 𝔼𝒟𝒫⁢(∂ℒ𝒫/∂Φi,j)2subscript𝔼subscript𝒟𝒫superscriptsubscriptℒ𝒫subscriptΦ

𝑖𝑗2\mathbb{E}\_{\mathcal{D}\_{\mathcal{P}}}(\partial\mathcal{L}\_{\mathcal{P}}/%
\partial\Phi\_{i,j})^{2}blackboard\_E start\_POSTSUBSCRIPT caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_POSTSUBSCRIPT ( ∂ caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT / ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT. In this work, this approximated form is defined as ℒEmb-BWCFIMsuperscriptsubscriptℒEmb-BWCFIM\mathcal{L}\_{\textrm{Emb-BWC}}^{\textrm{FIM}}caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT FIM end\_POSTSUPERSCRIPT.


### A.3 Derivation of ℒEmb-BWCEFIMsuperscriptsubscriptℒEmb-BWCEFIM\mathcal{L}\_{\textrm{Emb-BWC}}^{\textrm{EFIM}}caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT EFIM end\_POSTSUPERSCRIPT

We assume that the parameters within an embedding should share the same importance. To this end, we define Φ~i=∑jΦi,jsubscript~Φ𝑖subscript𝑗subscriptΦ

𝑖𝑗\tilde{\Phi}\_{i}=\sum\_{j}\Phi\_{i,j}over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT = ∑ start\_POSTSUBSCRIPT italic\_j end\_POSTSUBSCRIPT roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT, then the total update of the embedding ΦisubscriptΦ𝑖\Phi\_{i}roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT can be represented as Δ⁢Φi=Φ~i′−Φ~i=∑j(Φi,j′−Φi,j)ΔsubscriptΦ𝑖superscriptsubscript~Φ𝑖′subscript~Φ𝑖subscript𝑗superscriptsubscriptΦ

𝑖𝑗′subscriptΦ

𝑖𝑗\Delta\Phi\_{i}=\tilde{\Phi}\_{i}^{\prime}-\tilde{\Phi}\_{i}=\sum\_{j}(\Phi\_{i,j}^%
{\prime}-\Phi\_{i,j})roman\_Δ roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT = over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT - over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT = ∑ start\_POSTSUBSCRIPT italic\_j end\_POSTSUBSCRIPT ( roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT - roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT ). Then, the objective in [Eq. 16](https://arxiv.org/html/2411.01158v1#A1.E16 "In A.1 Derivation of ℒ_\"Emb-BWC\" ‣ Appendix A Derivation of Emb-BWC regularization ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") is reformulated to:

|  | ℒEmb-EWCEFIM=12⁢∑i=1E𝐇~i⁢(Φ~i′−Φ~i)2,superscriptsubscriptℒEmb-EWCEFIM12superscriptsubscript𝑖1𝐸subscript~𝐇𝑖superscriptsubscriptsuperscript~Φ′𝑖subscript~Φ𝑖2\mathcal{L}\_{\textrm{Emb-EWC}}^{\textrm{EFIM}}=\frac{1}{2}\sum\_{i=1}^{E}\tilde% {\mathbf{H}}\_{i}(\tilde{\Phi}^{\prime}\_{i}-\tilde{\Phi}\_{i})^{2},caligraphic\_L start\_POSTSUBSCRIPT Emb-EWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT EFIM end\_POSTSUPERSCRIPT = divide start\_ARG 1 end\_ARG start\_ARG 2 end\_ARG ∑ start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_E end\_POSTSUPERSCRIPT over~ start\_ARG bold\_H end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ( over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT - over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT , |  | (19) |
| --- | --- | --- | --- |

where 𝐇~i=∂2ℒ𝒫∂Φ~i2=∂∂Φ~i⁢(∂ℒ𝒫∂Φ~i)subscript~𝐇𝑖superscript2subscriptℒ𝒫superscriptsubscript~Φ𝑖2subscript~Φ𝑖subscriptℒ𝒫subscript~Φ𝑖\tilde{\mathbf{H}}\_{i}=\frac{\partial^{2}\mathcal{L}\_{\mathcal{P}}}{\partial%
\tilde{\Phi}\_{i}^{2}}=\frac{\partial}{\partial\tilde{\Phi}\_{i}}(\frac{\partial%
\mathcal{L}\_{\mathcal{P}}}{\partial\tilde{\Phi}\_{i}})over~ start\_ARG bold\_H end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT = divide start\_ARG ∂ start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT end\_ARG = divide start\_ARG ∂ end\_ARG start\_ARG ∂ over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT end\_ARG ( divide start\_ARG ∂ caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT end\_ARG ).
Next, we continue to derive 𝐇~isubscript~𝐇𝑖\tilde{\mathbf{H}}\_{i}over~ start\_ARG bold\_H end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT. Given that Φ~i=∑j=1dΦi,jsubscript~Φ𝑖superscriptsubscript𝑗1𝑑subscriptΦ

𝑖𝑗\tilde{\Phi}\_{i}=\sum\_{j=1}^{d}\Phi\_{i,j}over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT = ∑ start\_POSTSUBSCRIPT italic\_j = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT, we first use the chain rule to find ∂ℒ𝒫∂Φ~isubscriptℒ𝒫subscript~Φ𝑖\frac{\partial\mathcal{L}\_{\mathcal{P}}}{\partial\tilde{\Phi}\_{i}}divide start\_ARG ∂ caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT end\_ARG According to chain rule, the derivative of ℒ𝒫subscriptℒ𝒫\mathcal{L}\_{\mathcal{P}}caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT with respect to ∂Φ~isubscript~Φ𝑖\partial\tilde{\Phi}\_{i}∂ over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT can be computed as:

|  | ∂ℒ𝒫∂Φ~i=∑j∂ℒ𝒫∂Φi,j⁢∂Φi,j∂Φ~i.subscriptℒ𝒫subscript~Φ𝑖subscript𝑗subscriptℒ𝒫subscriptΦ  𝑖𝑗subscriptΦ  𝑖𝑗subscript~Φ𝑖\frac{\partial\mathcal{L}\_{\mathcal{P}}}{\partial\tilde{\Phi}\_{i}}=\sum\_{j}% \frac{\partial\mathcal{L}\_{\mathcal{P}}}{\partial{\Phi}\_{i,j}}\frac{\partial{% \Phi}\_{i,j}}{\partial\tilde{\Phi}\_{i}}.divide start\_ARG ∂ caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT end\_ARG = ∑ start\_POSTSUBSCRIPT italic\_j end\_POSTSUBSCRIPT divide start\_ARG ∂ caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT end\_ARG divide start\_ARG ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT end\_ARG . |  | (20) |
| --- | --- | --- | --- |

Since Φ~i=Φi,1+Φi,2+…+Φi,dsubscript~Φ𝑖subscriptΦ

𝑖1subscriptΦ

𝑖2…subscriptΦ

𝑖𝑑\tilde{\Phi}\_{i}=\Phi\_{i,1}+\Phi\_{i,2}+\ldots+\Phi\_{i,d}over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT = roman\_Φ start\_POSTSUBSCRIPT italic\_i , 1 end\_POSTSUBSCRIPT + roman\_Φ start\_POSTSUBSCRIPT italic\_i , 2 end\_POSTSUBSCRIPT + … + roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_d end\_POSTSUBSCRIPT, each of ∂Φi,j∂Φ~isubscriptΦ

𝑖𝑗subscript~Φ𝑖\frac{\partial{\Phi}\_{i,j}}{\partial\tilde{\Phi}\_{i}}divide start\_ARG ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT end\_ARG for j=1,2,…,d𝑗

12…𝑑j=1,2,\ldots,ditalic\_j = 1 , 2 , … , italic\_d equals 1111. Therefore, the equation simplifies to:

|  | ∂ℒ𝒫∂Φ~i=∑j=1d∂ℒ𝒫∂Φi,j.subscriptℒ𝒫subscript~Φ𝑖superscriptsubscript𝑗1𝑑subscriptℒ𝒫subscriptΦ  𝑖𝑗\frac{\partial\mathcal{L}\_{\mathcal{P}}}{\partial\tilde{\Phi}\_{i}}=\sum\_{j=1}^% {d}\frac{\partial\mathcal{L}\_{\mathcal{P}}}{\partial{\Phi}\_{i,j}}.divide start\_ARG ∂ caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT end\_ARG = ∑ start\_POSTSUBSCRIPT italic\_j = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT divide start\_ARG ∂ caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT end\_ARG . |  | (21) |
| --- | --- | --- | --- |

When taking the derivative of this with respect to Φi,jsubscriptΦ

𝑖𝑗{\Phi}\_{i,j}roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT again, using the chain rule, we have:

|  | ∂2ℒ𝒫∂Φ~i2=∂∂Φ~i⁢(∑j=1d∂ℒ𝒫∂Φi,j)=∑j=1d∑k=1d∂∂Φi,k⁢(∂ℒ𝒫∂Φi,j)⁢∂Φi,k∂Φ~isuperscript2subscriptℒ𝒫superscriptsubscript~Φ𝑖2subscript~Φ𝑖superscriptsubscript𝑗1𝑑subscriptℒ𝒫subscriptΦ  𝑖𝑗superscriptsubscript𝑗1𝑑superscriptsubscript𝑘1𝑑subscriptΦ  𝑖𝑘subscriptℒ𝒫subscriptΦ  𝑖𝑗subscriptΦ  𝑖𝑘subscript~Φ𝑖\displaystyle\frac{\partial^{2}\mathcal{L}\_{\mathcal{P}}}{\partial\tilde{\Phi}% \_{i}^{2}}=\frac{\partial}{\partial\tilde{\Phi}\_{i}}(\sum\_{j=1}^{d}\frac{% \partial\mathcal{L}\_{\mathcal{P}}}{\partial{\Phi}\_{i,j}})=\sum\_{j=1}^{d}\sum\_{% k=1}^{d}\frac{\partial}{\partial{\Phi}\_{i,k}}(\frac{\partial\mathcal{L}\_{% \mathcal{P}}}{\partial{\Phi}\_{i,j}})\frac{\partial{\Phi}\_{i,k}}{\partial\tilde% {\Phi}\_{i}}divide start\_ARG ∂ start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT end\_ARG = divide start\_ARG ∂ end\_ARG start\_ARG ∂ over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT end\_ARG ( ∑ start\_POSTSUBSCRIPT italic\_j = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT divide start\_ARG ∂ caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT end\_ARG ) = ∑ start\_POSTSUBSCRIPT italic\_j = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT ∑ start\_POSTSUBSCRIPT italic\_k = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_d end\_POSTSUPERSCRIPT divide start\_ARG ∂ end\_ARG start\_ARG ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_k end\_POSTSUBSCRIPT end\_ARG ( divide start\_ARG ∂ caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT end\_ARG ) divide start\_ARG ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_k end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT end\_ARG |  | (22) |
| --- | --- | --- | --- |

Given that Φi,j⁢(Φi,k)subscriptΦ

𝑖𝑗subscriptΦ

𝑖𝑘\Phi\_{i,j}(\Phi\_{i,k})roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT ( roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_k end\_POSTSUBSCRIPT ) are all parameters in embedding lookup tables, they are independent of each other. Thus, when j=k𝑗𝑘j=kitalic\_j = italic\_k, ∂∂Φi,k⁢(∂ℒ𝒫∂Φi,j)⁢∂Φi,k∂Φ~i=∂2ℒ𝒫∂Φi,j2subscriptΦ

𝑖𝑘subscriptℒ𝒫subscriptΦ

𝑖𝑗subscriptΦ

𝑖𝑘subscript~Φ𝑖superscript2subscriptℒ𝒫superscriptsubscriptΦ

𝑖𝑗2\frac{\partial}{\partial{\Phi}\_{i,k}}(\frac{\partial\mathcal{L}\_{\mathcal{P}}}%
{\partial{\Phi}\_{i,j}})\frac{\partial{\Phi}\_{i,k}}{\partial\tilde{\Phi}\_{i}}=%
\frac{\partial^{2}\mathcal{L}\_{\mathcal{P}}}{\partial{\Phi}\_{i,j}^{2}}divide start\_ARG ∂ end\_ARG start\_ARG ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_k end\_POSTSUBSCRIPT end\_ARG ( divide start\_ARG ∂ caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT end\_ARG ) divide start\_ARG ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_k end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT end\_ARG = divide start\_ARG ∂ start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT end\_ARG, otherwise it equals 0. We finally get 𝐇~i=∑j∂2ℒ𝒫∂Φi,j2=∑j𝐇⁢(𝒟𝒫,Φi)j,jsubscript~𝐇𝑖subscript𝑗superscript2subscriptℒ𝒫superscriptsubscriptΦ

𝑖𝑗2subscript𝑗𝐇subscriptsubscript𝒟𝒫subscriptΦ𝑖

𝑗𝑗\tilde{\mathbf{H}}\_{i}=\sum\_{j}\frac{\partial^{2}\mathcal{L}\_{\mathcal{P}}}{%
\partial{\Phi}\_{i,j}^{2}}=\sum\_{j}\mathbf{H}(\mathcal{D}\_{\mathcal{P}},\Phi\_{i%
})\_{j,j}over~ start\_ARG bold\_H end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT = ∑ start\_POSTSUBSCRIPT italic\_j end\_POSTSUBSCRIPT divide start\_ARG ∂ start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_ARG start\_ARG ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT end\_ARG = ∑ start\_POSTSUBSCRIPT italic\_j end\_POSTSUBSCRIPT bold\_H ( caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT , roman\_Φ start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) start\_POSTSUBSCRIPT italic\_j , italic\_j end\_POSTSUBSCRIPT.


We still approximate the Hessian with the FIM as in Section A.2, and combining this with the definition of FIM, we arrive at the final objective:

|  | ℒEmb-EWCEFIM≈12⁢∑i=1E𝐅~i⁢(Φ~i′−Φ~i)2,superscriptsubscriptℒEmb-EWCEFIM12superscriptsubscript𝑖1𝐸subscript~𝐅𝑖superscriptsubscriptsuperscript~Φ′𝑖subscript~Φ𝑖2\mathcal{L}\_{\textrm{Emb-EWC}}^{\textrm{EFIM}}\approx\frac{1}{2}\sum\_{i=1}^{E}% \tilde{\mathbf{F}}\_{i}(\tilde{\Phi}^{\prime}\_{i}-\tilde{\Phi}\_{i})^{2},caligraphic\_L start\_POSTSUBSCRIPT Emb-EWC end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT EFIM end\_POSTSUPERSCRIPT ≈ divide start\_ARG 1 end\_ARG start\_ARG 2 end\_ARG ∑ start\_POSTSUBSCRIPT italic\_i = 1 end\_POSTSUBSCRIPT start\_POSTSUPERSCRIPT italic\_E end\_POSTSUPERSCRIPT over~ start\_ARG bold\_F end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ( over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT - over~ start\_ARG roman\_Φ end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT , |  | (23) |
| --- | --- | --- | --- |

where 𝐅~i=∑j𝔼𝒟𝒫⁢(∂ℒ𝒫/∂Φi,j)2subscript~𝐅𝑖subscript𝑗subscript𝔼subscript𝒟𝒫superscriptsubscriptℒ𝒫subscriptΦ

𝑖𝑗2\tilde{\mathbf{F}}\_{i}=\sum\_{j}\mathbb{E}\_{\mathcal{D}\_{\mathcal{P}}}(\partial%
\mathcal{L}\_{\mathcal{P}}/{\partial\Phi\_{i,j}})^{2}over~ start\_ARG bold\_F end\_ARG start\_POSTSUBSCRIPT italic\_i end\_POSTSUBSCRIPT = ∑ start\_POSTSUBSCRIPT italic\_j end\_POSTSUBSCRIPT blackboard\_E start\_POSTSUBSCRIPT caligraphic\_D start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT end\_POSTSUBSCRIPT ( ∂ caligraphic\_L start\_POSTSUBSCRIPT caligraphic\_P end\_POSTSUBSCRIPT / ∂ roman\_Φ start\_POSTSUBSCRIPT italic\_i , italic\_j end\_POSTSUBSCRIPT ) start\_POSTSUPERSCRIPT 2 end\_POSTSUPERSCRIPT.


## Appendix B Pseudo-code of training process

To help better understand the training process, we provide the brief pseudo-code of it in Algorithm [1](https://arxiv.org/html/2411.01158v1#alg1 "Algorithm 1 ‣ Appendix B Pseudo-code of training process ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").


Input : Training set 𝒟trainsubscript𝒟train\mathcal{D}\_{\text{train}}caligraphic\_D start\_POSTSUBSCRIPT train end\_POSTSUBSCRIPT

Output : Tuned few-shot molecular property prediction model with parameter θ𝜃\thetaitalic\_θ

1
while *not converge* do

2      
Sample B𝐵Bitalic\_B episode from training set 𝒟trainsubscript𝒟train\mathcal{D}\_{\text{train}}caligraphic\_D start\_POSTSUBSCRIPT train end\_POSTSUBSCRIPT to form a mini-batch {Et}t=1Bsubscriptsuperscriptsubscript𝐸𝑡𝐵𝑡1\{E\_{t}\}^{B}\_{t=1}{ italic\_E start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT } start\_POSTSUPERSCRIPT italic\_B end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_t = 1 end\_POSTSUBSCRIPT;

3      
for *t=1𝑡1t=1italic\_t = 1 to B𝐵Bitalic\_B* do

4            
Calculate classification loss on support set ℒt,𝒮c⁢l⁢s⁢(fθ)subscriptsuperscriptℒ𝑐𝑙𝑠

𝑡𝒮subscript𝑓𝜃\mathcal{L}^{cls}\_{t,\mathcal{S}}(f\_{\theta})caligraphic\_L start\_POSTSUPERSCRIPT italic\_c italic\_l italic\_s end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_t , caligraphic\_S end\_POSTSUBSCRIPT ( italic\_f start\_POSTSUBSCRIPT italic\_θ end\_POSTSUBSCRIPT ) by [Eq. 8](https://arxiv.org/html/2411.01158v1#S4.E8 "In 4.3 Optimization ‣ 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") on Etsubscript𝐸𝑡E\_{t}italic\_E start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT: θ′←θ−αi⁢n⁢n⁢e⁢r⁢∇θℒt,𝒮c⁢l⁢s⁢(fθ)←superscript𝜃′𝜃subscript𝛼𝑖𝑛𝑛𝑒𝑟subscript∇𝜃subscriptsuperscriptℒ𝑐𝑙𝑠

𝑡𝒮subscript𝑓𝜃\theta^{\prime}\leftarrow\theta-\alpha\_{inner}\nabla\_{\theta}\mathcal{L}^{cls}%
\_{t,\mathcal{S}}(f\_{\theta})italic\_θ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT ← italic\_θ - italic\_α start\_POSTSUBSCRIPT italic\_i italic\_n italic\_n italic\_e italic\_r end\_POSTSUBSCRIPT ∇ start\_POSTSUBSCRIPT italic\_θ end\_POSTSUBSCRIPT caligraphic\_L start\_POSTSUPERSCRIPT italic\_c italic\_l italic\_s end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_t , caligraphic\_S end\_POSTSUBSCRIPT ( italic\_f start\_POSTSUBSCRIPT italic\_θ end\_POSTSUBSCRIPT );

5            
Do inner-loop update by [Eq. 9](https://arxiv.org/html/2411.01158v1#S4.E9 "In 4.3 Optimization ‣ 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") on Etsubscript𝐸𝑡E\_{t}italic\_E start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT;

6            
Calculate classification loss on query set ℒt,𝒬c⁢l⁢s⁢(fθ′)subscriptsuperscriptℒ𝑐𝑙𝑠

𝑡𝒬subscript𝑓superscript𝜃′\mathcal{L}^{cls}\_{t,\mathcal{Q}}(f\_{\theta^{\prime}})caligraphic\_L start\_POSTSUPERSCRIPT italic\_c italic\_l italic\_s end\_POSTSUPERSCRIPT start\_POSTSUBSCRIPT italic\_t , caligraphic\_Q end\_POSTSUBSCRIPT ( italic\_f start\_POSTSUBSCRIPT italic\_θ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT end\_POSTSUBSCRIPT ) by [Eq. 8](https://arxiv.org/html/2411.01158v1#S4.E8 "In 4.3 Optimization ‣ 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") on Etsubscript𝐸𝑡E\_{t}italic\_E start\_POSTSUBSCRIPT italic\_t end\_POSTSUBSCRIPT;

7            

8      Calculate update constraint ℒEmb-BWCsubscriptℒEmb-BWC\mathcal{L}\_{\textrm{Emb-BWC}}caligraphic\_L start\_POSTSUBSCRIPT Emb-BWC end\_POSTSUBSCRIPT by [Eq. 6](https://arxiv.org/html/2411.01158v1#S4.E6 "In 4.1.2 Emb-BWC: embedding layer-oriented Bayesian weight consolidation ‣ 4.1 Parameter-efficient tuning for PMEs ‣ 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction");

9      
Do outer-loop optimization by [Eq. 10](https://arxiv.org/html/2411.01158v1#S4.E10 "In 4.3 Optimization ‣ 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") and [Eq. 11](https://arxiv.org/html/2411.01158v1#S4.E11 "In 4.3 Optimization ‣ 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction"): θ←θ−αo⁢u⁢t⁢e⁢r⁢∇θℒ⁢(fθ′)←𝜃𝜃subscript𝛼𝑜𝑢𝑡𝑒𝑟subscript∇𝜃ℒsubscript𝑓superscript𝜃′\theta\leftarrow\theta-\alpha\_{outer}\nabla\_{\theta}\mathcal{L}(f\_{\theta^{%
\prime}})italic\_θ ← italic\_θ - italic\_α start\_POSTSUBSCRIPT italic\_o italic\_u italic\_t italic\_e italic\_r end\_POSTSUBSCRIPT ∇ start\_POSTSUBSCRIPT italic\_θ end\_POSTSUBSCRIPT caligraphic\_L ( italic\_f start\_POSTSUBSCRIPT italic\_θ start\_POSTSUPERSCRIPT ′ end\_POSTSUPERSCRIPT end\_POSTSUBSCRIPT );

10      

Return optimized model parameter θ𝜃\thetaitalic\_θ.


Algorithm 1 Training process of Pin-Tuning.


## Appendix C Discussion of tunable parameter size and total model size

### C.1 Tunable parameter size of molecular encoder

We compare the tunable parameter size of full fine-tuning and our Pin-Tuning.
[Section 3.3](https://arxiv.org/html/2411.01158v1#S3.SS3 "3.3 Pre-trained molecular encoders (PMEs) ‣ 3 Preliminaries ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") describes the parameters of the PME, which include those for the embedding layers and the message passing layers. We assume there are |En|subscript𝐸𝑛|E\_{n}|| italic\_E start\_POSTSUBSCRIPT italic\_n end\_POSTSUBSCRIPT | original node features and |Ee|subscript𝐸𝑒|E\_{e}|| italic\_E start\_POSTSUBSCRIPT italic\_e end\_POSTSUBSCRIPT | edge features. Considering there is one node embedding layer and L𝐿Litalic\_L edge embedding layers, the total number of parameters for the embedding part is |En|⁢d+L⁢|Ee|⁢dsubscript𝐸𝑛𝑑𝐿subscript𝐸𝑒𝑑|E\_{n}|d+L|E\_{e}|d| italic\_E start\_POSTSUBSCRIPT italic\_n end\_POSTSUBSCRIPT | italic\_d + italic\_L | italic\_E start\_POSTSUBSCRIPT italic\_e end\_POSTSUBSCRIPT | italic\_d. The parameters in the message passing layer consist of the 2-layer MLP including biases shown in [Eq. 2](https://arxiv.org/html/2411.01158v1#S3.E2 "In 3.3 Pre-trained molecular encoders (PMEs) ‣ 3 Preliminaries ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") and its subsequent batch normalization, with each layer having L⁢(2⁢d⁢d1+d+d1+2⁢d)𝐿2𝑑subscript𝑑1𝑑subscript𝑑12𝑑L(2dd\_{1}+d+d\_{1}+2d)italic\_L ( 2 italic\_d italic\_d start\_POSTSUBSCRIPT 1 end\_POSTSUBSCRIPT + italic\_d + italic\_d start\_POSTSUBSCRIPT 1 end\_POSTSUBSCRIPT + 2 italic\_d ) parameters. In summary, the total number of parameters to update in full fine-tuning is

|  | NF⁢i⁢n⁢e−T⁢u⁢n⁢i⁢n⁢g=|En|⁢d+L⁢(|Ee|⁢d+2⁢d⁢d1+3⁢d+d1).subscript𝑁𝐹𝑖𝑛𝑒𝑇𝑢𝑛𝑖𝑛𝑔subscript𝐸𝑛𝑑𝐿subscript𝐸𝑒𝑑2𝑑subscript𝑑13𝑑subscript𝑑1N\_{Fine-Tuning}=|E\_{n}|d+L(|E\_{e}|d+2dd\_{1}+3d+d\_{1}).italic\_N start\_POSTSUBSCRIPT italic\_F italic\_i italic\_n italic\_e - italic\_T italic\_u italic\_n italic\_i italic\_n italic\_g end\_POSTSUBSCRIPT = | italic\_E start\_POSTSUBSCRIPT italic\_n end\_POSTSUBSCRIPT | italic\_d + italic\_L ( | italic\_E start\_POSTSUBSCRIPT italic\_e end\_POSTSUBSCRIPT | italic\_d + 2 italic\_d italic\_d start\_POSTSUBSCRIPT 1 end\_POSTSUBSCRIPT + 3 italic\_d + italic\_d start\_POSTSUBSCRIPT 1 end\_POSTSUBSCRIPT ) . |  | (24) |
| --- | --- | --- | --- |


In our Pin-Tuning method, the parameters of the embedding layers are still updated. However, in each message passing layer, the original parameters are completely frozen, and the parts that require updating are the two feed-forward layers and the layer normalization in the bottleneck adapter module, amounting to L⁢(2⁢d⁢d2+d+d2+2⁢d)𝐿2𝑑subscript𝑑2𝑑subscript𝑑22𝑑L(2dd\_{2}+d+d\_{2}+2d)italic\_L ( 2 italic\_d italic\_d start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT + italic\_d + italic\_d start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT + 2 italic\_d ) parameters for this part. Therefore, the total number of parameters that need to be updated in our Pin-Tuning is

|  | NP⁢i⁢n−T⁢u⁢n⁢i⁢n⁢g=|En|⁢d+L⁢(|Ee|⁢d+2⁢d⁢d2+3⁢d+d2).subscript𝑁𝑃𝑖𝑛𝑇𝑢𝑛𝑖𝑛𝑔subscript𝐸𝑛𝑑𝐿subscript𝐸𝑒𝑑2𝑑subscript𝑑23𝑑subscript𝑑2N\_{Pin-Tuning}=|E\_{n}|d+L(|E\_{e}|d+2dd\_{2}+3d+d\_{2}).italic\_N start\_POSTSUBSCRIPT italic\_P italic\_i italic\_n - italic\_T italic\_u italic\_n italic\_i italic\_n italic\_g end\_POSTSUBSCRIPT = | italic\_E start\_POSTSUBSCRIPT italic\_n end\_POSTSUBSCRIPT | italic\_d + italic\_L ( | italic\_E start\_POSTSUBSCRIPT italic\_e end\_POSTSUBSCRIPT | italic\_d + 2 italic\_d italic\_d start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT + 3 italic\_d + italic\_d start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT ) . |  | (25) |
| --- | --- | --- | --- |

The difference in the number of parameters updated between the two tuning methods is Δ⁢N=(d1−d2)⁢L⁢(2⁢d+1)Δ𝑁subscript𝑑1subscript𝑑2𝐿2𝑑1\Delta N=(d\_{1}-d\_{2})L(2d+1)roman\_Δ italic\_N = ( italic\_d start\_POSTSUBSCRIPT 1 end\_POSTSUBSCRIPT - italic\_d start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT ) italic\_L ( 2 italic\_d + 1 ).


### C.2 Total model size

We provide a comparison of total model size between our Pin-Tuning and the state-of-the-art baseline method, GS-Meta. The total model size consists of both frozen parameters and trainable parameters. The results are presented in [Table 4](https://arxiv.org/html/2411.01158v1#A3.T4 "In C.2 Total model size ‣ Appendix C Discussion of tunable parameter size and total model size ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").
The total size of our model is comparable to GS-Meta, but the number of parameters that need to be trained is far less than GS-Meta.


Table 4: Comparison of total model size. ∗ indicates that the parameters are frozen.


|  | GS-Meta | Ours |
| --- | --- | --- |
| Size of Molecular Encoder | 1.86M | 1.86M∗ |
| Size of Adapter | - | 0.21M |
| Size of Context Encoder | 0.62M | 0.62M |
| Size of Classifier | 0.18M | 0.27M |
| Size of Total Model | 2.66M | 2.96M |
| Size of Tunable Part of the Model | 2.66M | 1.10M |


## Appendix D Details of datasets

We carry out experiments in MoleculeNet benchmark [[61](https://arxiv.org/html/2411.01158v1#bib.bib61)] on five widely used few-shot molecular property prediction datasets:


- •
  
  Tox21: This dataset covers qualitative toxicity measurements and was utilized in the 2014 Tox21 Data Challenge.
- •
  
  SIDER: The Side Effect Resource (SIDER) functions as a repository for marketed drugs and adverse drug reactions (ADR), categorized into 27 system organ classes.
- •
  
  MUV: The Maximum Unbiased Validation (MUV) is determined through the application of a refined nearest neighbor analysis, specifically designed for validating virtual screening techniques.
- •
  
  ToxCast: This dataset comprises a compilation of compounds with associated toxicity labels derived from high-throughput screening.
- •
  
  PCBA: PubChem BioAssay (PCBA) represents a database containing the biological activities of small molecules generated through high-throughput screening.

Dataset statistics are summarized in [Table 5](https://arxiv.org/html/2411.01158v1#A4.T5 "In Appendix D Details of datasets ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") and [Table 6](https://arxiv.org/html/2411.01158v1#A4.T6 "In Appendix D Details of datasets ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").


Table 5: Dataset statistics.


| Dataset | Tox21 | SIDER | MUV | ToxCast | PCBA |
| --- | --- | --- | --- | --- | --- |
| #Compound | 7831 | 1427 | 93127 | 8575 | 437929 |
| #Property | 12 | 27 | 17 | 617 | 128 |
| #Train Property | 9 | 21 | 12 | 451 | 118 |
| #Test Property | 3 | 6 | 5 | 158 | 10 |
| %Positive Label | 6.24 | 56.76 | 0.31 | 12.60 | 0.84 |
| %Negative Label | 76.71 | 43.24 | 15.76 | 72.43 | 59.84 |
| %Unknown Label | 17.05 | 0 | 84.21 | 14.97 | 39.32 |


Table 6: Statistics of sub-datasets of ToxCast.


| Assay Provider | #Compound | #Property | #Train Property | #Test Property | %Label active | %Label inactive | %Missing Label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| APR | 1039103910391039 | 43434343 | 33333333 | 10101010 | 10.3010.3010.3010.30 | 61.6161.6161.6161.61 | 28.0928.0928.0928.09 |
| ATG | 3423342334233423 | 146146146146 | 106106106106 | 40404040 | 5.925.925.925.92 | 93.9293.9293.9293.92 | 0.160.160.160.16 |
| BSK | 1445144514451445 | 115115115115 | 84848484 | 31313131 | 17.7117.7117.7117.71 | 82.2982.2982.2982.29 | 0.000.000.000.00 |
| CEETOX | 508508508508 | 14141414 | 10101010 | 4444 | 22.2622.2622.2622.26 | 76.3876.3876.3876.38 | 1.361.361.361.36 |
| CLD | 305305305305 | 19191919 | 14141414 | 5555 | 30.7230.7230.7230.72 | 68.3068.3068.3068.30 | 0.980.980.980.98 |
| NVS | 2130213021302130 | 139139139139 | 100100100100 | 39393939 | 3.213.213.213.21 | 4.524.524.524.52 | 92.2792.2792.2792.27 |
| OT | 1782178217821782 | 15151515 | 11111111 | 4444 | 9.789.789.789.78 | 87.7887.7887.7887.78 | 2.442.442.442.44 |
| TOX21 | 8241824182418241 | 100100100100 | 80808080 | 20202020 | 5.395.395.395.39 | 86.2686.2686.2686.26 | 8.358.358.358.35 |
| Tanguay | 1039103910391039 | 18181818 | 13131313 | 5555 | 8.058.058.058.05 | 90.8490.8490.8490.84 | 1.111.111.111.11 |


## Appendix E Details of baselines

We compare our Pin-Tuning with two types of baseline models for few-shot molecular property prediction tasks, categorized according to the training strategy of molecular encoders: trained-from-scratch methods and pre-trained methods.


Trained-from-scratch methods:


- •
  
  Siamese [[25](https://arxiv.org/html/2411.01158v1#bib.bib25)]: Siamese is used to rank similarity between input molecule pairs with a dual network.
- •
  
  ProtoNet [[46](https://arxiv.org/html/2411.01158v1#bib.bib46)]: ProtoNet learns a metric space for few-shot classification, enabling classification by calculating the distances between each query molecule and the prototype representations of each class.
- •
  
  MAML [[10](https://arxiv.org/html/2411.01158v1#bib.bib10)]: MAML adapts the meta-learned parameters to achieve good generalization performance on new tasks with a small amount of training data and gradient steps.
- •
  
  TPN [[35](https://arxiv.org/html/2411.01158v1#bib.bib35)]: TPN classifies the entire test set at once by learning to propagate labels from labeled instances to unlabeled test instances using a graph construction module that exploits the manifold structure in the data.
- •
  
  EGNN [[22](https://arxiv.org/html/2411.01158v1#bib.bib22)]: EGNN predicts edge labels on a graph constructed from input samples to explicitly capture intra-cluster similarity and inter-cluster dissimilarity.
- •
  
  IterRefLSTM [[1](https://arxiv.org/html/2411.01158v1#bib.bib1)]: IterRefLSTM adapts Matching Networks [[53](https://arxiv.org/html/2411.01158v1#bib.bib53)] to handle molecular property prediction tasks.

Pre-trained methods:


- •
  
  Pre-GNN [[20](https://arxiv.org/html/2411.01158v1#bib.bib20)]: Pre-GNN is a classic pre-trained molecular model, taking the GIN as backbone and pre-training it with different self-supervised tasks.
- •
  
  Meta-MGNN [[14](https://arxiv.org/html/2411.01158v1#bib.bib14)]: Meta-MGNN leverages Pre-GNN for learning molecular representations and incorporates meta-learning and self-supervised learning techniques.
- •
  
  PAR [[58](https://arxiv.org/html/2411.01158v1#bib.bib58)]: PAR uses class prototypes to update input representations and designs label propagation for similar inputs in the relational graph, thus enabling the transformation of generic molecular embeddings into property-aware spaces.
- •
  
  GS-Meta [[73](https://arxiv.org/html/2411.01158v1#bib.bib73)]: GS-Meta constructs a Molecule-Property relation graph (MPG) and redefines episodes in meta-learning as subgraphs of the MPG.

Following prior work [[58](https://arxiv.org/html/2411.01158v1#bib.bib58)], for the methods we reproduced, we use GIN as the graph-based molecular encoder to extract molecular representations. Specifically, we use the GIN provided by Pre-GNN [[20](https://arxiv.org/html/2411.01158v1#bib.bib20)] which consists of 5 GIN layers with 300-dimensional hidden units. Pre-GNN, Meta-MGNN, PAR, and GS-Meta further use the pre-trained GIN which is also provided by Pre-GNN.


## Appendix F Implementation details

### F.1 Hardware and software

Our experiments are conducted on Linux servers equipped with an AMD CPU EPYC 7742 (256) @ 2.250GHz, 256GB RAM and NVIDIA 3090 GPUs. Our model is implemented in PyTorch version 1.12.1, PyTorch Geometric version 2.3.1 (https://pyg.org/) with CUDA version 11.3, RDKit version 2023.3.3 and Python 3.9.18. Our code is available at: <https://github.com/CRIPAC-DIG/Pin-Tuning>.


### F.2 Model configuration

For featurization of molecules, we use atomic number and chirality tag as original atom features, as well as bond type and bond direction as bond features, which is in line with most molecular pre-training methods.
Following previous works, we set d=300𝑑300d=300italic\_d = 300. For MLPs in [Eq. 2](https://arxiv.org/html/2411.01158v1#S3.E2 "In 3.3 Pre-trained molecular encoders (PMEs) ‣ 3 Preliminaries ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction"), we use the ReLU activation with d1=600subscript𝑑1600d\_{1}=600italic\_d start\_POSTSUBSCRIPT 1 end\_POSTSUBSCRIPT = 600.
Pre-trained GIN model provided by Pre-GNN [[20](https://arxiv.org/html/2411.01158v1#bib.bib20)] is adopted as the PTME in our framework.
We tune the weight of update constraint (i.e., λ𝜆\lambdaitalic\_λ) in {0.01, 0.1, 1, 10},
tune the learning rate of inner loop (i.e., αinnersubscript𝛼inner\alpha\_{\text{inner}}italic\_α start\_POSTSUBSCRIPT inner end\_POSTSUBSCRIPT) in {1e-3, 5e-3, 1e-2,5e-2,1e-1, 5e-1, 1, 5}, and tune the learning rate of outer loop (i.e., αoutersubscript𝛼outer\alpha\_{\text{outer}}italic\_α start\_POSTSUBSCRIPT outer end\_POSTSUBSCRIPT) in {1e-5, 1e-4, 1e-3,1e-2,1e-1}.
Based on the results of hyper-parameter tuning, we adopt αinner=0.5,αouter=1⁢e−3formulae-sequencesubscript𝛼inner0.5subscript𝛼outer1𝑒3\alpha\_{\text{inner}}=0.5,\alpha\_{\text{outer}}=1e-3italic\_α start\_POSTSUBSCRIPT inner end\_POSTSUBSCRIPT = 0.5 , italic\_α start\_POSTSUBSCRIPT outer end\_POSTSUBSCRIPT = 1 italic\_e - 3 and d2=50subscript𝑑250d\_{2}=50italic\_d start\_POSTSUBSCRIPT 2 end\_POSTSUBSCRIPT = 50.
The ContextEncoder⁢(⋅)ContextEncoder⋅\texttt{ContextEncoder}(\cdot)ContextEncoder ( ⋅ ) described in [Section 4.2](https://arxiv.org/html/2411.01158v1#S4.SS2 "4.2 Enabling contextual perceptiveness in MP-Adapter ‣ 4 The proposed Pin-Tuning method ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") is implemented using a 2-layer message passing neural network [[11](https://arxiv.org/html/2411.01158v1#bib.bib11)]. In each MPNN layer, we employ a linear layer to aggregate messages from the neighborhoods of nodes and utilize distinct edge features to differentiate between various edge types in the context graphs.
For baselines, we follow their recommended settings.


Table 7: 10-shot performance on each sub-dataset of ToxCast.


| Model | APR | ATG | BSK | CEETOX | CLD | NVS | OT | TOX21 | Tanguay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ProtoNet | 73.58 | 59.26 | 70.15 | 66.12 | 78.12 | 65.85 | 64.90 | 68.26 | 73.61 |
| MAML | 72.66 | 62.09 | 66.42 | 64.08 | 74.57 | 66.56 | 64.07 | 68.04 | 77.12 |
| TPN | 74.53 | 60.74 | 65.19 | 66.63 | 75.22 | 63.20 | 64.63 | 73.30 | 81.75 |
| EGNN | 80.33 | 66.17 | 73.43 | 66.51 | 78.85 | 71.05 | 68.21 | 76.40 | 85.23 |
| Pre-GNN | 80.61 | 67.59 | 76.65 | 66.52 | 78.88 | 75.09 | 70.52 | 77.92 | 83.05 |
| Meta-MGNN | 81.47 | 69.20 | 78.97 | 66.57 | 78.30 | 79.60 | 69.55 | 78.77 | 83.98 |
| PAR | 86.09 | 72.72 | 82.45 | 72.12 | 83.43 | 74.94 | 71.96 | 82.81 | 88.20 |
| GS-Meta | 90.15 | 82.54 | 88.21 | 74.19 | 86.34 | 76.29 | 74.47 | 90.63 | 91.47 |
| Pin-Tuning | 92.78 | 83.58 | 89.49 | 75.96 | 87.70 | 76.33 | 75.56 | 90.80 | 92.25 |
| ΔΔ\Deltaroman\_Δ*Improve.* | 2.92% | 1.26% | 1.45% | 2.39% | 1.58% | 0.05% | 1.46% | 0.19% | 0.85% |


Table 8: 5-shot performance on each sub-dataset of ToxCast.


| Model | APR | ATG | BSK | CEETOX | CLD | NVS | OT | TOX21 | Tanguay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ProtoNet | 70.38 | 58.11 | 63.96 | 63.41 | 76.70 | 62.27 | 64.52 | 65.99 | 70.98 |
| MAML | 68.88 | 60.01 | 67.05 | 62.42 | 73.32 | 69.18 | 64.56 | 66.73 | 75.88 |
| TPN | 70.76 | 57.92 | 63.41 | 64.73 | 70.44 | 61.36 | 61.99 | 66.49 | 77.27 |
| EGNN | 74.06 | 60.56 | 64.60 | 63.20 | 71.44 | 62.62 | 66.70 | 65.33 | 75.69 |
| Pre-GNN | 80.38 | 66.96 | 75.64 | 64.88 | 78.03 | 74.08 | 70.42 | 75.74 | 82.73 |
| Meta-MGNN | 81.22 | 69.90 | 79.67 | 65.78 | 77.53 | 73.99 | 69.20 | 76.25 | 83.76 |
| PAR | 83.76 | 70.24 | 80.82 | 69.51 | 81.32 | 70.60 | 71.31 | 79.71 | 84.71 |
| GS-Meta | 89.36 | 81.92 | 86.12 | 74.48 | 83.10 | 74.72 | 73.26 | 89.71 | 91.15 |
| Pin-Tuning | 89.94 | 82.37 | 87.61 | 75.20 | 85.07 | 75.49 | 74.70 | 90.89 | 92.14 |
| ΔΔ\Deltaroman\_Δ*Improve.* | 0.65% | 0.55% | 1.73% | 0.97% | 2.37% | 1.03% | 1.97% | 1.32% | 1.09% |


## Appendix G More experimental results and discussions

More discussion of [Figure 1](https://arxiv.org/html/2411.01158v1#S1.F1 "In 1 Introduction ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").
The results show that molecular encoders with more molecule-specific inductive biases, such as CMPNN [[47](https://arxiv.org/html/2411.01158v1#bib.bib47)] and Graphormer [[66](https://arxiv.org/html/2411.01158v1#bib.bib66)], performed slightly worse than GIN-Mol [[20](https://arxiv.org/html/2411.01158v1#bib.bib20)] on this few-shot task. This is because more complex encoders require more parameters to provide inductive biases, which are difficult to train effectively under a few-shot setting.


More main results. The detailed comparison between Pin-Tuning and baseline models on sub-datasets of ToxCast are summarized in [Table 7](https://arxiv.org/html/2411.01158v1#A6.T7 "In F.2 Model configuration ‣ Appendix F Implementation details ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") and [Table 8](https://arxiv.org/html/2411.01158v1#A6.T8 "In F.2 Model configuration ‣ Appendix F Implementation details ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").
Our method outperforms all baseline models under both the 10-shot and 5-shot settings, demonstrating the superiority of our method compared to existing methods.


More discussion of ablation study.
Different datasets show varying sensitivity to the removal of components. On small-scale datasets like Tox21 and SIDER, removing components leads to a significant performance drop. On large-scale datasets like ToxCast and PCBA, the impact of removing components is less pronounced. This is because more episodes can be constructed on large-scale datasets, which aids in adaptation. This observation indicates that Pin-Tuning can bring considerable benefits in situations where data is extremely scarce.


More case studies. We provide more case studies in [Figure 8](https://arxiv.org/html/2411.01158v1#A7.F8 "In Appendix G More experimental results and discussions ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") and [9](https://arxiv.org/html/2411.01158v1#A7.F9 "Figure 9 ‣ Appendix G More experimental results and discussions ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction") as a supplement to [Section 5.5](https://arxiv.org/html/2411.01158v1#S5.SS5 "5.5 Case study ‣ 5 Experiments ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction").


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_GS-Meta-SR-HSE_markersize24.png)


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_GS-Meta-sider-CD_markersize24.png)


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_GS-Meta-sider-ELD_markersize24.png)


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_GS-Meta-sider-IPPC_markersize24.png)


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_GS-Meta-sider-NSD_markersize24.png)


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_GS-Meta-sider-PPPC_markersize24.png)


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_GS-Meta-sider-RUD_markersize24.png)


Figure 8: More molecular representations encoded by GS-Meta [[73](https://arxiv.org/html/2411.01158v1#bib.bib73)].


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_Pin-Tuning-SR-HSE_markersize24.png)


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_Pin-Tuning-sider-CD_markersize24.png)


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_Pin-Tuning-sider-ELD_markersize24.png)


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_Pin-Tuning-sider-IPPC_markersize24.png)


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_Pin-Tuning-sider-NSD_markersize24.png)


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_Pin-Tuning-sider-PPPC_markersize24.png)


![Refer to caption](09_Pin-Tuning_images/extracted_5972891_figures_case_study_png_Pin-Tuning-sider-RUD_markersize24.png)


Figure 9: More molecular representations encoded by Pin-Tuning.


## Appendix H Limitations and future directions

As we discusses in [Section 5.2](https://arxiv.org/html/2411.01158v1#S5.SS2 "5.2 Performance comparison ‣ 5 Experiments ‣ Pin-Tuning: Parameter-Efficient In-Context Tuning for Few-Shot Molecular Property Prediction"), although our method significantly outperforms the state-of-the-art baseline method, our method exhibits higher standard deviations in the experimental results under multiple runs with different seeds.


We further speculate that these high standard deviations might be due to the uncertainty in the context information within episodes. The explicitly introduced molecular context, on one hand, provides effective guidance for tuning pre-trained molecular encoders, but on the other hand, this information also carries a high degree of uncertainty. We aim to model the target property through the molecule-property relationships within episodes, but each episode is obtained by sampling very few samples from the large space corresponding to the target property. The uncertainty between different episodes is relatively high. How to quantify and calibrate this uncertainty is another question worth exploring, which we will investigate in our future work.


