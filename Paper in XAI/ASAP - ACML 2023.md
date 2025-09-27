
---

# Summary and Review of "ASAP: Attention-Based State Space Abstraction for Policy Summarization"

### General Information

- **Title**: ASAP: Attention-Based State Space Abstraction for Policy Summarization 

- **Authors**: Yanzhe Bekkemoen, Helge Langseth

- **Publication**: Proceedings of Machine Learning Research, ACML 2023

---

### Motivation

The primary motivation for this research is the "black-box" nature of Deep Reinforcement Learning (RL) models, which makes them difficult for end-users to understand, trust, and deploy in high-stakes domains. While many methods from Explainable AI (XAI) have been adapted for RL, they often fall short in addressing the unique challenges of sequential decision-making.

The authors identify two key weaknesses in existing approaches:

1. **Local vs. Global Explanations**: Most popular methods, like saliency maps, provide _local_ explanations, focusing on a single decision at a single timestep. This is insufficient for RL, where actions have both short-term and long-term consequences, necessitating a
    
    _global_, "bird's-eye view" of the policy's overall strategy. 
    
2. **Uninterpretable Abstractions**: Previous methods that create global summaries by clustering states into "hyperstates" often produce abstractions that are themselves difficult to interpret. 8888These methods may require significant manual effort to make sense of (e.g., defining natural language predicates) or rely on clustering objectives that are not intuitive to humans, such as state values. 9999
    

ASAP aims to solve these issues by proposing a novel method to create global policy summaries where the hyperstates are grounded in

**feature attention**, a quantity that is more intuitive for human users. 10

---

### Methodology

ASAP learns to summarize a policy by simultaneously clustering states into hyperstates and generating attention maps that explain them. The key innovation is that the clustering process is based

**only on the attention maps**, while the policy indirectly influences the clustering through these maps. 11

The architecture consists of two main learnable networks trained end-to-end:

- An
    
    **Encoder (fθ​)**: This network takes a state si​ and outputs a categorical probability distribution, ui​, which represents the likelihood of si​ belonging to each of the _k_ hyperstates. 12
    
- An
    
    **Importance Function (hϕ​)**: This is a linear layer that takes the probability distribution ui​ and generates a corresponding feature attention map, gi​. 13
    

A

**masked state**, s^i​, is created by using the attention map to combine the original state si​ and a baseline state sbaseline​ (which represents the "absence" of features and is set to the environment's start state). 141414141414141414 The formula is:

s^i​=gi​⊙si​+(1−gi​)⊙sbaseline​

The model is trained by minimizing a composite loss function with three components15:

1. **Policy Fidelity Loss**: The mean squared error between the action distributions of the original policy given the original state π(⋅∣si​) and the masked state π(⋅∣s^i​). This ensures the learned attention maps are
    
    _faithful_ to the model's behavior. 16161616
    
2. **Clustering Loss**: The Kullback-Leibler (KL) divergence between the encoder's output and a target distribution that encourages purer, more confident cluster assignments. This helps form "tight" clusters. 17171717
    
3. **Sparsity Loss**: The L1 norm of the attention map (∣∣gi​∣∣1​), which encourages the model to produce simpler explanations by focusing on fewer important features. 18181818
    

---

### Implementation

- **Environments**: The method was tested on five environments with varying complexity: Mountain Car v0, Cart Pole v1, Acrobot v1, Flappy Bird v0, and Swimmer v4. 19
    
- **RL Algorithm**: All agent policies were trained using **Proximal Policy Optimization (PPO)**. 20
    
- **Frameworks**: The implementation utilized the CleanRL library and PyTorch. 21
    

---

### Result & Evaluation

ASAP was evaluated on its ability to produce faithful and meaningful explanations.

- **Qualitative Insight**: The Flappy Bird example demonstrated that ASAP can provide intuitive insights, correctly identifying that the agent uses the position of pipes and its own rotation to time its flaps.
    
- **State Semantics**: Using t-SNE visualizations and silhouette scores, the authors showed that ASAP effectively groups semantically similar states together, even though this is not an explicit objective. 23232323
    
- **Hyperstate Fidelity**: By replacing the complex policy with a single deterministic action for each hyperstate, the agent was still able to perform well. This indicates that ASAP successfully groups states where the policy behaves consistently. 24242424
    
- **Faithfulness**: The experiments showed that using ASAP's attention maps to mask state features did not degrade the agent's performance, confirming that the maps are faithful in identifying the most critical information for decision-making. 25252525
    

---

### Conclusion & Review

#### **Strengths**

- **Novelty and Interpretability**: The core idea of using _only_ feature attention for clustering is a significant contribution. It elegantly sidesteps the issue of uninterpretable clustering objectives (like state values) and grounds the explanation in a concept that is more accessible to humans.
    
- **Comprehensive Evaluation**: The paper provides a robust evaluation across multiple criteria (semantics, fidelity, faithfulness) and environments, offering strong evidence for the method's effectiveness.
    
- **Holistic Explanation**: The output of ASAP is not just an attention map but a complete summary, including a policy graph and representative states, which together provide a rich, multi-faceted view of the agent's strategy.
    

#### **Weaknesses**

- **Manual Hyperparameter Selection**: A key weakness is that the number of hyperstates, _k_, must be chosen manually. 26262626 This process is subjective, time-consuming, and hinders the method's scalability and automation.
    
- **Limited Scope of Application**: The method was only demonstrated on environments with low-dimensional, symbolic state spaces. 27Its applicability to high-dimensional inputs like pixels remains an open and significant challenge, which the authors acknowledge. 28
    
- **Lack of User Study**: As is common in many XAI papers, this work lacks a user study to empirically validate its primary claim: that the explanations are useful and understandable to humans. 29 The authors themselves identify this as an important area for future work.
    

Overall, ASAP is a well-designed and thoroughly evaluated method that introduces a promising new direction for creating global, interpretable summaries of RL policies. Its weaknesses primarily relate to its current stage of development, pointing to clear and important avenues for future research.

---

### Related Links and Concepts

- [Reinforcement Learning (RL)](https://en.wikipedia.org/wiki/Reinforcement_learning)
    
- [Explainable Artificial Intelligence (XAI)](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence)
    
- [Proximal Policy Optimization (PPO)](https://www.google.com/search?q=https://openai.com/research/proximal-policy-optimization-algorithms)
    
- [t-SNE (t-distributed Stochastic Neighbor Embedding)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)
    
- [Shapley Value](https://en.wikipedia.org/wiki/Shapley_value)
    
- [DeepLIFT](https://arxiv.org/abs/1704.02685)
    
- [Kullback–Leibler (KL) Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
### Note for training ASAP
- Training PPO script:
```bash
# Edit hyperparameters and env_id in ./configs/train_ppo.toml
python -m src.ppo --config-file train_ppo.toml
```
- Training ASAP script:
```bash
# Need to edit in the __main__ code and configurations 
python -m src.representation_learning.cluster_importance
```


- Training Time
- PPO 1M: 17m55' on RTX 3090
- PPO 10M: 1h57' on RTX 3090