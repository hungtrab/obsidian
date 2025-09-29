
---

## Summary of "SkillTree: Explainable Skill-Based Deep Reinforcement Learning"

**Paper**: SkillTree: Explainable Skill-Based Deep Reinforcement Learning for Long-Horizon Control Tasks

**Authors**: Yongyan Wen, Siyuan Li, Rongchang Zuo, Lei Yuan, Hangyu Mao, Peng Liu

**arXiv ID**: [2411.12173v1](https://arxiv.org/abs/2411.12173)

**File**: ![[SkillTree2411.12173v1.pdf]]
### **TL;DR** 📝

SkillTree is a new hierarchical framework that makes deep reinforcement learning (DRL) more transparent and understandable, especially for complex, long-term robotic tasks. It tackles the "black-box" problem of typical DRL models by using a **decision tree** as its high-level policy. Instead of learning complex, continuous actions directly, SkillTree first learns a set of discrete, reusable "skills". The decision tree then simply learns which skill to select based on the current state, making the agent's decision-making process easy to interpret. The method achieves performance comparable to traditional neural network-based approaches while providing clear, skill-level explanations for its actions.

---

### **Motivation** 🤔
- **Limitations of Traditional Explainable Models**: Decision Trees (DTs) are a well-known alternative that are inherently explainable due to their simple, rule-based structure. However, they have significant drawbacks in complex scenarios:
    
    - **Long-Horizon Tasks**: They struggle to capture the extended decision-making processes required for long-term goals, leading to overly complex and hard-to-optimize trees.
    
    - **High-Dimensional Spaces**: They lack the expressive power to effectively handle the high-dimensional observation spaces common in robotics.
    
    - **Continuous Actions**: Their discrete nature (limited leaf nodes) makes it difficult to effectively represent the continuous control policies needed for precise robotic manipulation.
    

SkillTree aims to overcome these challenges by combining the strengths of both approaches: the hierarchical abstraction of skill-based learning and the inherent interpretability of decision trees.

---

### **Methodology** 🛠️

The SkillTree framework is implemented in three main stages:

#### 1. Learning Discrete Skill Embeddings 📚

The first step is to create a fixed set of meaningful, reusable behaviors, or "skills".

- **Data and Model**: This stage uses a task-agnostic offline dataset of trajectories (state-action sequences). A model inspired by **VQ-VAE** is used to process fixed-length segments of these trajectories.
    
- **Codebook Creation**: An encoder maps a trajectory segment to a continuous embedding vector ($z_e$). This vector is then "quantized" by finding the closest matching embedding in a learnable **"codebook"** or "skill table" (Z). This process forces the agent to learn a discrete set of K representative skills.
    
- **Policies and Prior**: A state-conditioned decoder serves as the low-level policy ($\pi_l$​), learning to reconstruct the original actions from a state and a given skill embedding ($z_q$). Simultaneously, a **skill prior** (p(k∣s)) is trained to predict the most likely skill based only on the initial state of a sequence, which helps guide exploration later on.    

#### 2. Training the Explainable High-Level Policy 🌳

With the discrete skills learned, the next stage is to train a high-level policy (πh​) for the specific downstream task.

- **Hierarchical Control**: The high-level policy operates at a lower frequency (e.g., every _h_ steps). At each decision point, it observes the current state and outputs a probability distribution over the discrete skill indices (0 to K-1).

- **Differentiable Decision Tree**: Crucially, this high-level policy is implemented not as a neural network, but as a **differentiable soft decision tree**. This allows the tree's parameters (decision boundaries at each node) to be optimized using standard backpropagation algorithms like Soft Actor-Critic (SAC).

- **Guided Learning**: The training is regularized by the previously learned skill prior. The objective function encourages the policy to not deviate too far from the prior's predictions, improving learning efficiency.


#### 3. Distilling into a Simpler Tree 🔬

The final soft decision tree is explainable but can still be complex. The final stage simplifies it further.

- **Imitation Learning**: The trained soft decision tree policy is used as an "expert" to generate new trajectories for the task.

- **Simplification**: A traditional, simpler "hard" decision tree algorithm like **CART** is then trained on this new dataset of state-skill pairs31.
    
- **Final Model**: This process distills the knowledge from the soft tree into a much simpler, more interpretable model with fewer parameters, while preserving high performance32323232.
    

---

### **Key Results** 📈

The experiments were conducted on several long-horizon robotic manipulation tasks (Franka Kitchen, CALVIN, Office) with sparse rewards33333333.

- **Competitive Performance**: SkillTree demonstrated learning efficiency and final performance comparable to competitive skill-based baselines that use neural networks (SPiRL, VQ-SPiRL)34. This proves that using a decision tree for the high-level policy does not lead to a significant performance drop35.
    
- **Superior Explainability**: The final distilled decision trees were very simple (e.g., depth of 3) yet highly effective36. The decision nodes clearly showed which state features (e.g., object positions) were critical for choosing a skill37.
    
- **Effective Distillation**: The distillation process, especially when combined with data cleaning (using only high-reward trajectories), often improved the final performance beyond the original soft tree and other baselines38. In contrast, a CART model trained to imitate a standard DRL agent performed poorly and resulted in a massive, uninterpretable tree39.
    

---

### **Contributions & Limitations** 🎯

#### Contributions

1. **First Application in its Class**: To the authors' knowledge, SkillTree is the first successful application of a DT-based explainable method to complex, long-horizon continuous control tasks40.
    
2. **Discrete Skill Representation**: The paper introduces an effective method for learning discrete skill representations, which simplifies the policy learning process and enhances explainability41.
    
3. **Balancing Performance and Explainability**: It demonstrates that it's possible to achieve performance on par with black-box neural network methods while providing significant transparency into the agent's decision-making process42.
    

#### Limitations

1. **Low-Level Policy is a Black Box**: The low-level policy (the decoder) is still a neural network and thus lacks explainability43. The transparency exists only at the higher, skill-selection level.
    
2. **Requires Offline Data**: The framework relies on the availability of a large, task-agnostic offline dataset for the initial skill learning phase, which may not always be practical to obtain44.
    

---

### **Related Concepts and Further Reading** 🔗

- **[Deep Reinforcement Learning (DRL)](https://en.wikipedia.org/wiki/Deep_reinforcement_learning)**: The field of machine learning that combines deep neural networks with reinforcement learning principles.
    
- **[Explainable AI (XAI)](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence)**: A subfield of AI focused on developing methods and models that allow humans to understand and trust the results and output created by machine learning algorithms.
    
- **[Decision Tree Learning](https://en.wikipedia.org/wiki/Decision_tree_learning)**: A predictive modeling approach used in machine learning which uses a decision tree to go from observations about an item to conclusions about the item's target value.
    
    - **[CART Algorithm](https://en.wikipedia.org/wiki/Classification_and_regression_tree)**: A specific algorithm for generating decision trees.
        
- **[Hierarchical Reinforcement Learning (HRL)](https://www.google.com/search?q=https://en.wikipedia.org/wiki/Hierarchical_reinforcement_learning)**: A branch of RL where the control problem is decomposed into a hierarchy of subproblems or sub-policies.
    
- **[Vector Quantized Variational Autoencoder (VQ-VAE)](https://arxiv.org/abs/1711.00937)**: A type of generative model that uses a discrete latent representation, which inspired the skill learning method in this paper.