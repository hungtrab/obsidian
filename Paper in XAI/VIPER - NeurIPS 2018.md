
---

# Summary: Verifiable Reinforcement Learning via Policy Extraction

**Paper:** Verifiable Reinforcement Learning via Policy Extraction

**Authors:** Osbert Bastani, Yewen Pu, Armando Solar-Lezama (MIT)

**Publication:** 32nd Conference on Neural Information Processing Systems (NeurIPS 2018)

---

## Abstract

This paper proposes a novel approach to learn **verifiable** policies by training **decision trees**. Decision trees are non-parametric and can represent complex policies, yet their structured nature makes them amenable to efficient formal verification.

The core contribution is an algorithm called **VIPER** (Verifiability via Iterative Policy ExtRaction), which extracts a decision tree policy from a high-performance Deep Neural Network (DNN) policy (the "oracle"). VIPER is shown to learn compact and high-performing decision trees for tasks like Atari Pong and Cart-Pole, which are then formally verified for properties such as robustness, correctness, and stability.

---

## 1. Problem Statement

The central problem is the lack of safety guarantees for policies learned by Deep Neural Networks in RL. DNNs function as "black boxes," making it computationally expensive, if not impossible, to formally verify critical properties like:

- **Safety:** The policy will not enter an unsafe state.
    
- **Stability:** The system will converge to a desired equilibrium (e.g., a robot balancing).
    
- **Robustness:** Small perturbations in the input state will not lead to drastically different actions.

---
## 2. Proposed Solution: VIPER

The paper's solution is a two-stage process:

1. First, train a high-performance (but non-verifiable) DNN policy using standard DRL techniques. This DNN acts as an **oracle**.
    
2. Second, use the
    
    **VIPER** algorithm to distill the knowledge from the oracle into a simple, structured, and verifiable decision tree policy that mimics the oracle's performance10.
    

This approach combines the learning power of DNNs with the verifiability of decision trees.

---

## 3. Methodology

VIPER is built upon an improved imitation learning algorithm called **Q-DAGGER**.

### 3.1. Q-DAGGER: A Theoretical Foundation

The authors first propose Q-DAGGER, an extension of the well-known imitation learning algorithm **[DAGGER](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf)** (Dataset Aggregation).

- **Standard DAGGER:** Uses a 0-1 loss, treating all mistakes equally. It only cares if the learner's action matches the expert's.

- **Q-DAGGER's Innovation:** It uses a more informative loss function derived from the oracle's **[Q-function](https://en.wikipedia.org/wiki/Q-learning)**. The loss, lt​(s,π)=Vt(π∗)​(s)−Qt(π∗)​(s,π(s)), measures the "regret" or performance drop from taking the learner's action instead of the expert's optimal one. This forces the algorithm to prioritize learning correctly in **critical states** where mistakes are costly.
### 3.2. VIPER: Policy Extraction for Decision Trees

VIPER is the practical algorithm that adapts the Q-DAGGER principle to train decision trees using the **[CART algorithm](https://www.google.com/search?q=https://en.wikipedia.org/wiki/Decision_tree_learning%23CART)**. Since directly modifying CART's loss function is difficult, VIPER uses a clever **resampling** technique:

1. **Aggregate Data:** Collect state-action pairs by running the current policy and having the oracle label them.

2. **Resample with Weights:** Create a new training dataset by resampling from the aggregated data. The probability of a state being sampled is proportional to the Q-DAGGER loss at that state.

3. **Train:** Train a new decision tree on this weighted, resampled dataset.

This process ensures the decision tree training focuses on the most critical states, resulting in a more compact and robust policy.

---

## 4. Verification Techniques

Once a decision tree policy is learned, its structured nature allows for efficient verification of several properties.

### 4.1. Correctness

- **Goal:** Prove that the policy never enters a failure state (e.g., never loses a game of Pong).
    
- **Method:** The system dynamics and the piecewise-linear decision tree policy are encoded as a logical formula. An **[SMT Solver](https://en.wikipedia.org/wiki/Satisfiability_modulo_theories)** (like Z3) is used to check if there is any satisfying assignment for the _negation_ of the correctness property. If not, the property is proven to hold.
### 4.2. Stability

- **Goal:** Prove that the system will asymptotically reach a stable goal state (e.g., the cart-pole is balanced at the center).

- **Method:** This uses concepts from control theory, specifically **[Lyapunov stability](https://en.wikipedia.org/wiki/Lyapunov_stability)**. The authors use **[Sum-of-Squares (SOS) Optimization](https://www.google.com/search?q=https://www.cds.caltech.edu/archive/help/part_2/sos.html)** to automatically find a Lyapunov function that proves stability for the system within a certain region of attraction.
### 4.3. Robustness

- **Goal:** Find the largest perturbation (within an L∞​ ball) to an input state that does not change the policy's output action.

- **Method:** For a decision tree, this can be formulated and solved efficiently as a **[Linear Program](https://en.wikipedia.org/wiki/Linear_programming)**. This is orders of magnitude faster than verifying robustness for DNNs.
---
## 5. Key Experiments & Results

- **Atari Pong (Robustness):** VIPER extracted a 769-node tree that achieved a perfect score, same as the DNN oracle21. Robustness verification took **~3 seconds** per state, while the state-of-the-art DNN verifier (Reluplex) took minutes to over an hour.

- **Toy Pong (Correctness):** The verification process successfully **found a counterexample (a bug)** in the learned policy. The authors then manually fixed the policy (by extending the paddle length slightly) and re-ran the verifier to prove the fixed system was now 100% correct.

- **Cart-Pole (Stability):** A 3-node tree learned by VIPER achieved a perfect score. Stability was verified in **3.9 seconds** using SOS, identifying a large region of attraction, whereas enumerative methods were orders of magnitude slower and less effective.
 
- **Comparison to Baselines:** VIPER produced decision trees an **order of magnitude smaller** than standard DAGGER for the same performance level on Pong, making them much easier to verify.

---

## 6. Main Contributions

1. **A Novel Framework:** Proposes a general framework for learning verifiable RL policies by extracting decision trees from DNN oracles.

2. **VIPER Algorithm:** Introduces VIPER, an effective imitation learning algorithm that leverages an oracle's Q-function to learn compact and high-performance decision tree policies.

3. **Demonstrated Verifiability:** Shows how existing verification techniques can be applied efficiently to decision tree policies to prove important properties like correctness, stability, and robustness on common RL benchmarks.


---

## 7. Related Concepts & Further Reading

- [Markov Decision Process (MDP)](https://en.wikipedia.org/wiki/Markov_decision_process)
    
- [Reinforcement Learning](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
    
- [Imitation Learning](https://www.google.com/search?q=https://lilianweng.github.io/posts/2022-05-30-imitation-learning/)
    
- [Knowledge Distillation (Model Compression)](https://www.google.com/search?q=https://www.hinton.ai/distillation.html)
    
- [Formal Verification](https://en.wikipedia.org/wiki/Formal_verification)