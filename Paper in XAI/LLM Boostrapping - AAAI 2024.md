# Report on: Bootstrapping Cognitive Agents with a Large Language Model

- **Title**: Bootstrapping Cognitive Agents with a Large Language Model
- **Authors**: Feiyu (Gavin) Zhu, Reid Simmons
- **Source**: arXiv preprint arXiv:2403.00810v1 [cs.AI], submitted on February 25, 2024 - AAAI 2024

---

## 1. Introduction

The paper addresses a fundamental challenge in artificial intelligence: the trade-off between the broad, general-purpose knowledge of Large Language Models (LLMs) and the structured, interpretable reasoning of cognitive architectures.

- **Problem with LLMs**: While LLMs like GPT-4 possess vast world knowledge, they are often criticized for being prone to hallucination, difficult to fine-tune for specific domains, and their knowledge can be noisy or unreliable.
- **Problem with Cognitive Architectures**: Traditional cognitive architectures (like SOAR or ACT-R) provide excellent interpretability and are flexible to update, but they typically require extensive manual effort to program the initial set of behavioral rules (productions).

This work proposes a hybrid framework that combines the strengths of both approaches. The core idea is to use the noisy, commonsense knowledge from an LLM to "bootstrap" an agent built on a cognitive architecture. The LLM acts as an initial knowledge source to automatically generate production rules, while the cognitive architecture provides a framework to apply, verify, and refine these rules through interaction with an environment. This separates the process of **knowledge generation** (LLM) from **knowledge application** (cognitive agent), aiming for a system that is both knowledgeable and robust.

---

## 2. Related Work

The authors position their framework against several existing lines of research in AI and robotics.

- **Interactive Task Learning (ITL) and Program Synthesis**:
    - **Related Approaches**: These methods aim to teach robots new skills, often through human interaction. Some recent works use LLMs to reduce human effort, but still require supervision, such as answering yes/no questions to guide the learning process.
    - **Weakness**: They are often labor-intensive and depend on a human-in-the-loop for guidance and goal specification.
    - **This Paper's Advantage**: The proposed framework is designed to eliminate the need for direct human supervision by using strategic prompting and self-reflection mechanisms to generate and refine rules autonomously.

- **LLMs for Direct Plan or Code Generation**:
    - **Related Approaches**: Many studies use LLMs to directly generate entire scripts or plans (e.g., in Python or PDDL) to solve a specific robotics task.
    - **Weakness**: These generated plans are typically "situation-grounded," meaning they are tailored to a specific instance of a task and environment. They are not easily generalizable to new scenarios without re-generating the entire plan.
    - **This Paper's Advantage**: Instead of generating one-off plans, this framework generates parameterized and reusable production rules with learnable utility weights. This allows for greater generalization and the ability to choose the best action from multiple applicable rules.

- **LLMs for Direct Action Selection**:
    - **Related Approaches**: Other works use an LLM in a loop, querying it at each step to decide the next immediate action for the agent.
    - **Weakness**: This approach is not cost-effective, as it requires a high number of tokens for continuous operation. Furthermore, it is difficult to persistently update the LLM's knowledge from a single mistake or success.
    - **This Paper's Advantage**: By compiling LLM knowledge into persistent production rules, the agent becomes more efficient over time, significantly reducing the number of required LLM queries (and associated costs) during deployment.

---

## 3. Methodology

The proposed architecture consists of a cognitive agent with four key components that work in a perceive-plan-act cycle.

- **Agent Architecture**:
    1.  **World Knowledge**: A declarative memory of general facts (e.g., "tomatoes belong in the fridge"). When the agent encounters an unknown fact, it queries the LLM and stores the answer for future use.
    2.  **Environment Knowledge**: Represents the agent's current understanding of the environment based on its observations (e.g., "gripper is empty," "table is clear").
    3.  **Procedural Memory (Productions)**: A set of `IF-THEN` production rules that dictate the agent's behavior. These are the core of the agent's skills.
    4.  **Task Stack**: Manages the current hierarchy of tasks and subtasks, inspired by mechanisms in ACT-R and SOAR.

- **The Bootstrapping Loop**:
    The agent learns by starting with a curriculum of task families (e.g., `find a/an <object>`). For a given task, if the agent has an applicable production rule, it executes it. If it gets stuck (i.e., no rule applies), it triggers the core learning mechanism:
    1.  **Query LLM for Action**: The agent summarizes its current state, task, and available actions, and queries an LLM for the best next action. It uses Chain-of-Thought prompting to elicit detailed reasoning from the LLM.
    2.  **Generate Production Rule**: This is a key two-step process:
        - The LLM is first prompted to summarize its reasoning into a **generalized English description** of a production rule. For example, "if you need to find something, you should explore places where that object is commonly stored".
        - A second LLM query translates this English description into executable **Python code** that fits a predefined class structure (`precondition` and `apply` methods).
    3.  **Improve and Reinforce Rules**: The generated rules are not always perfect. The framework includes three mechanisms for refinement:
        - **Critic LLM for Over-Constraining**: After a task is learned, a critic LLM analyzes the rules and suggests modifications to remove unnecessary conditions, making them more general.
        - **Cycle Detection for Over-Generalization**: If the agent enters a repetitive loop, it detects the cycle and queries the LLM for a new, loop-breaking action, which can lead to a more nuanced rule.
        - **Reinforcement for Utility**: When a task is successfully completed, the production rules on the shortest path to success receive a positive utility update via the Bellman equation. This helps the agent prioritize more effective rules in the future.

---

## 4. Implementation

- **Simulator**: The experiments were conducted in the AI2THOR simulator, using kitchen environments. The agent has access to ground-truth object labels and properties when objects are nearby.
- **LLM**: The study used `GPT4-0613` for all tasks, with the temperature set to 0 to maximize determinism.
- **Bootstrapping Curriculum**: The agent was trained from scratch using a manually specified curriculum of five task families:
    1. `explore <receptacle>`
    2. `find a/an <object>`
    3. `pick and place a/an <object> in/on a/an <receptacle>`
    4. `slice a/an <object>`
    5. `put things on the countertop away`
- **Source Code**: The paper mentions that the code and full prompts are provided in supplementary material. A public repository is available here: [https://github.com/feiyuz/bootstrapping-cognitive-agents](https://github.com/feiyuz/bootstrapping-cognitive-agents).

---

## 5. Evaluation and Results

The bootstrapped agent was compared to an "action-only" baseline that used the same LLM and prompts to select an action at every single step.

- **Tasks**: The evaluation was performed on three tasks: `find an object`, `slice an object`, and `clear the countertops`.
- **Quantitative Results**: The results, summarized in Table 1 of the paper, were definitive.
    - **Token Efficiency**: The bootstrapped agent was vastly more efficient. For the `slice` and `clear` tasks, it used **0 tokens** during testing, relying entirely on its learned rules. In contrast, the action-only agent used over 100,000 tokens for the `slice` task alone. This demonstrates a significant reduction in computational cost and latency.
    - **Success Rate**: The bootstrapped agent achieved a 15/15 success rate on all tasks. The action-only agent failed one `find` trial (14/15) because it incorrectly assumed a "mug" was a "cup" and terminated its search prematurely. The rule-based nature of the bootstrapped agent prevented this type of logical error.
    - **Path Length**: While the number of steps was similar for the `find` and `slice` tasks, the bootstrapped agent took significantly *more* steps to complete the `clear the countertops` task. This was due to a learned "stylistic" rule that compelled it to find a *new empty cabinet* for each object, whereas the baseline agent put multiple objects into the same cabinet.

---

## 6. Conclusion and Critical Perspective

### Summary of Paper's Conclusion
The paper successfully demonstrates a framework for bootstrapping a cognitive agent with the knowledge of an LLM. This hybrid approach creates an agent that is significantly more token-efficient and slightly more reliable than a purely LLM-driven agent. By translating unstructured LLM knowledge into structured, reusable production rules, the framework fosters generalization, interpretability, and scalability. It presents a compelling alternative to end-to-end models and shows a promising path for building more robust and practical AI agents.

### Independent Critical Perspective
This is a high-quality paper that presents an elegant and well-executed idea. The fusion of symbolic (cognitive architecture) and sub-symbolic (LLM) approaches is a classic AI ambition, and this work provides a modern, practical implementation.

**Strengths**:
- **Pragmatism**: The framework directly addresses the critical issue of cost and latency in LLM-based agents. The results on token efficiency are not just incremental; they represent a fundamental improvement in operational viability.
- **Explainability**: By materializing the agent's logic into human-readable production rules, the system is far more transparent and debuggable than an end-to-end neural network. This is a significant advantage for safety-critical applications.
- **Novel Methodology**: The two-step rule generation (English description first, then code) is a clever technique to manage prompt complexity and create a modular workflow that could potentially incorporate human feedback at the English-language level.

**Potential Weaknesses and Open Questions**:
- **Scalability of Curriculum Design**: The process begins with a manually specified curriculum. While this is less work than writing every rule by hand, it may become a new bottleneck as the number of desired skills grows. How does the system perform when scaling to dozens or hundreds of tasks?
- **Reliance on a State-of-the-Art LLM**: The entire process hinges on the advanced reasoning capabilities of GPT-4. The performance of the bootstrapping process with less capable, open-source, or fine-tuned models is an open question. The quality of the initial "noisy knowledge" is critical, and lower-quality models might generate flawed or nonsensical rules that the refinement process cannot fix.
- **The Sim-to-Real Gap**: The authors rightfully acknowledge this limitation. The agent operates on perfect, symbolic state information from the simulator (e.g., object labels, properties like "is opened"). A real-world robot would have to contend with noisy, incomplete, and ambiguous sensor data from a perception system. The crisp preconditions of the generated Python rules (`IF gripper is empty`) would likely fail frequently in the real world, requiring a much more robust state estimation and rule-matching mechanism.
- **Optimality of Learned Behavior**: The "clear the countertops" result is telling. The agent learned a *consistent* and *valid* strategy, but not the most *efficient* one. This highlights that the agent is biased by the initial LLM suggestions and its limited experiences. It doesn't guarantee the discovery of an optimal policy, which could be a limitation in performance-critical domains.