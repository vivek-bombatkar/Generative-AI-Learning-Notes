# My Learning Notes from [NUS generative-ai-fundamentals-to-advanced-techniques-programme](https://nus.comp.emeritus.org/generative-ai-fundamentals-to-advanced-techniques-programme)

---

# Reinforcement Learning vs Supervised Learning

## Context
- **Course:** Advanced Generative AI
- **Module/Lecture:** Learning Paradigms
- **Date:** TODO: unclear in source

## Key ideas in one minute
- Supervised learning learns from labelled examples.
- Reinforcement learning learns from interaction and feedback.
- Supervised learning minimizes prediction error.
- Reinforcement learning maximizes long-term reward.
- Reinforcement learning relies on trial and error.
- Supervised learning has no notion of an environment.

## Definitions
- **Supervised Learning:** A learning paradigm where a model is trained on labelled input-output pairs and receives direct feedback on errors.
- **Reinforcement Learning (RL):** A learning paradigm where an agent learns by interacting with an environment using rewards and penalties.

## Core concepts explained

### Supervised Learning
- **What it is**
  - Learning from labelled datasets.
- **Why it matters**
  - Highly effective when historical data with correct answers is available.
- **How it works**
  1. Provide input data with labels.
  2. Model makes predictions.
  3. Compute error against true labels.
  4. Update model to minimize error.

### Reinforcement Learning
- **What it is**
  - Learning through interaction with an environment using rewards and penalties.
- **Why it matters**
  - Suitable for sequential decision-making problems where labels are not available.
- **How it works**
  1. Agent observes the environment state.
  2. Agent takes an action.
  3. Environment returns a reward or penalty.
  4. Agent updates its policy to maximize long-term reward.

## Architectures and workflows

### Supervised Learning Workflow
- Dataset with labels
- Model prediction
- Loss calculation
- Gradient-based optimization

### Reinforcement Learning Workflow
- Agent
- Environment
- Reward signal
- Policy update loop

## Algorithms / Step-by-step

### Supervised Learning
1. Collect labelled data.
2. Split into training and validation sets.
3. Train model to minimize loss.
4. Evaluate performance.

### Reinforcement Learning
1. Initialize agent and environment.
2. Perform actions based on current policy.
3. Receive reward feedback.
4. Update policy using reward signal.
5. Repeat until convergence.

## Examples
- **Supervised Learning:** Image classification, spam detection, price prediction.
- **Reinforcement Learning:** Game playing, robotics control, recommendation strategies with delayed rewards.

## Common mistakes and pitfalls
- Assuming reinforcement learning requires labelled data.
- Using reinforcement learning when a supervised dataset is sufficient.
- Ignoring long-term reward in reinforcement learning.
- Treating reinforcement learning feedback as direct error signals.

## Quick revision checklist
- [ ] Does the problem have labelled data?
- [ ] Is there an environment interaction loop?
- [ ] Are rewards delayed or immediate?
- [ ] Is long-term optimization required?

## Open questions / TODO
- TODO: examples of specific RL algorithms not shown in source


---
## Visualise Deep Learning Models:
  - https://projector.tensorflow.org/
  - https://adamharley.com/nn_vis/cnn/3d.html
