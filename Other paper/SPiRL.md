# SPiRL Architecture

  

This document explains the architecture used in this repository for “Accelerating Reinforcement Learning with Learned Skill Priors,” including the encoder, decoder, learned skill prior, variants (state vs image; open-loop vs closed-loop), and where each component lives in the codebase.

  

Resources

- Paper: https://arxiv.org/abs/2010.11944

- Project site: https://clvrai.github.io/spirl/

- Main training scripts: `spirl/train.py` (skill prior learning), `spirl/rl/train.py` (downstream RL)

  
  

## Notation and shapes

  

- state_dim: dimension of the environment state vector (if state-based)

- action_dim: dimension of the action

- nz_vae: latent skill dimension z (default 10)

- nz_enc: encoder/feature dimension (default 32)

- n_rollout_steps: number of decoded steps per skill (default 10)

  

Unless stated otherwise, “context” refers to what the skill prior is conditioned on: either the first state frame or one/more image frames.

  
  

## Building blocks (where to find them)

  

- MLP predictor: `Predictor` in `spirl/modules/subnetworks.py`

- Sequence encoder (LSTM): `BaseProcessingLSTM` in `spirl/modules/recurrent_modules.py`

- Recurrent decoder (LSTM rollout): `RecurrentPredictor` in `spirl/modules/recurrent_modules.py`

- Image encoder (conv pyramid): `Encoder`/`ConvEncoder` in `spirl/modules/subnetworks.py`

- Mixture density head (GMM): `MDN` and `GMM` in `spirl/modules/mdn.py`

- Flow prior (RealNVP): `ConditionedFlowModel`, `RealNVP` in `spirl/modules/flow_models.py`

  

Direct links to code

- `spirl/models/skill_prior_mdl.py`

- `spirl/models/closed_loop_spirl_mdl.py`

- `spirl/modules/subnetworks.py`

- `spirl/modules/recurrent_modules.py`

- `spirl/modules/mdn.py`

- `spirl/modules/flow_models.py`

  
  

## Core architecture (common to variants)

  

Given a window of actions (and optionally context), the model:

1) infers a posterior q(z | ·) over a latent skill z

2) constructs a learned prior p̂(z | context)

3) samples a z (training: from q; RL: typically from p̂)

4) decodes z (with context) into a sequence of actions of length n_rollout_steps

  

Losses

- Reconstruction NLL: MSE-equivalent with unit-variance Gaussian over actions

- KL regularization: KL(q || p), where p is a fixed standard Gaussian; optional β tuning to meet a target KL

- Learned prior loss: by default NLL of p̂ at samples z_q from q (alternatively KL(q || p̂))

  

Key hyperparameters (overridable via configs)

- nz_vae=10, nz_enc=32, nz_mid=32, nz_mid_lstm=128

- n_lstm_layers=1, n_processing_layers=3

- n_rollout_steps=10

- learned_prior_type ∈ {gauss, gmm, flow}

- n_prior_nets (prior ensemble size), num_prior_net_layers=6, nz_mid_prior=128

  
  

## Variant A: State-based SPiRL (open-loop decoder)

  

Class

- `SkillPriorMdl` in `spirl/models/skill_prior_mdl.py`

  

Inference network q(z | actions[, state0])

- If cond_decode=False: q sees only actions

- If cond_decode=True: q sees actions concatenated with repeated context (state at t=0)

- Implementation: `BaseProcessingLSTM(..., in_dim=..., out_dim=nz_enc)` → `Linear(nz_enc → 2*nz_vae)` → `MultivariateGaussian`

  

Learned skill prior p̂(z | context)

- Context: `inputs.states[:, 0]` (first state)

- Prior types:

- Gaussian: `Predictor(context → 2*nz_vae)` → `MultivariateGaussian`

- GMM: `Predictor(context → nz_mid)` → `MDN(nz_mid → mixture over nz_vae)` → `GMM`

- Flow: `ConditionedFlowModel` with a context MLP and RealNVP layers → `FlowDistributionWrapper`

- Ensemble: `nn.ModuleList` of size `n_prior_nets`; trained via NLL of p̂ on z_q; loss is aggregated per-prior

  

Decoder (open-loop, autoregressive LSTM)

- `RecurrentPredictor` with `ForwardLSTMCell`

- Cell input per step: `[prev_action, z]`

- Initialization:

- If cond_decode=True: two MLP initializers from context (for the LSTM’s input and hidden state)

- Else: learned constant trainable parameters

- Output: predicted action sequence of length `n_rollout_steps`

  

Training/data flow (state-based)

1) q from actions (+ context if cond_decode)

2) fixed prior p for KL; learned prior p̂ from context

3) sample z (training from q; RL typically from p̂)

4) decode z (+ context) into actions

5) compute reconstruction + KL + prior losses

  

Relevant methods

- `SkillPriorMdl.build_network`, `_build_inference_net`, `_build_prior_net`, `_build_decoder_initializer`, `decode`, `loss`

  
  

## Variant B: Image-based SPiRL (open-loop decoder)

  

Class

- `ImageSkillPriorMdl` in `spirl/models/skill_prior_mdl.py`

  

Differences from Variant A

- Context comes from images (first n_input_frames). Prior input pipeline:

- `ResizeSpatial(prior_input_res)` → `Encoder` (conv pyramid) → `RemoveSpatial()` → feature

- Then the same prior head as in Variant A (Gaussian/GMM/Flow)

- If cond_decode=True: a `cond_encoder` encodes the context images into a vector that is fed to both q and the decoder initializers

- Targets: action sequence is aligned to exclude the leading frames used as image context

  

Relevant methods

- `_build_prior_net` (wraps `Encoder` before the prior head)

- `_build_inference_net` (uses `cond_encoder` when cond_decode=True)

- `_build_decoder_initializer`, `_learned_prior_input`, `_regression_targets`

  
  

## Variant C: Closed-loop low-level decoder (state-based)

  

Class

- `ClSPiRLMdl` in `spirl/models/closed_loop_spirl_mdl.py`

  

Key differences vs open-loop

- Requires `cond_decode=True`

- Decoder is a per-timestep MLP (no LSTM rollout):

- `Predictor(input_size = enc_size + nz_vae, output_size = action_dim, mid_size = nz_mid_prior)`

- `enc_size = state_dim`

- Decoding uses the observed sequence encoding: `seq_enc = inputs.states[:, :-1]` and concatenates each step with z

- Inference q conditions on `[actions, seq_enc]`

  

Relevant methods

- `build_network`, `decode`, `_build_inference_net`, `_run_inference`, `_get_seq_enc`

  
  

## Variant D: Closed-loop low-level decoder (image-based)

  

Class

- `ImageClSPiRLMdl` in `spirl/models/closed_loop_spirl_mdl.py`

  

Differences vs C

- Uses an image encoder to compute `seq_enc` from stacked image frames (across the sequence)

- `enc_size = nz_enc` (encoded image feature size)

- The learned prior path for images is reused from `ImageSkillPriorMdl`

  

Relevant methods

- `_build_inference_net` (builds `img_encoder`), `_get_seq_enc` (encodes stacked image sequences), `enc_obs`

  
  

## Learned skill prior options (details)

  

Gaussian prior

- Code: `_build_prior_net` (default) in `spirl/models/skill_prior_mdl.py`

- Architecture: `Predictor(context → 2*nz_vae)` → `MultivariateGaussian`

  

GMM prior (MDN)

- Code: `_build_prior_net` with `learned_prior_type == 'gmm'`

- Architecture: `Predictor(context → nz_mid)` → `MDN(input=nz_mid, output=nz_vae, num_gaussians)` → `GMM`

  

Flow prior (RealNVP)

- Code: `_build_prior_net` with `learned_prior_type == 'flow'`, and `spirl/modules/flow_models.py`

- Architecture: `ConditionedFlowModel(context_mlp → cond_feature)` + a stack of `RealNVP` bijectors

- Interface: wrapped in `FlowDistributionWrapper` to expose `sample`, `log_prob`, etc.

  
  

## Losses and β tuning

  

- Reconstruction: `NLL` between a unit-variance Gaussian with mean = `decoder(...)` and the ground-truth action sequence

- KL loss: `KLDivLoss(beta)` with `beta = kl_div_weight` or a dual-optimized β to reach `target_kl`

- Prior loss: either `NLL(p̂, z_q.detach())` (default) or `KL(q.detach() || p̂)`

- Ensemble prior loss is averaged per prior in the ensemble

  

Relevant code

- `SkillPriorMdl.loss`, `_compute_learned_prior_loss`, `_get_beta_opt`, `_update_beta`

  
  

## RL usage (low-level policy interface)

  

- `SkillPriorMdl.run(...)` (state-based) samples z from the learned prior (or uniform if disabled), decodes `n_rollout_steps` actions into an action plan, and returns one action per call. The plan is cleared on `reset()`.

- `ImageSkillPriorMdl.unflatten_obs(...)` helps split concatenated [state, image] observations when using image-based context in RL.

  
  

## Quick map from concepts to code

  

- State-based, open-loop: `spirl/models/skill_prior_mdl.py` → `SkillPriorMdl`

- Image-based, open-loop: `spirl/models/skill_prior_mdl.py` → `ImageSkillPriorMdl`

- State-based, closed-loop: `spirl/models/closed_loop_spirl_mdl.py` → `ClSPiRLMdl`

- Image-based, closed-loop: `spirl/models/closed_loop_spirl_mdl.py` → `ImageClSPiRLMdl`

- LSTM utilities: `spirl/modules/recurrent_modules.py`

- Conv/MLP modules: `spirl/modules/subnetworks.py`

- MDN/GMM: `spirl/modules/mdn.py`

- Flow prior: `spirl/modules/flow_models.py`

  
  

## Minimal shape summary (defaults)

  

- q input: actions (T × action_dim), optionally + context feature; output: `2*nz_vae`

- p̂ input: context feature (state0 or images) → distribution over `z ∈ R^{nz_vae}`

- Decoder inputs: z (+ context). Open-loop: LSTM over `n_rollout_steps`. Closed-loop: per-step MLP on `[enc_t, z]`.

- Outputs: actions over `n_rollout_steps`.

  
  

---

  

If you need a diagram tailored to a specific environment/config (e.g., Kitchen with image-based closed-loop), let me know which config you’re running and I’ll annotate the exact data flow for that setup.