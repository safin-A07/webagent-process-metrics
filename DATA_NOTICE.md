# Data notice

The files in `results/` are derived from three public sources. No raw trajectory, screenshot, accessibility tree,
task description or model output is redistributed.

## HAL traces (Holistic Agent Leaderboard, Online-Mind2Web)

Kapoor et al., Holistic Agent Leaderboard. Hugging Face `agent-evals/hal_traces`. The traces are encrypted by the
provider to limit benchmark contamination. This repository contains only per-trajectory metric values keyed by
configuration and task identifier, and SHA-256 checksums of the archives.

## Online-Mind2Web

Xue et al., An Illusion of Progress? Assessing the Current State of Web Agents (COLM 2025). Hugging Face
`osunlp/Online-Mind2Web`, CC-BY-4.0. Task identifiers, websites, difficulty levels and human reference lengths
appear in the derived files; task descriptions do not.

## AgentRewardBench

Lù et al., AgentRewardBench: Evaluating Automatic Evaluations of Web Agent Trajectories. Hugging Face
`McGill-NLP/agent-reward-bench`. The derived files contain per-trajectory metric values, token counts, and the
expert success, repetition and optimality labels. As required for derivatives, the dataset's terms of use are
reproduced here:

> By downloading this Dataset, you agree to comply with the following terms of use:
> - Restrictions: You agree not to use the Dataset in any way that is unlawful or would infringe upon the rights
>   of others.
> - Acknowledgment: By using the Dataset, you acknowledge that the Dataset may contain data derived from
>   third-party sources, and you agree to abide by any additional terms and conditions that may apply to such
>   third-party data.
> - Fair Use Declaration: The Dataset may be used for research if it constitutes "fair use" under copyright laws
>   within your jurisdiction. You are responsible for ensuring your use complies with applicable laws.
>
> Derivatives must also include the terms of use above.
