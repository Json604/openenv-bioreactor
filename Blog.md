# Training language models to run a bioreactor

*A submission to the OpenEnv Hackathon, India 2026.*

It is hour 180 of a 230-hour run inside a 100,000-liter industrial fermenter. Inside the tank, a few thousand kilograms of penicillin-producing fungus are alive and metabolizing — eating sugar, breathing oxygen, building product. On the operator's screen, the dissolved-oxygen reading begins to drift down. Twenty-two percent. Twenty-one. The safety floor is twenty.

The operator has minutes to decide. Cut the sugar feed and the cells will starve. Add more air or stir faster and they may recover, but push too hard and the foaming gets out of hand. Get this sequence wrong twice in a row and a multi-week batch — worth more than a million dollars in product and lost plant capacity — is gone.

## The capability gap

This is the kind of decision a sufficiently capable language model should, in principle, be able to help with. The state of the plant is a few dozen numbers and trends. The action is small — adjust the feed, adjust the air, adjust the agitator. The objective is well-defined. And yet, no current LLM has been trained to make this kind of decision well, and the reason has very little to do with intelligence.

The reason is that there is nowhere for one to practice. Real bioreactors are slow, expensive, and unforgiving. You cannot let a model learn by ruining real batches. You cannot give it a thousand attempts on a real plant. Until there is a place for a language model to act, fail, and be corrected at low cost, no amount of clever prompting will close the gap.

Previous papers built better autopilots. I built the simulator where future autonomous bioreactor operators learn the job.

## Why I built this

I came to this problem from work at a biotech startup. I had already been pulling parts of the IndPenSim model into internal pipelines, and had separately written a computer-vision tool for mixing-time analysis after the Kineticolor work published in *Organic Process Research and Development* (Houson and colleagues, 2022). What pushed me from "interesting" to "this is the right project" was the announcement, late last year, of Ginkgo Bioworks' autonomous laboratory built on top of OpenAI's GPT-5: cell-free protein-synthesis experiments designed and executed end to end by the model, with a forty percent reduction in cost and a twenty-seven percent lift in protein yield. That work targets the discrete-experiment side of biotechnology — which recipe to try next on the bench. The closed-loop *control* of a continuous physical process — keeping a hundred-thousand-liter fermenter alive for weeks, every twelve minutes, with safety constraints that are not negotiable — is an adjacent problem with a different shape, and as far as I can find, no public training environment for it existed. That is the gap I built this to fill.

## The simulator

The work was deliberately not to invent a new bioreactor model. The bioprocess engineering literature has spent decades calibrating simulators of fed-batch fermentations — runs in which fresh sugar is added gradually during production rather than dumped in once at the start — and a 2015 paper by Goldrick and colleagues is widely treated as the standard reference for industrial penicillin. It ships with a detailed mathematical model: a system of 33 differential equations that describe how the cell biomass, the available sugar, the dissolved oxygen, the carbon dioxide produced, the fluid volume, and the rest of the plant state evolve over time.

I ported that simulator from MATLAB to Python and verified that the port reproduces the published trajectories on the same batch. The calibration overlay is in the repository.

## What the agent sees and does

From there, the question was what an agent should be allowed to see and do. The decision I made was strict: only what a real plant operator sees on a SCADA-style console — current measurements, recent trends, alarm flags, and the last action taken. The agent never sees the 33-dimensional state vector underneath. This matters. A model that reads the underlying equations directly is not learning to operate a plant; it is learning to fit a known function. Every twelve simulated minutes, the agent reads the console and outputs a small JSON object — a feed adjustment, an aeration adjustment, an agitation adjustment, and an optional short reason. Temperature and pH stay on automatic underneath, the way they do in a real plant.

## A reward designed not to be gamed

What the agent is rewarded for is the part most worth getting right. A single scalar reward — for example, "how much penicillin did you make" — would be easy for a model to game. It might learn to push aeration to the limit and accept dangerous oxygen drops because the productivity number rose anyway. Reinforcement learning is famously good at finding shortcuts that satisfy the letter of a reward without the spirit of it.

So I split the reward into seven independent components. One watches whether the JSON output is well-formed and in range. One watches whether dissolved oxygen stays above the safety floor. One watches whether penicillin is actually growing. One watches whether the available sugar stays in its healthy band. One watches whether temperature and pH remain near their setpoints. One penalizes wild swings in the controls. The last rewards the total amount of penicillin produced by the end of the batch. They are computed and logged independently. If the model finds a shortcut on any one of them, the other six pull it back, and the per-component log makes the shortcut visible immediately.

## The training run, honestly

I ran a small training experiment as a proof of life: two hundred optimization steps of GRPO on a three-billion-parameter open model, fine-tuned through a low-rank adapter rather than full retraining. The reward curve and the KL-regularized loss are in the README, with the wandb run linked alongside. The mean reward across the run stayed in the 0.40 to 0.46 band — small, expected, and honest. Two hundred steps on a sixty-four-prompt dataset with a frozen base model is not enough for a three-billion-parameter policy to find a meaningfully different operating mode inside a seven-component reward; the loop runs end to end and the gradients flow, but the behavior change has not yet emerged from the noise. The point of this run was never to ship a finished operator. It was to confirm that the environment, the rewards, and the training loop are wired correctly and that a longer or larger run could plausibly learn. They are.

## Even Claude Opus 4.7 loses to do-nothing

The harder and more interesting result came from the baseline comparison. On the held-out `do-recovery-medium` scenario, I rolled out the random policy, the do-nothing fixed-recipe baseline, the hand-written rule-based operator, and frontier Anthropic Claude Opus 4.7 in zero-shot mode. Mean episode reward across five seeds: roughly 23 for fixed-recipe, 19 for random, 15 for Claude, and 13 for the rule-based agent. The frontier model, with no training, lost to the do-nothing baseline. It wrote valid JSON the whole way, kept dissolved oxygen above the safety floor the whole way, and still lost — because it intervened too aggressively, and the control-effort penalty consumed the gain. That is what the trained operator's job is, in this environment: to learn when to act and when to wait. The fact that even Opus 4.7 cannot zero-shot the right cadence is exactly the capability gap this environment is built to teach into.

## Where this sits in prior work

Reinforcement learning is not new to bioprocess control, and I want to be honest about the lineage this work sits inside. A recent line of research applies deep reinforcement learning to fed-batch penicillin built on the same IndPenSim simulator (Li, Qiu and You, 2024), to pH control in a real industrial photobioreactor (Gil and colleagues, 2025), to temperature control in ethanol fermentation (Rajasekhar and colleagues, 2024), and to yeast cultivation under partial supervision (Pandian and Noel, 2018). I drew on this lineage. What makes the present work different is not the plant — penicillin has been studied before — but the agent and the interface. The prior systems train task-specific neural controllers that read state vectors and emit continuous control signals. The thing being trained here is a language model, reading an operator's console and emitting JSON, learning from a reward that is split across seven components precisely so it cannot be gamed by maximizing one of them.

## What this opens up

Each new bioprocess will still need its own plant adapter — fed-batch fermentations are how the world makes a large share of its antibiotics, biologics, vaccines, and industrial enzymes, and each has its own equations and faults — but the interface generalizes. Operator-style observations rather than raw physics, a small structured action vocabulary, composable rewards designed to resist gaming, a validated simulator underneath, and a training loop that produces a signal an LLM can learn from. The interesting question is not whether my agent got measurably better at penicillin in two hundred training steps. The interesting question is whether language models can eventually be trained to operate slow, dangerous, expensive physical systems, given somewhere to practice. This work is one early data point in favor of yes.

---

The environment, the training notebooks, the calibration report, and the baseline results are all in the [README](./README.md) and the [Hugging Face Space](https://huggingface.co/spaces/Json604/openenv-bioreactor).

## Postscript: what may change in the repository after the submission deadline

Late on submission day I attempted a fresh GRPO run on an H200 instance for faster wall-clock training. That flavor hit a CUDA initialization error inside the Hugging Face Jobs container that day and never reached the training loop, so at 16:55 IST I resubmitted the same script on an A10G GPU — purely to re-verify reproducibility on a different node. **The reward curve and artifacts shown above, and committed at submission time, are from the verified and complete training run that this submission stands on (W&B run [`1ycts2ex`](https://wandb.ai/personal-meta/bioperator-env/runs/1ycts2ex), 200 GRPO steps, finished, 83 minutes on H200 earlier the same day).** Anything pushed to this repository after 17:00 IST on 26 April 2026 is supplementary verification and does not change what was submitted.

## References

- Goldrick, S., Stefan, A., Lovett, D., Montague, G., Lennox, B. (2015). *The development of an industrial-scale fed-batch fermentation simulation.* Journal of Biotechnology, 193, 70–82. — the validated penicillin simulator (IndPenSim) that this environment ports to Python.
- Houson, I., et al. (2022). *Computer Vision for Kinetic Analysis of Lab- and Process-Scale Mixing Phenomena.* Organic Process Research and Development. [doi:10.1021/acs.oprd.2c00216](https://pubs.acs.org/doi/10.1021/acs.oprd.2c00216) — the Kineticolor work I drew on for an earlier mixing-time analysis tool at the same startup.
- Ginkgo Bioworks (2026). *Autonomous Laboratory Driven by OpenAI's GPT-5 Achieves 40% Improvement Over State-of-the-Art Scientific Benchmark.* PR Newswire, March 2026. [Press release](https://www.prnewswire.com/news-releases/ginkgo-bioworks-autonomous-laboratory-driven-by-openais-gpt-5-achieves-40-improvement-over-state-of-the-art-scientific-benchmark-302680619.html). — autonomous experimental-design lab on cell-free protein synthesis; the adjacent prior-art that motivated my interest in the closed-loop control problem this environment targets.
- Li, H., Qiu, T., You, F. (2024). *AI-based optimal control of fed-batch biopharmaceutical process leveraging deep reinforcement learning.* Chemical Engineering Science. — DRL applied to fed-batch penicillin control on IndPenSim.
- Gil, J. D., Del Rio Chanona, E. A., Guzmán, J. L., Berenguel, M. (2025). *Reinforcement learning meets bioprocess control through behaviour cloning: Real-world deployment in an industrial photobioreactor.* arXiv preprint. — RL pH control, deployed on a physical photobioreactor.
- Rajasekhar, N., et al. (2024). *Reinforcement learning based temperature control of a fermentation bioreactor for ethanol production.* — RL temperature control for ethanol fermentation.
- Pandian, J. B., Noel, M. M. (2018). *Control of a bioreactor using a new partially supervised reinforcement learning algorithm.* Journal of Process Control. — partially supervised RL on a yeast bioreactor.
