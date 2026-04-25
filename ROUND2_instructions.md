```text
BioOperatorEnv: Full project handoff in simple words

1. Project name

Main name:
BioOperatorEnv

Other names we thought about:
- BioControlEnv
- FermenterControlEnv
- IndPenControlEnv
- Autonomous Bioreactor Operator Playground

Why "BioOperatorEnv" is the best:
- "Control" sounds like a normal control algorithm project.
- "Operator" better explains what we are doing.
- The agent is acting like a plant operator:
  it reads the plant situation,
  understands trends and alarms,
  picks the next action,
  and is scored on whether it kept the process safe and productive.

One-line version:
BioOperatorEnv is a training playground for autonomous bioreactor operators.

Slightly longer one-line version:
BioOperatorEnv is an OpenEnv environment where LLM/RL agents learn to safely run industrial bioreactors before touching real plants.

Best simple metaphor:
BioOperatorEnv is a flight simulator for autonomous bioreactor agents.


2. Core vision

Long-term vision:
Fully autonomous biomanufacturing plants of any scale.

What we are doing right now:
Before autonomous agents can be trusted in real bioreactor plants, they need a safe and realistic place to practice. They need to learn process behavior, make mistakes, recover from disturbances, and show that they can run a batch better than simple baseline methods.

Important:
We are NOT claiming that we already solved fully autonomous biomanufacturing.

We ARE claiming:
Autonomous bioreactor agents need a training playground, and we are building that playground using a validated industrial fermentation simulator as the plant engine.

Why this matters:
Real bioprocess batches are expensive, slow, safety-sensitive, and product-quality-sensitive.
You cannot let an AI "experiment" on a real 100,000 L batch.
If it overfeeds, starves oxygen, drifts pH, overheats, or damages cells, the plant loses time, materials, product, and quality.

Core thesis:
Autonomous bioreactor agents cannot learn by ruining real batches.
They need simulator-backed practice first.


3. How this fits the hackathon

The hackathon is not asking for a normal app.
It is asking for an environment where an LLM can act, get feedback, and improve using reinforcement learning.

The shared guide says the stack is:
environment -> reward/checking -> TRL trainer -> Unsloth -> OpenEnv / Spaces

The guide also says a good task should have:
1. Step-by-step actions
2. Success that can be checked automatically
3. A task that is hard, but not so hard that the model never succeeds

BioOperatorEnv fits well because:
- The agent acts step by step during a batch
- The simulator can automatically check whether the agent kept the process safe and productive
- We can start with easy tasks and slowly make them harder

Best matching hackathon theme:
World Modeling / Professional Tasks

Why:
The bioreactor is a changing world.
The agent does not see everything.
It has to read measurements, infer what is happening, and choose safe actions.

That is exactly the kind of environment the hackathon wants.

What judges want:
- A new and interesting environment
- A clear story
- Real training improvement
- Good reward design
- Working OpenEnv setup
- A demo people can understand quickly

That is what we should optimize for.


4. What problem are we solving?

Simple version:
Bioprocess plants cannot let AI agents learn on real batches, because mistakes are expensive, slow, unsafe, and can ruin product quality.
BioOperatorEnv gives the agent a realistic simulator-backed place to learn bioreactor operation before touching a real plant.

More detailed version:
Industrial bioprocesses are expensive, nonlinear, changing over time, and full of disturbances.
Real operators constantly adjust feed, aeration, agitation, pH correction, temperature control, and other process factors while trying to protect product yield and avoid unsafe conditions.
Existing RL papers show that RL can improve certain controllers, but there is still a need for a reusable OpenEnv environment where LLM/RL agents can practice autonomous bioreactor operation under realistic disturbances, limits, and safety rules.

Exact problem statement:
BioOperatorEnv solves the missing training-infrastructure problem for autonomous bioreactor agents.
It turns a validated industrial fermentation simulator into an OpenEnv environment where an LLM/RL agent can read process measurements, choose safe actions, get objective rewards, and improve before deployment to real equipment.


5. Why this is a real problem

Bioprocesses are hard because the "machine" is alive.

In a normal machine:
If you change input X, output Y is often somewhat predictable.

In a bioreactor:
Cells grow, eat nutrients, create heat, change pH, use oxygen, make product, and behave differently over time.

The photobioreactor paper explains that the real production work is done by living cells, but operators can only change bigger things like pH, temperature, nutrients, oxygen, and CO2.

So the real operator problem is:
Given the readings I have right now, what should I adjust so the batch stays safe and productive?

That is a real problem and a good fit for RL.


6. What "incomplete readings" means

A real bioreactor operator does not know everything happening inside the cells.

The operator may measure things like:
- temperature
- pH
- dissolved oxygen
- volume
- feed rate
- air flow
- agitation speed
- pressure
- off-gas O2 / CO2
- maybe biomass estimate
- maybe product estimate
- maybe substrate estimate

But the operator usually does NOT directly know:
- exact cell metabolic state
- true oxygen use at every point in the tank
- exact local substrate around every cell
- cell stress level
- future heat generation
- future growth rate
- exact contamination risk
- true product formation ability

This means the agent should not see the full internal simulator truth.

The agent should see a plant-console view.

Example observation:
{
  "time_h": 48.0,
  "batch_phase": "production",
  "temperature_C": 25.4,
  "temperature_setpoint_C": 25.0,
  "pH": 6.42,
  "pH_setpoint": 6.50,
  "dissolved_oxygen_pct": 21.0,
  "DO_min_safe_pct": 18.0,
  "substrate_g_L": 0.11,
  "penicillin_g_L": 1.82,
  "biomass_g_L": 23.4,
  "volume_L": 84000,
  "feed_rate_L_h": 78,
  "aeration_rate_vvm": 0.85,
  "agitation_rpm": 120,
  "recent_trend": {
    "DO": "falling_fast",
    "pH": "slowly_decreasing",
    "temperature": "stable"
  },
  "alarm": "DO_near_low_limit",
  "previous_action": {
    "feed_delta_L_h": 5,
    "aeration_delta_vvm": 0,
    "agitation_delta_rpm": 0
  }
}

Important:
The agent does NOT see the ODE equations.
The agent does NOT see every hidden biological variable.
The agent sees what an operator would reasonably see.

That is what makes this an operator-training environment.


7. What previous papers already prove

We should not pretend this area has no prior work.
It does.
That is actually helpful, because it proves the domain is real.

7.1 Penicillin / IndPenSim RL paper

What it proves:
- Industrial fed-batch bioprocess control is still an open challenge
- The process changes over time
- The full process state is not fully visible
- There is uncertainty from batch to batch
- RL can improve a specific penicillin controller

Important for us:
We must NOT claim:
"We are the first to use RL for penicillin control"
or
"We are the first to use IndPenSim with RL"

That is false.

Correct claim:
Prior work shows RL can improve one penicillin controller.
We are building an OpenEnv environment for LLM/RL operator training, using the simulator as the plant engine.

7.2 Photobioreactor pH RL paper

What it proves:
- RL can be used for pH control
- It is useful to start from offline data before real deployment
- Real-world online experimentation is costly and risky

Important for us:
This strongly supports our "playground before plant" story.

It is still a controller paper.
It is not a reusable OpenEnv LLM-agent environment.

7.3 Ethanol temperature RL paper

What it proves:
- RL has been used for temperature control in fermentation
- RL can reject disturbances and be compared against other RL methods

Important for us:
Our novelty is NOT "RL controls temperature."
That already exists.

Correct claim:
Existing papers show specific RL controllers for specific variables.
We are building a reusable environment for LLM agents.

7.4 Yeast bioreactor RL paper

What it proves:
- Pure RL can be slow and hard because it has to explore
- A warm start or guided learning can help

Important for us:
This supports our plan to:
- start from an existing instruct model
- possibly give it simple formatting examples first
- then use RL to improve it

So prior work is not a threat.
It is evidence that:
- the space is real
- RL is promising
- safe practice before deployment matters


8. What we are NOT claiming

Do NOT say:
- We are the first to use RL for bioreactor control
- We are the first to use IndPenSim with RL
- We built the best controller in the world
- One trained model will control every bioreactor
- This is not a simulator

Instead say:
- We are not selling a simulator by itself
- We are using a validated simulator as the physics/biology engine for an agent-training environment


9. What we ARE claiming

Correct claim:
Prior work proves that RL can improve certain bioprocess controllers, including temperature control, pH control, yeast fermentation control, and penicillin control.
But those works are mostly one-controller studies.
BioOperatorEnv solves the missing environment layer:
an OpenEnv-compatible training playground where LLM/RL agents can learn autonomous bioreactor operation under incomplete readings, disturbances, control limits, and safety-based rewards.

Best short framing:
Previous papers built better autopilots.
We are building the simulator where future autonomous bioreactor operators learn to fly.


10. What exactly is different from previous papers?

Most earlier papers do something like this:

one bioprocess problem
-> one simulator/model
-> one RL algorithm
-> one controller
-> compare against PID/MPC/ANN

Examples:
- one pH controller
- one temperature controller
- one penicillin controller
- one yeast controller

BioOperatorEnv does this:

bioprocess simulator
-> OpenEnv environment
-> plant-console observations
-> structured JSON actions
-> reward made of separate parts
-> fault and disturbance scenarios
-> train/evaluate LLM/RL agents
-> before/after behavior demo

That is the key difference.

Classic controller:
state numbers -> action numbers

Our operator-style agent:
plant situation + trends + alarms + limits -> structured action + optional reason

Example observation:
Current batch situation:
- DO is falling quickly
- feed was increased recently
- pH is stable
- agitation is near upper limit
- aeration still has room
- product formation is slowing
Choose the next safe operator action.

Example action:
{
  "feed_delta_L_h": -3,
  "aeration_delta_vvm": 0.10,
  "agitation_delta_rpm": 0,
  "cooling_delta": 0,
  "acid_base_action": "none",
  "reason": "DO is falling after a feed increase. Reduce oxygen demand slightly and increase aeration first because agitation is already near limit."
}

This is not just "control".
It is decision-making under plant-like conditions.


11. How BioOperatorEnv works

BioOperatorEnv has two main layers.

11.1 Plant engine

This is the actual simulator or digital twin.

For the MVP, this is the MATLAB penicillin simulator / IndPenSim-style penicillin process model.

It contains the process behavior:
- growth
- substrate use
- product formation
- oxygen transfer
- heat generation
- volume changes
- feed effects
- pH effects
- fault / anomaly logic

Important:
The LLM does NOT invent the bioreactor behavior.
The simulator decides what physically/biologically happens.

If the agent increases feed too much:
- substrate may rise
- oxygen demand may rise
- DO may fall
- productivity may suffer
- safety penalty may happen

If the agent increases aeration:
- oxygen transfer may improve
- DO may recover
- but maybe extra control effort penalty happens

This is how the system is "physics-aware":
The plant response comes from the simulator, not from the LLM.
The LLM only chooses actions.

11.2 Agent-training interface

This is the OpenEnv wrapper.

It defines:
- reset()
- step(action)
- state() or observation
- reward
- done
- info

The wrapper decides:
- what the agent sees
- what actions are allowed
- how long each step is
- what counts as unsafe
- what reward is given
- when the episode ends
- how results are logged
- how to avoid reward cheating


12. Current MATLAB simulator problem

The current MATLAB code probably runs like a full-batch generator.

Something like:
- run batch 1 from start to end
- run batch 2 from start to end
- faults already defined in code
- duration predefined
- number of batches predefined
- control recipe predefined
- save all outputs
- finish

That is like pressing play on a movie.

But RL needs a video game.

The agent must be able to act during the run.

So the key implementation task is:
Turn the full-batch MATLAB simulator into a step-by-step simulator.

Current style:
run_entire_batch()

Needed style:
reset_batch()
step_batch(action)
step_batch(action)
step_batch(action)
...

That is the main conversion.


13. Simulator conversion plan

13.1 Required MATLAB functions

We need at least two MATLAB functions.

Function 1: reset_batch

Purpose:
Start a fresh batch or a chosen scenario.

Pseudo-MATLAB:
function state = reset_batch(seed, scenario)
    % set random seed
    % load initial conditions
    % set batch time
    % choose fault/disturbance scenario
    % set default recipe/control values
    % initialize hidden simulator state
    % return state
end

State should contain:
- current time
- full internal simulator state
- current control values
- scenario information
- fault flags
- previous action
- any internal memory needed

Function 2: step_batch

Purpose:
Apply one agent action and run the simulator forward by one control interval.

Pseudo-MATLAB:
function [next_state, obs, info] = step_batch(state, action, dt)
    % validate action
    % clip action to safe limits
    % apply action to control variables
    % integrate process from t to t + dt
    % update hidden state
    % generate noisy operator-facing observations
    % return next state, observation, info
end

Inputs:
- state
- action
- dt

Outputs:
- next_state
- obs
- info

Where:
- next_state is the hidden full simulator state
- obs is what the agent sees
- info is extra debug/reward information

13.2 What one RL step should do

Each step should do this:
1. Receive action from the agent
2. Parse and validate it
3. Clip it to safe limits
4. Apply it to feed/aeration/agitation/cooling/pH controls
5. Run the simulator from current time to current time + dt
6. Create the plant-console observation
7. Compute reward
8. Check if episode should end
9. Return observation, reward, done, info

Pseudo-Python:
def step(action):
    safe_action = validate_and_clip(action)

    next_state, raw_obs, sim_info = plant.step_batch(
        state=self.state,
        action=safe_action,
        dt=self.control_interval_h
    )

    obs = make_operator_observation(raw_obs, next_state)
    reward, reward_info = compute_reward(obs, safe_action, sim_info)
    done, done_reason = check_done(obs, sim_info)

    self.state = next_state

    return obs, reward, done, {
        **sim_info,
        **reward_info,
        "done_reason": done_reason,
    }

13.3 Control interval

Do not let the agent act at every tiny simulator integration point.

Choose a practical decision interval such as:
- 5 minutes
- 10 minutes
- 15 minutes

For MVP:
Use 10 simulated minutes per agent step.

Why:
- control effects are visible
- episode is not too long
- training stays manageable

Example:
Episode length = 50 steps
Step interval = 10 minutes
Total simulated time = 500 minutes = 8.33 hours

That is good for a fault-recovery task.


14. What the agent is given

The agent gets a plant-console observation.

It should be easy for an LLM to read and easy for code to parse.

Recommended observation:
{
  "time_h": 42.5,
  "batch_phase": "production",
  "measurements": {
    "temperature_C": 25.3,
    "pH": 6.47,
    "dissolved_oxygen_pct": 22.1,
    "substrate_g_L": 0.14,
    "biomass_g_L": 21.8,
    "penicillin_g_L": 1.52,
    "volume_L": 83500
  },
  "setpoints_or_limits": {
    "temperature_target_C": 25.0,
    "pH_target": 6.50,
    "DO_min_safe_pct": 20.0,
    "substrate_max_g_L": 0.30
  },
  "current_controls": {
    "feed_rate_L_h": 78,
    "aeration_rate_vvm": 0.85,
    "agitation_rpm": 120,
    "cooling_valve_pct": 45
  },
  "recent_trends": {
    "temperature": "stable",
    "pH": "slowly_decreasing",
    "dissolved_oxygen": "falling_fast",
    "substrate": "rising"
  },
  "alarm": "DO_near_low_limit",
  "previous_action": {
    "feed_delta_L_h": 5,
    "aeration_delta_vvm": 0,
    "agitation_delta_rpm": 0,
    "cooling_delta_pct": 0
  },
  "instruction": "Choose the next safe control action for the next 10 minutes."
}

Important design choice:
Include recent trends and previous action.

Why:
Real operators use trends, not just one reading.
It also helps the LLM understand what is happening.


15. What actions the agent can take

For the MVP, keep the action space small.

MVP action space: DO/feed recovery

Allowed actions:
{
  "feed_delta_L_h": -5 | 0 | 5,
  "aeration_delta_vvm": -0.10 | 0 | 0.10,
  "agitation_delta_rpm": -5 | 0 | 5,
  "reason": "optional short explanation"
}

Do NOT train the LLM to answer in free-form paragraphs.
It must output structured JSON.

Why:
- easier to validate
- easier to score
- easier to keep safe
- easier for OpenEnv

Later action space can expand to:
{
  "feed_delta_L_h": -5,
  "aeration_delta_vvm": 0.10,
  "agitation_delta_rpm": 5,
  "cooling_delta_pct": 0,
  "acid_base_action": "none",
  "reason": "..."
}

And later still:
{
  "feed_delta_L_h": -5,
  "aeration_delta_vvm": 0.10,
  "agitation_delta_rpm": 5,
  "cooling_delta_pct": 3,
  "base_pump_delta": 0,
  "acid_pump_delta": 0,
  "antifoam_action": "none",
  "reason": "..."
}


16. What reward teaches the agent

The reward is the teacher.

The agent does not automatically know fermentation.
It tries actions, the simulator shows the results, and the reward tells it whether it did well.

We should NOT use one single simple reward.
We should use a reward made of separate parts.

16.1 MVP reward parts

For the DO/feed recovery task:

total_reward =
  + DO safety reward
  + productivity reward
  + substrate control reward
  + stability reward
  - too much control movement penalty
  - invalid action penalty

Example:
reward = (
    0.35 * do_reward
    + 0.25 * productivity_reward
    + 0.15 * substrate_reward
    + 0.10 * stability_reward
    - 0.10 * control_effort_penalty
    - 0.05 * invalid_action_penalty
)

16.2 DO reward

Goal:
Keep dissolved oxygen above a safe threshold.

Example:
if DO >= 25%:
    do_reward = +1.0
elif 20% <= DO < 25%:
    do_reward = +0.3
elif 15% <= DO < 20%:
    do_reward = -0.5
else:
    do_reward = -1.0 and maybe early termination

16.3 Productivity reward

Goal:
Keep penicillin production healthy.

Example:
productivity_reward = normalized increase in penicillin concentration during the step

Important:
Do NOT let productivity be the only reward.
Otherwise the agent might break safety to chase more yield.

16.4 Substrate reward

Goal:
Avoid starvation and overdose.

Example:
ideal substrate range: 0.05 to 0.20 g/L

if substrate in range:
    substrate_reward = +1
elif slightly outside:
    substrate_reward = 0
else:
    substrate_reward = -1

16.5 Control effort penalty

Goal:
Avoid wild changes.

Example:
penalty = abs(feed_delta) + abs(aeration_delta) + abs(agitation_delta)

Why:
In a real plant, the agent cannot keep slamming controls hard every step.

16.6 Oscillation penalty

Goal:
Avoid switching direction too often.

Example:
if previous feed_delta was +5 and current feed_delta is -5:
    penalty += 0.1

This teaches smoother control.

16.7 Invalid action penalty

Goal:
Force the LLM to output valid JSON and valid values.

Example:
- invalid JSON -> -1.0
- unknown key -> -0.5
- value outside allowed range -> clip it and add penalty

This matters because the model must learn correct action format too.

16.8 Safety termination

End the episode early if:
- DO below critical threshold for too long
- pH outside hard safety range
- temperature outside hard safety range
- volume outside valid range
- substrate overdose beyond hard limit
- repeated invalid actions

Give a large penalty in these cases.


17. Why reward design is central

This environment is not just "simulator wrapped in code".

A major part of the project is:
a reward system for safe autonomous operation.

The reward should measure:
- tracking error
- recovery time
- control effort
- safety limits
- disturbance recovery
- yield preservation
- format correctness

So in the README we can say:
The environment is not just a simulator wrapper.
Its main contribution is a clear reward system that teaches the agent safety, stability, recovery, and productivity instead of only final yield.


18. First MVP task

Do NOT start with full autonomous penicillin production.

Start with one narrow task.

Recommended MVP:
DO/feed disturbance recovery in penicillin fermentation

Why this is the best first task:
- easy to understand
- easy to demo
- still feels like operator work
- uses feed/aeration/agitation actions
- avoids copying final-yield-only papers
- makes before/after improvement easy to show

Scenario:
At some point in the production phase:
- feed rate was increased
- substrate starts to rise
- oxygen demand goes up
- DO starts falling
- agitation is already near upper limit
- aeration still has room

The agent must recover safely.

Good behavior:
- reduce feed slightly
- increase aeration
- avoid pushing agitation beyond limit
- avoid overcorrecting
- restore DO above safe threshold
- preserve productivity

Bad behavior:
- increase feed more
- ignore the DO drop
- slam agitation to max repeatedly
- make invalid actions
- overcorrect until the process starves

Episode setup:
- start at selected batch hour, e.g. 40-60 h
- inject a disturbance
- run for 50-100 steps
- each step = 10 simulated minutes
- end when recovered, timed out, or failed


19. Curriculum plan

We should not start with the hardest task.

Curriculum:

Level 1: Valid action formatting
Goal:
Teach the model to output valid JSON

Reward:
- valid JSON
- correct keys
- values inside allowed range

Level 2: Single-variable DO recovery
Observation:
DO, feed, aeration, agitation, trend

Action:
feed_delta, aeration_delta, agitation_delta

Reward:
keep DO above safe threshold

Level 3: DO recovery with productivity
Add:
substrate, biomass, penicillin

Reward:
DO safety + productivity + substrate control

Level 4: pH drift
Add:
pH and acid/base action

Reward:
pH near target + avoid overdose

Level 5: temperature disturbance
Add:
temperature and cooling action

Reward:
temperature tracking + low control effort

Level 6: multi-variable operator task
Agent controls:
feed, aeration, agitation, cooling, acid/base

Reward:
safety + stability + productivity + recovery + low control abuse

Level 7: fault recovery
Scenarios:
- DO sensor drift
- pH sensor delay
- cooling becomes weak
- oxygen transfer limit
- substrate overfeed
- foaming event
- control limit reached


20. What "physics-aware" means in this project

BioOperatorEnv is physics-aware because:
the next plant state comes from the simulator or digital twin, not from the LLM.

The LLM does not predict the bioreactor.
The LLM only chooses actions.
The simulator decides what physically and biologically happens.

Correct explanation:
The agent is not asked to guess bioreactor physics.
It receives observations, chooses actions, and the simulator calculates what happens next.
That makes the environment physics-backed, while still training an LLM/RL operator policy.

Do NOT say:
"The LLM is physics-aware."

Better say:
"The environment is physics-aware; the agent learns from physics-based consequences."


21. How this can later scale to other bioprocesses

Be honest:
One trained agent will NOT automatically control every bioprocess.

A penicillin batch, yeast fermentation, mammalian cell culture, algae photobioreactor, and enzyme fermentation are all different.

They differ in:
- organism
- reactor scale
- sensors
- product
- safe ranges
- control knobs
- time scale
- growth behavior
- product formation behavior
- business objective

So the correct claim is:
BioOperatorEnv is a reusable framework, not a universal trained controller.
Each process needs its own simulator, digital twin, or data-driven plant model.
Once that plant engine exists, BioOperatorEnv provides the common training interface, reward system, safety checks, and evaluation pipeline.

What stays common across bioprocesses:
Many operator questions are always similar:
- What can I measure?
- What can I control?
- What is the safe range?
- What is the target?
- What disturbance is happening?
- What action should I take now?
- Did my previous action help or hurt?

Common observations:
- pH
- temperature
- dissolved oxygen
- biomass/cell density
- substrate
- product
- volume
- feed rate
- gas flow
- agitation
- alarms

Common actions:
- increase/decrease feed
- increase/decrease aeration
- increase/decrease agitation
- adjust cooling
- dose acid/base
- change setpoint
- hold action

Common rewards:
- stay safe
- track target values
- recover from disturbances
- avoid overcorrecting
- avoid wasting energy or material
- maximize product or biomass

What changes per process:
- biology model
- growth behavior
- product formation behavior
- safe ranges
- available sensors
- allowed controls
- reactor geometry
- batch duration
- business objective

So the system uses a process adapter.


22. Process adapter design

Common interface:
class PlantAdapter:
    def reset(self, seed: int, scenario: str):
        """Start a fresh episode and return initial hidden state."""

    def step(self, state, action, dt):
        """Apply action and advance the process by dt."""

    def observe(self, state):
        """Convert hidden simulator state into what the agent can see."""

    def safety_limits(self):
        """Return safety constraints for this process."""

    def default_targets(self):
        """Return default targets for this process."""

For the MVP:
PlantAdapter = IndPenSimAdapter

Later:
- YeastFermentationAdapter
- PhotobioreactorAdapter
- MammalianCellCultureAdapter

This is the value of the framework:
New processes do not need a whole new training system.
They need a new plant adapter and maybe a different reward setup.


23. Project architecture

BioOperatorEnv
|
|-- OpenEnv API
|   |-- reset()
|   |-- step(action)
|   |-- state()
|
|-- agent interface
|   |-- plant-console observation
|   |-- JSON action schema
|   |-- action validation
|
|-- reward system
|   |-- safety
|   |-- stability
|   |-- productivity
|   |-- low control abuse
|   |-- format correctness
|
|-- scenario generator
|   |-- DO drop
|   |-- substrate overfeed
|   |-- pH drift
|   |-- temperature disturbance
|   |-- control limit reached
|
|-- plant adapters
|   |-- IndPenSim adapter for MVP
|
|-- baselines
|   |-- random agent
|   |-- fixed recipe
|   |-- rule-based operator
|   |-- untrained LLM
|
|-- training
|   |-- TRL / GRPO script
|   |-- Unsloth loading
|   |-- reward logging
|
|-- demo
|   |-- before/after trajectories
|   |-- reward curves
|   |-- sample actions
|   |-- README / mini-blog / video


24. OpenEnv implementation expectations

We should implement:
- Action model
- Observation model
- State model
- Environment class
- reset()
- step()
- state()
- reward computation
- FastAPI/OpenEnv wrapper
- openenv.yaml

We should also keep the engineering clean:
- standard environment API
- client/server separation
- no mixing server internals into client
- valid manifest
- clear logs


25. Training plan

25.1 What are we training?

We are NOT training a model from scratch.

We start from an existing open-source instruct LLM.

Then we train small extra weights using LoRA/QLoRA through TRL and Unsloth.

Simple explanation:
We take an existing model,
show it plant observations,
let it output actions,
score those actions,
and make it more likely to produce better actions next time.

25.2 Mental model

Base instruct LLM
-> prompt has current plant observation
-> LLM outputs JSON action
-> environment parses action
-> simulator advances process
-> reward is calculated
-> TRL/GRPO updates the small trainable weights
-> model becomes more likely to output high-reward actions

25.3 Why LoRA/QLoRA

Simple reason:
Training the whole model is too expensive.

LoRA/QLoRA lets us:
- keep the base model mostly frozen
- train only a small part
- save memory
- make the hackathon feasible

25.4 Training stages

Stage 1: formatting warm-up
Goal:
Teach the LLM to output valid JSON actions

Method:
- prompt examples
- maybe a small supervised dataset of valid action examples
- reward valid formatting strongly

Stage 2: easy RL
Goal:
Learn simple DO recovery

Environment:
- short episode
- one disturbance
- small action space

Reward:
- DO safety
- valid action
- low control abuse

Stage 3: harder RL
Goal:
Balance DO recovery and productivity

Environment:
- add substrate, product, biomass
- add overfeeding and underfeeding effects

Reward:
- DO safety
- productivity
- substrate control
- stability

Stage 4: multi-variable operator
Goal:
Run feed/aeration/agitation/cooling/pH together

Environment:
- multiple disturbances
- longer horizon

Reward:
- safety
- productivity
- recovery time
- low control abuse


26. Baselines

Judges need to see improvement.

Recommended baselines:

26.1 Random agent
Randomly picks valid actions.

Expected:
- unsafe often
- bad recovery
- low reward

26.2 Fixed recipe
Keeps original recipe / default simulator behavior.

Expected:
- okay in normal case
- weaker under disturbance
- not very adaptive

26.3 Rule-based operator
Simple if-then logic.

Example:
if DO < 20:
    increase_aeration()
    decrease_feed()
elif substrate > max_limit:
    decrease_feed()
else:
    hold()

Expected:
- decent baseline
- simple and understandable
- may overreact or be wasteful

26.4 Untrained LLM
Base instruct model with the same prompt.

Expected:
- may sound smart
- may choose inconsistent actions
- may output bad JSON or wrong values

26.5 Trained LLM
Same model after RL.

Expected:
- more valid actions
- fewer safety failures
- faster DO recovery
- better total reward
- better productivity preservation
- less control abuse


27. Metrics to report

Do not only report total reward.

Report:
- average episode reward
- success rate
- safety failure rate
- invalid action rate
- average DO recovery time
- lowest DO reached
- time spent below safe DO
- average penicillin productivity during episode
- substrate overdose frequency
- total control effort
- oscillation count
- timeout rate

README plots should include:
1. average reward vs training steps
2. safety failures vs training steps
3. success rate vs training steps
4. baseline vs trained trajectory for a DO crash
5. reward part breakdown


28. Demo story

The demo should be understandable in under two minutes.

Demo narrative:
We are training an AI operator for a penicillin bioreactor.
At hour 48, the batch experiences oxygen stress after a feed disturbance.
The untrained model reacts badly and lets dissolved oxygen fall below the safe threshold.
After RL training inside BioOperatorEnv, the model learns to reduce feed slightly, increase aeration, avoid overusing agitation, and recover DO faster while preserving productivity.

Visual demo:
Show three trajectories on one plot:
- random agent
- fixed recipe or rule-based baseline
- trained LLM

Plot:
- DO over time
- feed over time
- aeration over time
- reward over time

Before training action example:
{
  "feed_delta_L_h": 5,
  "aeration_delta_vvm": 0,
  "agitation_delta_rpm": 0,
  "reason": "Increase feed to improve production."
}

After training action example:
{
  "feed_delta_L_h": -5,
  "aeration_delta_vvm": 0.10,
  "agitation_delta_rpm": 0,
  "reason": "DO is falling after feed increase. Reduce oxygen demand and increase aeration first."
}

This demo answers:
- what is the problem?
- what does the agent see?
- what does the agent do?
- what reward teaches it?
- what changed after training?
- why does it matter?


29. Reward cheating prevention

We must stop the model from gaming the reward.

Plan:
- JSON schema validation
- clip actions to allowed limits
- penalty for invalid or out-of-range actions
- hidden simulator state not available to agent
- agent only sees observation, not full reward internals
- separate reward parts for safety, productivity, effort, etc.
- episode timeouts
- early stopping for unsafe states
- log all actions and reward parts
- inspect sample generations during training

The agent must never be able to:
- directly edit simulator state
- directly edit reward values
- reset time
- access hidden values that were not given in the observation
- skip steps
- write to global state


30. MATLAB vs Python implementation

This is important.

The simulator is currently in MATLAB.

For local development:
We can refactor the MATLAB code into reset_batch and step_batch.

For final OpenEnv / Hugging Face deployment:
MATLAB may be hard to host.

So we need a practical plan.

Option A: MATLAB Engine for Python

Architecture:
OpenEnv Python server
-> MATLAB Engine
-> MATLAB simulator step_batch()

Pros:
- fastest local prototype
- uses existing MATLAB code
- high fidelity

Cons:
- hard to deploy on Hugging Face Spaces
- requires MATLAB license
- extra overhead
- harder for judges to run

Recommended use:
Local proof of concept only

Option B: Port the MVP plant logic to Python

Architecture:
OpenEnv Python server
-> Python plant adapter
-> SciPy ODE integration / simplified IndPenSim-like model

Pros:
- deployable
- cleaner
- judges can run it
- fits OpenEnv / TRL better
- easier to train with

Cons:
- takes work
- may not include all 33 ODEs at first
- must be honest about what level of fidelity we have

Recommended use:
Best final hackathon submission if possible

Option C: MATLAB-generated surrogate

Architecture:
MATLAB generates trajectories
-> train a next-state model
-> OpenEnv uses that surrogate plant

Pros:
- fast
- deployable
- uses MATLAB-generated data

Cons:
- less physically faithful
- may fail outside its training data
- harder to defend

Recommended use:
Fallback if direct Python port is too hard

Final practical recommendation:
1. Local: refactor MATLAB simulator into reset_batch / step_batch
2. Submission: build a Python OpenEnv wrapper
3. If MATLAB deployment is impossible, use a Python plant adapter or a simplified calibrated model


31. Exact coding target for Codex

Goal:
Build an OpenEnv-compatible environment called BioOperatorEnv.

It wraps a step-wise penicillin fermentation simulator and gives an LLM/RL agent a plant-console decision task.

The first task is DO/feed disturbance recovery.

Required environment methods:
reset(seed: int | None = None, scenario: str = "do_drop") -> Observation

step(action: Action) -> tuple[Observation, float, bool, dict]

state() -> dict

Required action model:
class BioOperatorAction(BaseModel):
    feed_delta_L_h: Literal[-5, 0, 5]
    aeration_delta_vvm: Literal[-0.10, 0.0, 0.10]
    agitation_delta_rpm: Literal[-5, 0, 5]
    reason: Optional[str] = None

Required observation model:
class BioOperatorObservation(BaseModel):
    time_h: float
    batch_phase: str
    temperature_C: float
    pH: float
    dissolved_oxygen_pct: float
    substrate_g_L: float
    biomass_g_L: float
    penicillin_g_L: float
    volume_L: float
    feed_rate_L_h: float
    aeration_rate_vvm: float
    agitation_rpm: float
    recent_trends: dict
    alarm: Optional[str]
    previous_action: Optional[dict]

Required reward info output:
info = {
    "reward_total": reward,
    "reward_components": {
        "do_safety": do_reward,
        "productivity": productivity_reward,
        "substrate_control": substrate_reward,
        "control_effort": control_penalty,
        "stability": stability_reward,
        "format_validity": format_reward
    },
    "safety_violation": bool,
    "success": bool,
    "done_reason": str
}

Required scenarios:
- normal
- do_drop
- substrate_overfeed
- aeration_limit

Required baselines:
- random_agent.py
- fixed_recipe_agent.py
- rule_based_agent.py
- untrained_llm_eval.py
- trained_llm_eval.py

Required plots:
- reward_curve.png
- success_rate_curve.png
- safety_violations_curve.png
- do_recovery_comparison.png
- action_trajectory_comparison.png


32. README structure

Title:
BioOperatorEnv: A Flight Simulator for Autonomous Bioreactor Agents

Subtitle:
An OpenEnv environment for training LLM/RL agents to safely operate simulated industrial bioreactors.

Problem:
Explain:
- real bioprocesses are nonlinear, partially visible, expensive, and safety-sensitive
- autonomous agents cannot learn on real batches
- they need simulator-backed practice first

Prior work:
Say:
Prior work has shown RL can improve specific bioprocess controllers, including pH control in photobioreactors, temperature control in ethanol fermentation, yeast bioreactor control, and penicillin fed-batch control.
BioOperatorEnv does not claim to be the first RL bioreactor controller.
Instead, it turns this domain into a reusable OpenEnv-style LLM-agent training environment.

Environment:
Explain:
- the agent sees plant-console observations
- the agent outputs JSON control actions
- the simulator advances the process
- the reward scores safety, recovery, productivity, and low control abuse

MVP task:
DO/feed disturbance recovery in penicillin fermentation

Agent observation:
Include sample JSON

Agent action:
Include sample JSON

Reward:
Show reward parts

Training:
Explain:
Base instruct LLM + TRL/GRPO + Unsloth + OpenEnv

Results:
Show:
- reward curves
- baseline comparison
- before/after action examples

Limitations:
Be honest:
- this is not a universal controller
- each new process needs a plant adapter
- the MVP starts with penicillin / IndPenSim-style simulation
- deployment to real reactors would need much more validation, limits, and human supervision

Future work:
- full IndPenSim fidelity
- more fault scenarios
- pH/temperature/DO multi-variable tasks
- yeast / PBR / cell-culture adapters
- imitation traces from operators or simple controllers
- offline + online RL
- human-in-the-loop safety evaluation


33. Final pitch

BioOperatorEnv is a flight simulator for autonomous bioreactor agents.
Prior work has shown that reinforcement learning can improve individual bioprocess controllers, such as pH, temperature, yeast fermentation, and penicillin fed-batch control.
But before LLM/RL agents can be trusted in real biomanufacturing plants, they need a safe place to practice.
BioOperatorEnv turns a validated industrial fermentation simulator into an OpenEnv-compatible training playground where an agent reads plant-console measurements, chooses structured control actions, and receives rewards for safe, stable, productive operation under disturbances and control limits.
We start with penicillin fermentation and a DO/feed recovery task, then expand toward multi-variable autonomous bioprocess operation.


34. Core defense in one paragraph

We are not claiming to invent RL for bioreactor control.
The literature already shows RL can work for specific bioprocess controllers.
Our contribution is to package the problem as an LLM-agent training environment:
standardized observations,
structured actions,
fault scenarios,
safety/productivity reward parts,
baselines,
OpenEnv deployment,
and before/after RL training evidence.
Existing work built controllers.
BioOperatorEnv builds the playground where autonomous bioreactor operators learn to act safely.


35. Build order

Build in this order:
1. define the MVP scenario: DO/feed recovery
2. refactor MATLAB simulator into reset_batch and step_batch
3. create Python/OpenEnv action and observation schemas
4. implement a local environment loop before using an LLM
5. implement reward parts and log them separately
6. run random and rule-based baselines
7. make an untrained LLM output JSON actions
8. run small TRL/Unsloth training
9. plot before/after behavior
10. deploy environment to Hugging Face Space
11. write README and short demo video/blog


36. Final decision record

Decision 1:
We will not do generic "RL env for Bioreactor CFD"

Reason:
- too broad
- CFD is heavy
- hard to train
- hard to demo
- reward may be too slow and sparse

Decision 2:
We will build an autonomous bioreactor operator training environment

Reason:
- step-by-step actions
- automatic rewards
- strong real-world story
- fits OpenEnv / LLM-agent hackathon
- easier to demo

Decision 3:
We will start with penicillin fermentation

Reason:
- existing MATLAB simulator available
- industrial-scale context
- rich process behavior
- good starting point

Decision 4:
We will not claim novelty in "RL for IndPenSim"

Reason:
- that already exists in prior work

Decision 5:
Our novelty is the OpenEnv LLM-agent training environment

Reason:
- hackathon is about environments for LLM training
- existing papers are mostly controller-specific
- our work adds standardized observations, JSON actions, reward parts, fault scenarios, baselines, and train/eval loops

Decision 6:
The MVP task is DO/feed disturbance recovery

Reason:
- clear
- safe
- demoable
- avoids copying final-yield optimization
- tests operator-like behavior

Decision 7:
Reward will be made of separate parts

Reason:
- avoids reward cheating
- matches hackathon expectations
- gives clear metrics and plots

Decision 8:
Simulator must become step-wise

Reason:
- current MATLAB code runs full batches
- RL needs reset/step interaction

Decision 9:
The environment is physics-aware through the plant engine

Reason:
- LLM chooses actions
- simulator computes consequences

Decision 10:
Scaling to other processes will happen through plant adapters

Reason:
- one trained model cannot control all reactors
- but the framework can be reused if each process provides a simulator or digital twin


37. Final version to give Codex

We are building BioOperatorEnv, an OpenEnv-compatible training playground for autonomous bioreactor operator agents.

The goal is not to create a new simulator from scratch or claim first RL control of bioreactors.
Prior work already shows RL can control specific bioprocess tasks.
Our contribution is the environment layer:
an LLM/RL agent receives plant-console observations from a simulator-backed bioreactor,
outputs structured JSON control actions,
and receives rewards for safety, stability, recovery, productivity, and low control abuse.

For the MVP, use the existing MATLAB penicillin simulator as the plant engine.
Refactor it from a full-batch runner into a step-wise simulator with reset_batch(seed, scenario) and step_batch(state, action, dt).
The first task is DO/feed disturbance recovery:
the agent sees DO falling after a feed disturbance and must choose feed/aeration/agitation actions to recover safely without sacrificing productivity or abusing controls.

Implement OpenEnv methods reset(), step(action), and state().
Define action and observation schemas.
Compute reward as separate parts:
DO safety,
productivity,
substrate control,
stability,
control effort,
and format validity.
Include random, fixed-recipe, rule-based, untrained LLM, and trained LLM baselines.
Generate reward curves and before/after trajectory plots.
Deploy as a Hugging Face Space and include a README explaining the problem, prior work, environment design, rewards, training, results, and limitations.
```
