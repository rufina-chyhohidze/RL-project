# Reinforcement Learning Project – MountainCarContinuous

This project explores **reward engineering** and **hyperparameter tuning** for Deep Reinforcement Learning (DRL) using **Stable Baselines 3**. The experiments are conducted on the `MountainCarContinuous-v0` environment.

The project includes:
- A baseline PPO implementation using the default environment reward  
- A custom reward function to improve learning behavior  
- An extension using hyperparameter tuning to improve training stability and performance  
- Logging and evaluation using TensorBoard  

---

## 1. Requirements

- Python **3.10 or higher**
- `pip`
- A UNIX-based system (macOS / Linux recommended)

---

## 2. Setup Instructions

### 2.1 Create and activate a virtual environment

From the project root directory:

```bash
python3 -m venv venv
source venv/bin/activate
```
After activation, your terminal prompt should start with (venv).

Install dependencies
``` bash 
pip install --upgrade pip
pip install -r requirements.txt
```
# 3. Project structure:
```
rl_project/
├── config/
│   ├── config_baseline.yaml
│   ├── config_custom.yaml
│   └── config_extension.yaml
├── src/
|   ├── test.ipynb
│   ├── train_baseline.py
│   ├── train_custom_reward.py
│   ├── train_extension.py
│   ├── custom_env_wrapper.py
│   ├── evaluate.py
│   └── visual.py
|   ├── utils.py
├── logs/        # TensorBoard logs (generated)
├── results/     # Trained models (generated)
└── README.md
```

## 4. Running the Experiments

### 4.1 Train the baseline model (default reward)

This trains PPO using the default environment reward.

```bash
python src/train_baseline.py

models are saved in: results/baseline 
```
### 4.2 Train with custom reward function
This trains PPO using the custom reward wrapper defined in custom_env_wrapper.py.
- python src/train_custom_reward.py
Models are saved in:
- results/custom_reward/
4.3 Run the extension (hyperparameter tuning)
The extension experiments tune PPO hyperparameters (e.g., clip_range, n_steps, or learning_rate) using the custom reward.
python src/train_extension.py
Models are saved in:
- results/extension/
Only one run per hyperparameter configuration is used, as allowed for hyperparameter tuning in the assignment.
### 5. Evaluating Trained Models
Evaluation is performed without exploration (deterministic=True) using the original environment reward.
Start Python:
- python
Then run, for example:
from evaluate import evaluate
### use test.ipynb to evaluate 

# Baseline PPO
evaluate("results/baseline/PPO_trial0", "MountainCarContinuous-v0", episodes=10)

# Custom reward PPO
evaluate("results/custom_reward/PPO_trial2", "MountainCarContinuous-v0", episodes=10)

# Extension (example)
evaluate("results/extension/PPO_clip_range0", "MountainCarContinuous-v0", episodes=10)
The script prints:
- Episodic returns
- Average return over the evaluation episodes
### 6. Visualizing Training (TensorBoard)
To inspect training curves:
tensorboard --logdir logs/
Open the printed URL (usually http://localhost:6006) in a browser.
Key metrics:
- rollout/ep_rew_mean
- rollout/ep_len_mean
### 7. Visualizing Agent Behavior
To visually inspect the trained agent in the environment:
python src/visual.py
This opens a window showing the MountainCar agent acting in the environment.
### 8. Notes
Training can take several minutes depending on hardware.
Results may vary slightly due to stochasticity in training.
For reproducibility, multiple trials are used in baseline and custom reward experiments.

Authors: Chyhohidze Rufina, Lukyanov Michael