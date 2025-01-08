import os
import torch
import wandb
import gymnasium as gym
import safety_gymnasium
from omegaconf import DictConfig, OmegaConf

# Defaults
# from stable_baselines3.common.callbacks import EvalCallback
# from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.vec_env import VecFrameStack

# Custom
from intervention_rl.trainers.utils.my_ppo import PPO_HIRL
from intervention_rl.utils.safe_vec_utils import make_safe_env, make_safe_eval_env
from intervention_rl.eval.callback_eval_mlp import MLPEvalCallback
from intervention_rl.eval.callback_eval_safety import SafetyEvalCallback
from intervention_rl.utils.callback_blocker import BlockerTrainingCallback
from intervention_rl.utils.callback_checkpoints import CustomCheckpointCallback

class PPOTrainer:
    def __init__(self, cfg: DictConfig, exp_dir: str):
        self.cfg = cfg
        self.exp_dir = exp_dir
        self.device = torch.device(cfg.device)

        # Create necessary directories for saving models
        self.agent_save_path = os.path.join(exp_dir, "agent")
        self.blocker_save_path = os.path.join(exp_dir, "blocker")
        os.makedirs(self.agent_save_path, exist_ok=True)
        os.makedirs(self.blocker_save_path, exist_ok=True)

        sanitized_config = OmegaConf.to_container(self.cfg, resolve=True)

        if cfg.wandb.use:
            wandb.init(
                project=cfg.wandb.project,
                name=cfg.hp_name,
                config=sanitized_config,
                sync_tensorboard=True
            )

        # Create the environment
        self.env = make_safe_env(cfg.env.name, n_stack=cfg.env.n_stack, seed=cfg.seed)
        self.eval_env = make_safe_eval_env(cfg.env.name, n_stack=cfg.env.n_stack, seed=cfg.seed + 100)

        # Initialize the model (PPO_HIRL)
        self.model = PPO_HIRL(
            policy=cfg.algo.ppo.policy,
            env=self.env,
            learning_rate=cfg.algo.ppo.learning_rate,
            n_steps=cfg.algo.ppo.n_steps,
            batch_size=cfg.algo.ppo.batch_size,
            n_epochs=cfg.algo.ppo.n_epochs,
            gamma=cfg.algo.ppo.gamma,
            gae_lambda=cfg.algo.ppo.gae_lambda,
            clip_range=cfg.algo.ppo.clip_range,
            clip_range_vf=None,
            normalize_advantage=cfg.algo.ppo.normalize_advantage,
            ent_coef=cfg.algo.ppo.ent_coef,
            vf_coef=cfg.algo.ppo.vf_coef,
            max_grad_norm=cfg.algo.ppo.max_grad_norm,
            use_sde=cfg.algo.ppo.use_sde,
            sde_sample_freq=cfg.algo.ppo.sde_sample_freq,
            verbose=cfg.algo.ppo.verbose,
            seed=cfg.seed,
            device=cfg.device,
            tensorboard_log=os.path.join(exp_dir, "tensorboard"),

            env_name=cfg.env.name,
            exp_type=cfg.exp_type,
            pretrained_blocker=cfg.pretrained_blocker,
            blocker_switch_time=cfg.env.blocker_switch_time,
            new_action=cfg.env.new_action,

            bonus_type=cfg.env.bonus_type,
            alpha=cfg.env.alpha,
            alpha_increase=cfg.env.alpha_increase,
            max_alpha=cfg.env.max_alpha,
            beta=cfg.env.beta,
            beta_increase=cfg.env.beta_increase,
            max_beta=cfg.env.max_beta,
            iota=cfg.env.iota,
            penalty_type=cfg.env.penalty_type,
            penalty=cfg.env.penalty,

            blocker_clearance=cfg.env.blocker_clearance,
            catastrophe_clearance=cfg.env.catastrophe_clearance,
        )

    def train(self):
        # Custom evaluation callback
        eval_callback = SafetyEvalCallback(
            cfg = self.cfg,
            eval_env=self.eval_env,
            eval_freq=self.cfg.env.eval_freq,
            eval_seed =self.cfg.eval.eval_seed,
            gif_freq=self.cfg.env.gif_freq,
            n_eval_episodes=self.cfg.eval.eval_episodes,
            new_action=self.cfg.env.new_action,
            verbose=self.cfg.eval.verbose,
        )

        callback_list = [eval_callback]

        # Blocker training callback with saving functionality
        if self.cfg.exp_type in ["ours", "hirl"]:
            blocker_callback = BlockerTrainingCallback(
                train_freq=self.cfg.env.blocker_train_freq,   # Frequency of blocker training
                epochs=self.cfg.env.blocker_epochs,           # Number of epochs to train the blocker
                save_freq=self.cfg.env.blocker_save_freq,     # Frequency of saving blocker model weights
                save_path=self.blocker_save_path,                  # Directory to save blocker model weights
                save_blocker_dataset=self.cfg.train.save_blocker_dataset,
                name_prefix="blocker_model",                       # Prefix for saved blocker model files
                verbose=self.cfg.algo.ppo.verbose
            )

            callback_list.append(blocker_callback)

        # Checkpoint callback to save agent weights
        checkpoint_callback = CustomCheckpointCallback(
            save_freq=self.cfg.env.save_freq,             # Save frequency from config
            save_path=self.agent_save_path,                    # Directory to save the models
            name_prefix="ppo_model",                      # Prefix for the saved model files
        )
        callback_list.append(checkpoint_callback)

        # Combine callbacks into a CallbackList
        callback_list = CallbackList(callback_list)

        # Start training
        self.model.learn(
            total_timesteps=self.cfg.env.total_timesteps,
            callback=callback_list,
            log_interval=self.cfg.env.log_freq,
        )