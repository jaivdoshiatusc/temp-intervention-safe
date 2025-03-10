import numpy as np
import wandb
import imageio
import cv2
import io  
from stable_baselines3.common.callbacks import BaseCallback

class SafetyEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, eval_seed, gif_freq, n_eval_episodes, verbose=1):
        super(SafetyEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_seed = eval_seed
        self.gif_freq = gif_freq
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose

        self.cum_catastrophe = 0
        self.cum_env_intervention = 0
        self.cum_exp_intervention = 0
        self.cum_disagreement = 0

        self.blocker_cum_catastrophe = 0
        self.blocker_cum_env_intervention = 0
        self.blocker_cum_exp_intervention = 0
        self.blocker_cum_disagreement = 0

        # Initialize the next evaluation step
        self.next_eval_step = self.eval_freq
        self.next_gif_step = self.gif_freq

    def _on_step(self) -> bool:
        # Check if it's time to evaluate
        if self.num_timesteps >= self.next_eval_step:
            self.evaluate()
            # Schedule the next evaluation
            self.next_eval_step += self.eval_freq
        return True
    
    def create_frame(self):
        frame = self.eval_env.render()
        return frame
    
    def evaluate(self):
        all_episode_rewards = []
        all_episode_lengths = []
        all_catastrophes = []  
        all_env_interventions = []
        all_exp_interventions = []
        all_disagreements = []

        # Check if we should create a GIF
        create_gif = self.num_timesteps >= self.next_gif_step and wandb.run is not None
        frames = [] if create_gif else None

        # Access experiment parameters
        exp_type = self.model.exp_type
        blocker_switch_time = self.model.blocker_switch_time
        num_timesteps = self.num_timesteps

        for episode in range(self.n_eval_episodes):
            episode_reward = 0.0
            episode_length = 0
            episode_catastrophes = 0
            episode_env_interventions = 0
            episode_exp_interventions = 0
            episode_disagreements = 0
            self._last_cost = 0.0

            episode_seed = self.eval_seed + episode
            self.eval_env.set_seed(episode_seed)
            obs, info = self.eval_env.reset()
            terminated = truncated = False
            
            while not terminated and not truncated:
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)
                action_item = action

                if exp_type in ["ours", "hirl"]:
                    blocker_heuristic_decision = self.model.blocker_heuristic.should_block(obs, self._last_cost)
                    blocker_model_decision, _, _ = self.model.blocker_model.should_block(
                        obs,
                        action_item,
                        blocker_heuristic_decision
                    )
                    if blocker_heuristic_decision != blocker_model_decision:
                        episode_disagreements += 1
                    if num_timesteps <= blocker_switch_time:
                        if blocker_heuristic_decision != [2,2]:
                            episode_env_interventions += 1
                            episode_exp_interventions += 1   
                            action_item = blocker_heuristic_decision                   
                    else:
                        if blocker_model_decision != [2,2]:
                            episode_env_interventions += 1 
                            action_item = blocker_model_decision
                        if blocker_heuristic_decision != [2,2]:
                            episode_exp_interventions += 1 
                        
                elif exp_type in ["expert"]:
                    blocker_heuristic_decision = self.model.blocker_heuristic.should_block(obs, self._last_cost)
                    if blocker_heuristic_decision != [2,2]:
                        action_item = blocker_heuristic_decision
                        episode_env_interventions += 1
                        episode_exp_interventions += 1
                
                new_obs, reward, cost, terminated, truncated, infos = self.eval_env.step(action_item)
                self._last_cost = cost
                
                # Capture frames for GIF
                if create_gif:
                    frame = self.create_frame()

                if self.model.blocker_heuristic.is_catastrophe(self._last_cost):
                    episode_catastrophes += 1  # Increment catastrophe count
                    
                if create_gif:        
                    frames.append(frame)

                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                if isinstance(cost, np.ndarray):
                    cost = cost[0]
                episode_reward += reward
                episode_length += 1
                obs = new_obs

            # Log episode statistics
            all_episode_rewards.append(episode_reward)
            all_episode_lengths.append(episode_length)
            all_catastrophes.append(episode_catastrophes)
            all_env_interventions.append(episode_env_interventions)
            all_exp_interventions.append(episode_exp_interventions)
            all_disagreements.append(episode_disagreements)

        # Compute cumulative metrics by summing over all episodes
        total_reward = sum(all_episode_rewards)
        total_ep_length = sum(all_episode_lengths)

        # Update cumulative counts
        self.cum_catastrophe += sum(all_catastrophes)
        self.cum_env_intervention += sum(all_env_interventions)
        self.cum_exp_intervention += sum(all_exp_interventions)
        self.cum_disagreement += sum(all_disagreements)

        if num_timesteps > blocker_switch_time:
            self.blocker_cum_catastrophe += sum(all_catastrophes)
            self.blocker_cum_env_intervention += sum(all_env_interventions)
            self.blocker_cum_exp_intervention += sum(all_exp_interventions)
            self.blocker_cum_disagreement += sum(all_disagreements)

        # Compute mean values
        mean_reward = np.mean(all_episode_rewards)
        std_reward = np.std(all_episode_rewards)
        mean_ep_length = np.mean(all_episode_lengths)
        mean_catastrophes = np.mean(all_catastrophes)
        mean_env_intervention = np.mean(all_env_interventions)
        mean_exp_intervention = np.mean(all_exp_interventions)
        mean_disagreement = np.mean(all_disagreements)

        # Record evaluation metrics (mean, cumulative)
        self.logger.record('eval/ep_rew_mean', mean_reward)
        self.logger.record('eval/ep_rew_std', std_reward)
        self.logger.record('eval/ep_len_mean', mean_ep_length)
        self.logger.record('eval/ep_catastrophe_mean', mean_catastrophes)
        self.logger.record('eval/ep_env_intervention_mean', mean_env_intervention)
        self.logger.record('eval/ep_exp_intervention_mean', mean_exp_intervention)
        self.logger.record('eval/ep_disagreement_mean', mean_disagreement)

        self.logger.record('eval/blocker_cum_catastrophe', self.blocker_cum_catastrophe)
        self.logger.record('eval/blocker_cum_env_intervention', self.blocker_cum_env_intervention)
        self.logger.record('eval/blocker_cum_exp_intervention', self.blocker_cum_exp_intervention)
        self.logger.record('eval/blocker_cum_disagreement', self.blocker_cum_disagreement)
        
        self.logger.record('eval/total_ep_length', total_ep_length)
        self.logger.record('eval/total_reward', total_reward)

        self.logger.record('eval/cum_catastrophe', self.cum_catastrophe)
        self.logger.record('eval/cum_env_intervention', self.cum_env_intervention)
        self.logger.record('eval/cum_exp_intervention', self.cum_exp_intervention)
        self.logger.record('eval/cum_disagreement', self.cum_disagreement)

        if create_gif:
            # Create GIF in-memory and upload to WandB without saving locally
            gif_buffer = io.BytesIO()  # In-memory buffer
            imageio.mimsave(gif_buffer, frames, format='GIF', fps=30)  # Save the GIF to buffer
            gif_buffer.seek(0)  # Reset buffer position to the beginning

            # Log the GIF to WandB
            wandb.log({f"eval_gif/gif": wandb.Video(gif_buffer, fps=30, format="gif")})

            gif_buffer.close()

            self.next_gif_step += self.gif_freq

        if self.verbose > 0:
            print(f"Eval num_timesteps={self.num_timesteps}, "
                  f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}, "
                  f"total_catastrophes={self.cum_catastrophe}, "
                  f"mean_catastrophes={mean_catastrophes:.2f}")

        # Dump the logs
        self.logger.dump(self.num_timesteps)
