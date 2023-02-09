import torch
import torch.optim as optim
from pathlib import Path
import yaml
import gym
from agents import PPOContinuousAgent
from common.utils import make_continuous_env, make_buffers
import numpy as np
import random
from tqdm import tqdm
import time


def main():
    # TRY NOT TO MODIFY: seeding
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    with open(str(Path().cwd() / 'config/config_ppo_continuous.yaml'), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ckpt_root = Path.cwd() / 'checkpoints'
    if not ckpt_root.exists():
        ckpt_root.mkdir(parents=True, exist_ok=True)
    run_name = f"{config['gym_id']}___{int(time.time())}"
    envs = gym.vector.SyncVectorEnv([
        make_continuous_env(gym_id=config["gym_id"], seed=0, idx=i, run_name=run_name) for i in
        range(config["num_envs"])])
    agent = PPOContinuousAgent(envs)
    observation = envs.reset()

    optimizer = optim.Adam(agent.parameters(), lr=config["lr"], eps=1e-5)
    device = torch.device('cpu')

    obs, actions, logprobs, rewards, dones, values, next_obs, next_done = make_buffers(envs, config, device)
    num_updates = config["total_timesteps"] // config["batch_size"]
    global_step = 0

    for update in tqdm(range(1, num_updates + 1), desc='Update...'):
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * config['lr']
        optimizer.param_groups[0]['lr'] = lrnow

        for step in tqdm(range(0, config["num_steps"]), desc=f'Stepping...{global_step}'):
            envs.envs[0].render(mode='human')
            time.sleep(0.03)
            global_step += 1 * config['num_envs']
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        # compute GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(config["num_steps"])):
                if t == config["num_steps"] - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config["gamma"] * config[
                    "gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(config["batch_size"])
        clipfracs = []
        for epoch in range(config["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, config["batch_size"], config["minibatch_size"]):
                end = start + config["minibatch_size"]
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.inference_mode():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    # old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config["clip_coef"]).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config["clip_coef"], 1 + config["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -config["clip_coef"],
                    config["clip_coef"],
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config["ent_coef"] * entropy_loss + v_loss * config["vf_coef"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config["max_grad_norm"])
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


if __name__ == "__main__":
    main()
