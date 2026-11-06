import os
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import SwitchGateEnv


# =========================================================
# CONFIGURATION
# =========================================================

GAMMA = 0.95

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.998

EPISODES = 5000
MAX_STEPS = 150

EVAL_EPISODES = 100

CHECKPOINTS = [1, 100, 300, 600, 1000, 2000, 3000, 4000, 5000]

RESULTS_DIR = "results_monte_carlo"

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# MONTE CARLO AGENT
# =========================================================

class MonteCarloAgent:

    def __init__(self, env, seed=42):

        self.env = env
        self.rng = np.random.default_rng(seed)

        self.num_actions = env.action_space.n

        rows, cols, masks = env.observation_space.nvec

        self.Q = np.zeros(
            (rows, cols, masks, self.num_actions),
            dtype=np.float64
        )

        self.returns_sum = np.zeros_like(self.Q)
        self.returns_count = np.zeros_like(self.Q)

        self.epsilon = EPSILON_START

    # -----------------------------------------------------
    # CHECK WHETHER TOGGLE IS ALLOWED
    # -----------------------------------------------------

    def can_toggle(self, state):

        row, col, mask = map(int, state)

        cell = self.env.grid[row][col]

        # Chỉ được Toggle khi đứng trên ô công tắc
        if cell not in ("A", "B"):
            return False

        # A tương ứng bit 0, B tương ứng bit 1
        switch_index = ord(cell) - ord("A")
        switch_bit = 1 << switch_index

        # Nếu công tắc đã bật thì không được Toggle lần nữa
        if mask & switch_bit:
            return False

        # Không cho bật lại công tắc đặc biệt ở Setting 4
        if (
            self.env.setting == 4
            and cell == "A"
            and getattr(self.env, "special_switch_used", False)
        ):
            return False

        return True

    # -----------------------------------------------------
    # VALID ACTIONS
    # -----------------------------------------------------

    def get_valid_actions(self, state):

        row, col, mask = map(int, state)

        # Các hướng di chuyển
        valid_actions = [0, 1, 2, 3]

        # Chỉ thêm Toggle khi công tắc đang tắt
        if self.can_toggle(state):
            valid_actions.append(4)

        return valid_actions

    # -----------------------------------------------------
    # EPSILON-GREEDY ACTION
    # -----------------------------------------------------

    def choose_action(self, state, explore=True):

        valid_actions = self.get_valid_actions(state)

        if not valid_actions:
            return 0

        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.choice(valid_actions))

        row, col, mask = map(int, state)

        q_values = self.Q[row, col, mask, valid_actions]

        best_value = np.max(q_values)

        best_actions = [
            action
            for action, value in zip(valid_actions, q_values)
            if np.isclose(value, best_value)
        ]

        return int(self.rng.choice(best_actions))

    # -----------------------------------------------------
    # GENERATE ONE EPISODE
    # -----------------------------------------------------

    def generate_episode(self, seed=None):

        state, info = self.env.reset(seed=seed)

        episode = []

        total_reward = 0
        reached_goal = False

        for step in range(MAX_STEPS):

            state_tuple = tuple(map(int, state))

            action = self.choose_action(state, explore=True)

            next_state, reward, terminated, truncated, info = (
                self.env.step(action)
            )

            episode.append(
                (state_tuple, action, float(reward))
            )

            total_reward += reward

            state = next_state

            if terminated:
                reached_goal = True
                break

            if truncated:
                break

        return episode, total_reward, len(episode), reached_goal

    # -----------------------------------------------------
    # SAVE CHECKPOINT
    # -----------------------------------------------------

    def save_checkpoint(self, setting, episode_number):

        path = os.path.join(
            RESULTS_DIR,
            f"mc_q_table_setting{setting}_ep{episode_number}.npy"
        )

        np.save(path, self.Q)

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    def train(self, episodes=EPISODES, seed=42):

        history = []

        print("\n======================================")
        print("MONTE CARLO TRAINING - SETTING 2")
        print("======================================")

        for episode_number in range(1, episodes + 1):

            episode_seed = seed + episode_number

            episode, total_reward, steps, success = (
                self.generate_episode(seed=episode_seed)
            )

            # Every-Visit Monte Carlo update
            G = 0

            for state, action, reward in reversed(episode):

                G = reward + GAMMA * G

                row, col, mask = state

                self.returns_sum[row, col, mask, action] += G
                self.returns_count[row, col, mask, action] += 1

                self.Q[row, col, mask, action] = (
                    self.returns_sum[row, col, mask, action]
                    / self.returns_count[row, col, mask, action]
                )

            # Giảm epsilon
            self.epsilon = max(
                EPSILON_MIN,
                self.epsilon * EPSILON_DECAY
            )

            history.append({
                "episode": episode_number,
                "reward": total_reward,
                "steps": steps,
                "success": int(success),
                "epsilon": self.epsilon
            })

            if episode_number in CHECKPOINTS:

                self.save_checkpoint(
                    self.env.setting,
                    episode_number
                )

                recent = history[-50:]

                avg_reward = np.mean([
                    item["reward"] for item in recent
                ])

                success_rate = np.mean([
                    item["success"] for item in recent
                ]) * 100

                print(
                    f"Episode {episode_number:5d} | "
                    f"Avg Reward: {avg_reward:8.2f} | "
                    f"Success: {success_rate:6.2f}% | "
                    f"Epsilon: {self.epsilon:.3f}"
                )

        self.save_final_results(history)

        return history

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    def save_final_results(self, history):

        setting = self.env.setting

        # Save final Q-table
        np.save(
            os.path.join(
                RESULTS_DIR,
                f"mc_q_table_setting{setting}_final.npy"
            ),
            self.Q
        )

        # Save training history
        csv_path = os.path.join(
            RESULTS_DIR,
            f"mc_training_setting{setting}.csv"
        )

        with open(
            csv_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "episode",
                    "reward",
                    "steps",
                    "success",
                    "epsilon"
                ]
            )

            writer.writeheader()
            writer.writerows(history)

        # Save reward plot
        rewards = [item["reward"] for item in history]

        plt.figure(figsize=(10, 5))
        plt.plot(rewards, alpha=0.35, label="Episode reward")

        if len(rewards) >= 50:
            moving_average = np.convolve(
                rewards,
                np.ones(50) / 50,
                mode="valid"
            )

            plt.plot(
                range(49, len(rewards)),
                moving_average,
                label="Moving average (50)"
            )

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Monte Carlo - Training Reward")
        plt.legend()
        plt.grid(True)

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"mc_reward_setting{setting}.png"
            )
        )

        plt.close()

        # Save success rate plot
        success_values = [
            item["success"] for item in history
        ]

        success_rates = []

        for i in range(len(success_values)):
            start = max(0, i - 49)

            success_rates.append(
                np.mean(success_values[start:i + 1]) * 100
            )

        plt.figure(figsize=(10, 5))
        plt.plot(success_rates)

        plt.xlabel("Episode")
        plt.ylabel("Success rate (%)")
        plt.title("Monte Carlo - Success Rate")
        plt.grid(True)

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"mc_success_setting{setting}.png"
            )
        )

        plt.close()

        # Save epsilon plot
        epsilons = [item["epsilon"] for item in history]

        plt.figure(figsize=(10, 5))
        plt.plot(epsilons)

        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Monte Carlo - Epsilon Decay")
        plt.grid(True)

        plt.savefig(
            os.path.join(
                RESULTS_DIR,
                f"mc_epsilon_setting{setting}.png"
            )
        )

        plt.close()

    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------

    def evaluate(self, episodes=EVAL_EPISODES):

        rewards = []
        steps_list = []
        successes = 0

        for episode_number in range(episodes):

            state, info = self.env.reset(
                seed=10000 + episode_number
            )

            total_reward = 0
            reached_goal = False

            for step in range(MAX_STEPS):

                action = self.choose_action(
                    state,
                    explore=False
                )

                next_state, reward, terminated, truncated, info = (
                    self.env.step(action)
                )

                total_reward += reward
                state = next_state

                if terminated:
                    reached_goal = True
                    break

                if truncated:
                    break

            rewards.append(total_reward)
            steps_list.append(step + 1)

            if reached_goal:
                successes += 1

        result = {
            "average_return": float(np.mean(rewards)),
            "success_rate": float(successes / episodes * 100),
            "average_steps": float(np.mean(steps_list)),
            "successes": successes,
            "episodes": episodes
        }

        return result


# =========================================================
# MAIN - TRAIN ONLY MAP 2
# =========================================================

if __name__ == "__main__":

    setting = 2

    env = SwitchGateEnv(setting=setting)

    agent = MonteCarloAgent(
        env=env,
        seed=42
    )

    agent.train(
        episodes=EPISODES,
        seed=42
    )

    print("\n======================================")
    print("MONTE CARLO EVALUATION - SETTING 2")
    print("======================================")

    result = agent.evaluate()

    for key, value in result.items():
        print(f"{key}: {value}")

    env.close()