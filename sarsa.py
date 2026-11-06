import os
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment import SwitchGateEnv


ALPHA = 0.1
GAMMA = 0.95

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

EPISODES = 3000
MAX_STEPS = 200

CHECKPOINTS = [
    1,
    100,
    300,
    600,
    1000,
    2000,
    3000,
]

MOVING_WINDOW = 50

RESULTS_DIR = "results_sarsa"


class SARSAAgent:

    def __init__(self, env, seed=42):

        self.env = env

        self.rng = np.random.default_rng(seed)

        self.epsilon = EPSILON_START

        self.q_table = np.zeros(
            (
                env.observation_space.nvec[0],
                env.observation_space.nvec[1],
                env.observation_space.nvec[2],
                env.action_space.n,
            ),
            dtype=np.float64,
        )

    def choose_action(self, state, explore=True):

        state = tuple(
            map(int, state)
        )

        q_values = self.q_table[state]

        # Epsilon-greedy:
        # Một phần thời gian chọn ngẫu nhiên để khám phá.
        if (
            explore
            and self.rng.random() < self.epsilon
        ):
            return int(
                self.rng.integers(
                    self.env.action_space.n
                )
            )

        # Chọn action có Q-value lớn nhất.
        best = np.flatnonzero(
            np.isclose(
                q_values,
                np.max(q_values),
            )
        )

        # Nếu có nhiều action bằng nhau,
        # chọn ngẫu nhiên giữa chúng.
        return int(
            self.rng.choice(best)
        )

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_action,
        done,
    ):

        state = tuple(
            map(int, state)
        )

        next_state = tuple(
            map(int, next_state)
        )

        current = self.q_table[
            state + (action,)
        ]

        # Nếu episode kết thúc:
        # không còn Q-value của state tiếp theo.
        if done:

            target = reward

        else:

            target = (
                reward
                + GAMMA
                * self.q_table[
                    next_state
                    + (next_action,)
                ]
            )

        # Công thức SARSA:
        #
        # Q <- Q + alpha * (target - Q)
        #
        self.q_table[
            state + (action,)
        ] += (
            ALPHA
            * (
                target
                - current
            )
        )

    def train(
        self,
        episodes=EPISODES,
        setting=1,
        seed=42,
    ):

        output_dir = os.path.join(
            RESULTS_DIR,
            f"setting_{setting}",
        )

        os.makedirs(
            output_dir,
            exist_ok=True,
        )

        history = []
        checkpoint_rows = []

        print()
        print("=" * 60)
        print(
            f"SARSA TRAINING | SETTING {setting}"
        )
        print("=" * 60)

        for episode in range(
            1,
            episodes + 1,
        ):

            # Episode đầu dùng seed được chỉ định.
            # Các episode sau để environment tự tạo trạng thái reset.
            if episode == 1:

                state, _ = self.env.reset(
                    seed=seed
                )

            else:

                state, _ = self.env.reset()

            action = self.choose_action(
                state,
                explore=True,
            )

            total_reward = 0.0

            success = False

            steps_taken = 0

            for step in range(
                MAX_STEPS
            ):

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = self.env.step(
                    action
                )

                done = (
                    terminated
                    or truncated
                )

                # Nếu episode chưa kết thúc,
                # SARSA chọn action tiếp theo
                # bằng chính policy đang dùng.
                if done:

                    next_action = None

                else:

                    next_action = (
                        self.choose_action(
                            next_state,
                            explore=True,
                        )
                    )

                self.update(
                    state,
                    action,
                    reward,
                    next_state,
                    next_action,
                    done,
                )

                total_reward += float(
                    reward
                )

                steps_taken = (
                    step + 1
                )

                state = next_state

                if terminated:

                    success = True
                    break

                if truncated:

                    break

                action = next_action

            # Giảm epsilon sau mỗi episode.
            self.epsilon = max(
                EPSILON_MIN,
                self.epsilon
                * EPSILON_DECAY,
            )

            record = {
                "episode": episode,
                "reward": total_reward,
                "steps": steps_taken,
                "success": int(success),
                "epsilon": self.epsilon,
            }

            history.append(
                record
            )

            # Lưu các mốc training.
            if (
                episode in CHECKPOINTS
                or episode == episodes
            ):

                np.save(
                    os.path.join(
                        output_dir,
                        f"q_table_episode_{episode}.npy",
                    ),
                    self.q_table,
                )

                checkpoint_rows.append(
                    record.copy()
                )

            # In tiến trình mỗi 100 episode.
            if episode % 100 == 0:

                recent = history[
                    -100:
                ]

                average_reward = np.mean(
                    [
                        x["reward"]
                        for x in recent
                    ]
                )

                success_rate = (
                    100
                    * np.mean(
                        [
                            x["success"]
                            for x in recent
                        ]
                    )
                )

                print(
                    f"SARSA | "
                    f"Setting {setting} | "
                    f"Episode "
                    f"{episode}/{episodes} | "
                    f"Reward TB: "
                    f"{average_reward:.2f} | "
                    f"Success: "
                    f"{success_rate:.1f}% | "
                    f"Epsilon: "
                    f"{self.epsilon:.3f}"
                )

        self._save_results(
            history,
            checkpoint_rows,
            output_dir,
        )

        return history

    def _save_results(
        self,
        history,
        checkpoints,
        output_dir,
    ):

        fields = [
            "episode",
            "reward",
            "steps",
            "success",
            "epsilon",
        ]

        # Lưu toàn bộ lịch sử training.
        with open(
            os.path.join(
                output_dir,
                "training_history.csv",
            ),
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(
                history
            )

        # Lưu các checkpoint.
        with open(
            os.path.join(
                output_dir,
                "training_checkpoints.csv",
            ),
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fields,
            )

            writer.writeheader()
            writer.writerows(
                checkpoints
            )

        # Lưu Q-table cuối cùng.
        np.save(
            os.path.join(
                output_dir,
                "q_table_final.npy",
            ),
            self.q_table,
        )

        if not history:
            return

        episodes = np.array(
            [
                x["episode"]
                for x in history
            ]
        )

        rewards = np.array(
            [
                x["reward"]
                for x in history
            ],
            dtype=float,
        )

        successes = np.array(
            [
                x["success"]
                for x in history
            ],
            dtype=float,
        )

        epsilons = np.array(
            [
                x["epsilon"]
                for x in history
            ],
            dtype=float,
        )

        window = min(
            MOVING_WINDOW,
            len(history),
        )

        kernel = (
            np.ones(window)
            / window
        )

        avg_rewards = np.convolve(
            rewards,
            kernel,
            mode="valid",
        )

        avg_success = np.convolve(
            successes,
            kernel,
            mode="valid",
        )

        avg_episodes = episodes[
            window - 1:
        ]

        # -------------------------------------------------
        # Reward curve
        # -------------------------------------------------

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            episodes,
            rewards,
            alpha=0.25,
            label="Reward per episode",
        )

        plt.plot(
            avg_episodes,
            avg_rewards,
            label="Moving average",
        )

        plt.xlabel(
            "Episode"
        )

        plt.ylabel(
            "Total reward"
        )

        plt.title(
            "SARSA: Training Reward"
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "reward_curve.png",
            ),
            dpi=180,
        )

        plt.close()

        # -------------------------------------------------
        # Success rate
        # -------------------------------------------------

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            avg_episodes,
            avg_success * 100,
        )

        plt.xlabel(
            "Episode"
        )

        plt.ylabel(
            "Success rate (%)"
        )

        plt.ylim(
            -2,
            102,
        )

        plt.title(
            "SARSA: Success Rate"
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "success_curve.png",
            ),
            dpi=180,
        )

        plt.close()

        # -------------------------------------------------
        # Epsilon curve
        # -------------------------------------------------

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            episodes,
            epsilons,
        )

        plt.xlabel(
            "Episode"
        )

        plt.ylabel(
            "Epsilon"
        )

        plt.title(
            "SARSA: Exploration Decay"
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "epsilon_curve.png",
            ),
            dpi=180,
        )

        plt.close()

        print(
            f"Đã lưu kết quả tại: "
            f"{output_dir}"
        )

    def evaluate(
        self,
        episodes=100,
    ):

        if episodes <= 0:
            raise ValueError(
                "episodes phải lớn hơn 0"
            )

        results = []

        # Evaluation không khám phá.
        # Chỉ sử dụng policy tốt nhất hiện tại.
        old_epsilon = self.epsilon

        self.epsilon = 0.0

        for episode in range(
            episodes
        ):

            state, _ = self.env.reset(
                seed=10000 + episode
            )

            total_reward = 0.0

            success = False

            steps_taken = 0

            for step in range(
                MAX_STEPS
            ):

                action = (
                    self.choose_action(
                        state,
                        explore=False,
                    )
                )

                (
                    state,
                    reward,
                    terminated,
                    truncated,
                    _,
                ) = self.env.step(
                    action
                )

                total_reward += float(
                    reward
                )

                steps_taken = (
                    step + 1
                )

                if terminated:

                    success = True
                    break

                if truncated:

                    break

            results.append(
                {
                    "success": int(
                        success
                    ),
                    "reward": total_reward,
                    "steps": steps_taken,
                }
            )

        self.epsilon = old_epsilon

        summary = {
            "success_rate": float(
                np.mean(
                    [
                        x["success"]
                        for x in results
                    ]
                )
            ),
            "average_reward": float(
                np.mean(
                    [
                        x["reward"]
                        for x in results
                    ]
                )
            ),
            "average_steps": float(
                np.mean(
                    [
                        x["steps"]
                        for x in results
                    ]
                ),
            ),
        }

        print()
        print(
            "SARSA Evaluation"
        )

        print(
            f"Success rate: "
            f"{summary['success_rate'] * 100:.1f}%"
        )

        print(
            f"Average reward: "
            f"{summary['average_reward']:.2f}"
        )

        print(
            f"Average steps: "
            f"{summary['average_steps']:.2f}"
        )

        return summary


def main():

    for setting in range(
        1,
        5,
    ):

        print(
            f"\n===== SARSA | "
            f"Setting {setting} ====="
        )

        env = SwitchGateEnv(
            setting=setting
        )

        try:

            agent = SARSAAgent(
                env,
                seed=42 + setting,
            )

            agent.train(
                setting=setting,
                seed=42 + setting,
            )

            agent.evaluate(
                episodes=100
            )

        finally:

            env.close()


if __name__ == "__main__":
    main()