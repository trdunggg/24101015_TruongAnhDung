import os
import csv

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from environment import SwitchGateEnv


# ==========================================================
# Q-LEARNING PARAMETERS
# ==========================================================

ALPHA = 0.1
GAMMA = 0.95

EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

EPISODES = 3000
MAX_STEPS = 150

# Lưu lại quá trình học ở nhiều thời điểm
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

RESULTS_DIR = "results_q_learning"


# ==========================================================
# Q-LEARNING AGENT
# ==========================================================

class QLearningAgent:

    def __init__(
        self,
        env,
        seed=42
    ):

        self.env = env

        self.rng = np.random.default_rng(
            seed
        )

        self.epsilon = EPSILON_START

        # --------------------------------------------------
        # Q-table
        #
        # [row, column, mask, action]
        # --------------------------------------------------

        self.q_table = np.zeros(
            (
                env.observation_space.nvec[0],
                env.observation_space.nvec[1],
                env.observation_space.nvec[2],
                env.action_space.n,
            ),
            dtype=np.float64,
        )

    # ======================================================
    # CHOOSE ACTION
    # ======================================================

    def choose_action(
        self,
        state,
        explore=True
    ):
        """
        Epsilon-greedy.

        explore=True:
            Có thể chọn ngẫu nhiên để khám phá.

        explore=False:
            Chọn action tốt nhất để đánh giá.
        """

        state = tuple(
            map(int, state)
        )

        q_values = self.q_table[state]

        # --------------------------------------------------
        # Exploration
        # --------------------------------------------------

        if (
            explore
            and self.rng.random()
            < self.epsilon
        ):

            return int(
                self.rng.integers(
                    self.env.action_space.n
                )
            )

        # --------------------------------------------------
        # Exploitation
        # --------------------------------------------------

        max_q = np.max(
            q_values
        )

        best_actions = np.flatnonzero(
            np.isclose(
                q_values,
                max_q
            )
        )

        # Nếu có nhiều action cùng giá trị,
        # chọn ngẫu nhiên giữa chúng.
        return int(
            self.rng.choice(
                best_actions
            )
        )

    # ======================================================
    # UPDATE Q-TABLE
    # ======================================================

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        """
        Công thức Q-learning:

        Q(s,a) <- Q(s,a)
                  + alpha[
                      reward
                      + gamma * max Q(s',a')
                      - Q(s,a)
                    ]
        """

        state = tuple(
            map(int, state)
        )

        next_state = tuple(
            map(int, next_state)
        )

        current_q = self.q_table[
            state + (action,)
        ]

        # --------------------------------------------------
        # Nếu trạng thái kết thúc
        # --------------------------------------------------

        if done:

            target = reward

        # --------------------------------------------------
        # Nếu chưa kết thúc
        # --------------------------------------------------

        else:

            next_max_q = np.max(
                self.q_table[
                    next_state
                ]
            )

            target = (
                reward
                + GAMMA * next_max_q
            )

        # --------------------------------------------------
        # Update
        # --------------------------------------------------

        self.q_table[
            state + (action,)
        ] += (
            ALPHA
            * (
                target
                - current_q
            )
        )

    # ======================================================
    # TRAIN
    # ======================================================

    def train(
        self,
        episodes=EPISODES,
        setting=1,
        seed=42
    ):
        """
        Huấn luyện Q-learning.

        Kết quả được lưu:
            - training_history.csv
            - training_checkpoints.csv
            - q_table_episode_*.npy
            - q_table_final.npy
            - reward_curve.png
            - success_curve.png
            - epsilon_curve.png
        """

        output_dir = os.path.join(
            RESULTS_DIR,
            f"setting_{setting}"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        history = []
        checkpoint_rows = []

        # ==================================================
        # TRAINING LOOP
        # ==================================================

        for episode in range(
            1,
            episodes + 1
        ):

            # Episode đầu dùng seed
            if episode == 1:

                state, _ = self.env.reset(
                    seed=seed
                )

            else:

                state, _ = self.env.reset()

            total_reward = 0.0
            success = False
            steps_taken = 0

            # ------------------------------------------------
            # Run one episode
            # ------------------------------------------------

            for step in range(
                MAX_STEPS
            ):

                # Chọn action bằng epsilon-greedy
                action = self.choose_action(
                    state,
                    explore=True
                )

                (
                    next_state,
                    reward,
                    terminated,
                    truncated,
                    _
                ) = self.env.step(
                    action
                )

                done = (
                    terminated
                    or truncated
                )

                # Update Q-table
                self.update(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

                total_reward += reward

                steps_taken = (
                    step + 1
                )

                state = next_state

                if terminated:

                    success = True
                    break

                if truncated:

                    break

            # ------------------------------------------------
            # Giảm epsilon
            # ------------------------------------------------

            self.epsilon = max(
                EPSILON_MIN,
                self.epsilon
                * EPSILON_DECAY
            )

            # ------------------------------------------------
            # Save episode information
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Save checkpoint
            # ------------------------------------------------

            if (
                episode in CHECKPOINTS
                or episode == episodes
            ):

                np.save(
                    os.path.join(
                        output_dir,
                        f"q_table_episode_{episode}.npy"
                    ),
                    self.q_table
                )

                checkpoint_rows.append(
                    record.copy()
                )

            # ------------------------------------------------
            # Print training progress
            # ------------------------------------------------

            if episode % 100 == 0:

                recent = history[
                    -100:
                ]

                avg_reward = np.mean(
                    [
                        x["reward"]
                        for x in recent
                    ]
                )

                success_rate = np.mean(
                    [
                        x["success"]
                        for x in recent
                    ]
                )

                print(
                    f"Q-learning | "
                    f"Setting {setting} | "
                    f"Episode "
                    f"{episode}/{episodes} | "
                    f"Reward TB: "
                    f"{avg_reward:.2f} | "
                    f"Success: "
                    f"{success_rate * 100:.1f}% | "
                    f"Epsilon: "
                    f"{self.epsilon:.3f}"
                )

        # ==================================================
        # SAVE ALL RESULTS
        # ==================================================

        self._save_results(
            history,
            checkpoint_rows,
            output_dir
        )

        return history

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    def _save_results(
        self,
        history,
        checkpoints,
        output_dir
    ):

        fields = [
            "episode",
            "reward",
            "steps",
            "success",
            "epsilon",
        ]

        # --------------------------------------------------
        # Save CSV files
        # --------------------------------------------------

        for filename, rows in [
            (
                "training_history.csv",
                history
            ),
            (
                "training_checkpoints.csv",
                checkpoints
            ),
        ]:

            filepath = os.path.join(
                output_dir,
                filename
            )

            with open(
                filepath,
                "w",
                newline="",
                encoding="utf-8-sig",
            ) as file:

                writer = csv.DictWriter(
                    file,
                    fieldnames=fields
                )

                writer.writeheader()
                writer.writerows(rows)

        # --------------------------------------------------
        # Save final Q-table
        # --------------------------------------------------

        np.save(
            os.path.join(
                output_dir,
                "q_table_final.npy"
            ),
            self.q_table
        )

        # ==================================================
        # PREPARE DATA
        # ==================================================

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
            dtype=float
        )

        successes = np.array(
            [
                x["success"]
                for x in history
            ],
            dtype=float
        )

        epsilons = np.array(
            [
                x["epsilon"]
                for x in history
            ],
            dtype=float
        )

        # --------------------------------------------------
        # Moving average
        # --------------------------------------------------

        window = min(
            MOVING_WINDOW,
            len(history)
        )

        kernel = (
            np.ones(window)
            / window
        )

        avg_rewards = np.convolve(
            rewards,
            kernel,
            mode="valid"
        )

        avg_success = np.convolve(
            successes,
            kernel,
            mode="valid"
        )

        avg_episodes = episodes[
            window - 1:
        ]

        # ==================================================
        # REWARD CURVE
        # ==================================================

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            episodes,
            rewards,
            alpha=0.25,
            label="Reward per episode"
        )

        plt.plot(
            avg_episodes,
            avg_rewards,
            label="Moving average"
        )

        plt.xlabel(
            "Episode"
        )

        plt.ylabel(
            "Total reward"
        )

        plt.title(
            "Q-learning: Training Reward"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "reward_curve.png"
            ),
            dpi=180
        )

        plt.close()

        # ==================================================
        # SUCCESS RATE CURVE
        # ==================================================

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            avg_episodes,
            avg_success * 100
        )

        plt.xlabel(
            "Episode"
        )

        plt.ylabel(
            "Success rate (%)"
        )

        plt.ylim(
            -2,
            102
        )

        plt.title(
            "Q-learning: Success Rate"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "success_curve.png"
            ),
            dpi=180
        )

        plt.close()

        # ==================================================
        # EPSILON CURVE
        # ==================================================

        plt.figure(
            figsize=(9, 5)
        )

        plt.plot(
            episodes,
            epsilons
        )

        plt.xlabel(
            "Episode"
        )

        plt.ylabel(
            "Epsilon"
        )

        plt.title(
            "Q-learning: Exploration Decay"
        )

        plt.grid(
            True,
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                output_dir,
                "epsilon_curve.png"
            ),
            dpi=180
        )

        plt.close()

        print(
            f"Đã lưu kết quả tại: "
            f"{output_dir}"
        )

    # ======================================================
    # EVALUATION
    # ======================================================

    def evaluate(
        self,
        episodes=100
    ):
        """
        Đánh giá Q-table sau khi training.
        Trong evaluation không còn exploration.
        """

        if episodes <= 0:
            raise ValueError(
                "episodes phải lớn hơn 0"
            )

        results = []

        for _ in range(
            episodes
        ):

            state, _ = self.env.reset()

            total_reward = 0.0
            success = False
            steps_taken = 0

            for step in range(
                MAX_STEPS
            ):

                # Không exploration
                action = self.choose_action(
                    state,
                    explore=False
                )

                (
                    state,
                    reward,
                    terminated,
                    truncated,
                    _
                ) = self.env.step(
                    action
                )

                total_reward += reward

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

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------

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
                )
            ),
        }

        print()
        print(
            "Q-learning Evaluation"
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


# ==========================================================
# MAIN
# ==========================================================

def main():

    for setting in range(
        1,
        5
    ):

        print()
        print(
            "========================================"
        )

        print(
            f"Q-learning | Setting {setting}"
        )

        print(
            "========================================"
        )

        env = SwitchGateEnv(
            setting=setting
        )

        try:

            agent = QLearningAgent(
                env,
                seed=42 + setting
            )

            agent.train(
                setting=setting,
                seed=42 + setting
            )

            agent.evaluate(
                episodes=100
            )

        finally:

            env.close()


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()