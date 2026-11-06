import os
import csv
import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from environment import SwitchGateEnv

from policy_iteration import (
    policy_iteration,
    value_iteration,
)

from q_learning import QLearningAgent
from sarsa import SARSAAgent
from monte_carlo import MonteCarloAgent


RESULTS_DIR = "results_experiments"

SETTINGS = [1, 2, 3, 4]

SEEDS = [42, 123, 2026]

EVAL_EPISODES = 100

MAX_STEPS = 200

TRAIN_EPISODES = 3000

LEARNING_ALGORITHMS = [
    "Q-learning",
    "SARSA",
    "Monte Carlo",
]


def save_csv(path, rows):

    if not rows:
        return

    folder = os.path.dirname(path)

    if folder:
        os.makedirs(
            folder,
            exist_ok=True,
        )

    fields = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row.keys()
        )
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def summarize(rows):

    return {
        "average_return": float(
            np.mean(
                [
                    row["reward"]
                    for row in rows
                ]
            )
        ),
        "success_rate": float(
            100
            * np.mean(
                [
                    row["success"]
                    for row in rows
                ]
            )
        ),
        "average_steps": float(
            np.mean(
                [
                    row["steps"]
                    for row in rows
                ]
            )
        ),
    }


def evaluate_policy(
    env,
    policy,
    episodes=EVAL_EPISODES,
):

    results = []

    for episode in range(episodes):

        state, _ = env.reset(
            seed=5000 + episode
        )

        total_reward = 0.0
        success = False
        steps = 0

        for step in range(MAX_STEPS):

            state_key = tuple(
                int(x)
                for x in state
            )

            action = int(
                policy.get(
                    state_key,
                    0,
                )
            )

            (
                state,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            total_reward += reward

            steps = step + 1

            if terminated:
                success = True
                break

            if truncated:
                break

        results.append(
            {
                "success": int(success),
                "reward": total_reward,
                "steps": steps,
            }
        )

    return summarize(results)


def evaluate_agent(
    env,
    agent,
    episodes=EVAL_EPISODES,
):

    results = []

    for episode in range(episodes):

        state, _ = env.reset(
            seed=5000 + episode
        )

        total_reward = 0.0
        success = False
        steps = 0

        for step in range(MAX_STEPS):

            action = agent.choose_action(
                state,
                explore=False,
            )

            (
                state,
                reward,
                terminated,
                truncated,
                _,
            ) = env.step(action)

            total_reward += reward

            steps = step + 1

            if terminated:
                success = True
                break

            if truncated:
                break

        results.append(
            {
                "success": int(success),
                "reward": total_reward,
                "steps": steps,
            }
        )

    return summarize(results)


def run_planning_algorithm(
    setting,
    algorithm,
):

    env = SwitchGateEnv(
        setting=setting
    )

    try:

        if algorithm == "Policy Iteration":

            (
                _,
                policy,
                _,
            ) = policy_iteration(
                setting=setting,
                env=env,
            )

        elif algorithm == "Value Iteration":

            (
                _,
                policy,
                _,
            ) = value_iteration(
                setting=setting,
                env=env,
            )

        else:
            raise ValueError(
                "Planning algorithm khong hop le."
            )

        metrics = evaluate_policy(
            env,
            policy,
        )

        return {
            "setting": setting,
            "algorithm": algorithm,
            "seed": "",
            **metrics,
        }

    finally:
        env.close()


def run_learning_algorithm(
    setting,
    seed,
    algorithm,
):

    env = SwitchGateEnv(
        setting=setting
    )

    try:

        if algorithm == "Q-learning":

            agent = QLearningAgent(
                env,
                seed=seed,
            )

        elif algorithm == "SARSA":

            agent = SARSAAgent(
                env,
                seed=seed,
            )

        elif algorithm == "Monte Carlo":

            agent = MonteCarloAgent(
                env,
                seed=seed,
            )

        else:

            raise ValueError(
                f"Khong tim thay algorithm: {algorithm}"
            )

        history = agent.train(
            episodes=TRAIN_EPISODES,
            setting=setting,
            seed=seed,
        )

        metrics = evaluate_agent(
            env,
            agent,
        )

        result = {
            "setting": setting,
            "algorithm": algorithm,
            "seed": seed,
            **metrics,
        }

        history_rows = []

        for item in history:

            history_rows.append(
                {
                    "setting": setting,
                    "algorithm": algorithm,
                    "seed": seed,
                    "episode": item.get(
                        "episode",
                        "",
                    ),
                    "reward": item.get(
                        "reward",
                        item.get(
                            "episode_return",
                            0,
                        ),
                    ),
                    "steps": item.get(
                        "steps",
                        item.get(
                            "episode_steps",
                            0,
                        ),
                    ),
                    "success": item.get(
                        "success",
                        item.get(
                            "reached_goal",
                            0,
                        ),
                    ),
                    "epsilon": item.get(
                        "epsilon",
                        "",
                    ),
                }
            )

        return result, history_rows

    finally:
        env.close()


def plot_learning_curves(
    histories
):

    for setting in SETTINGS:

        for algorithm in LEARNING_ALGORITHMS:

            rows = [
                row
                for row in histories
                if row["setting"] == setting
                and row["algorithm"] == algorithm
            ]

            if not rows:
                continue

            plt.figure(
                figsize=(10, 5)
            )

            for seed in SEEDS:

                seed_rows = sorted(
                    [
                        row
                        for row in rows
                        if row["seed"] == seed
                    ],
                    key=lambda row:
                    int(row["episode"]),
                )

                if not seed_rows:
                    continue

                rewards = np.array(
                    [
                        float(
                            row["reward"]
                        )
                        for row in seed_rows
                    ]
                )

                window = min(
                    100,
                    len(rewards),
                )

                average = np.convolve(
                    rewards,
                    np.ones(window)
                    / window,
                    mode="valid",
                )

                x = np.arange(
                    window,
                    len(rewards) + 1,
                )

                plt.plot(
                    x,
                    average,
                    label=f"Seed {seed}",
                )

            plt.xlabel(
                "Episode"
            )

            plt.ylabel(
                "Average Return"
            )

            plt.title(
                f"{algorithm} Training - Setting {setting}"
            )

            plt.grid(
                True,
                alpha=0.3,
            )

            plt.legend()

            plt.tight_layout()

            folder = os.path.join(
                RESULTS_DIR,
                "learning_curves",
            )

            os.makedirs(
                folder,
                exist_ok=True,
            )

            filename = (
                algorithm
                .lower()
                .replace(
                    " ",
                    "_",
                )
            )

            path = os.path.join(
                folder,
                f"{filename}_setting{setting}.png",
            )

            plt.savefig(
                path,
                dpi=180,
            )

            plt.close()

            print(
                f"Saved curve: {path}"
            )


def build_summary(results):

    summary = []

    algorithms = [
        "Policy Iteration",
        "Value Iteration",
        "Q-learning",
        "SARSA",
        "Monte Carlo",
    ]

    for setting in SETTINGS:

        for algorithm in algorithms:

            rows = [
                row
                for row in results
                if row["setting"] == setting
                and row["algorithm"] == algorithm
            ]

            if not rows:
                continue

            summary.append(
                {
                    "setting": setting,
                    "algorithm": algorithm,
                    "runs": len(rows),
                    "average_return": float(
                        np.mean(
                            [
                                row["average_return"]
                                for row in rows
                            ]
                        )
                    ),
                    "success_rate": float(
                        np.mean(
                            [
                                row["success_rate"]
                                for row in rows
                            ]
                        )
                    ),
                    "average_steps": float(
                        np.mean(
                            [
                                row["average_steps"]
                                for row in rows
                            ]
                        )
                    ),
                }
            )

    return summary


def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print(
        "SWITCH-GATE PUZZLE"
    )
    print(
        "5 ALGORITHM COMPARISON"
    )
    print("=" * 70)

    all_results = []
    all_histories = []

    # Planning algorithms
    for setting in SETTINGS:

        for algorithm in [
            "Policy Iteration",
            "Value Iteration",
        ]:

            print()
            print(
                f"{algorithm} - Setting {setting}"
            )

            result = run_planning_algorithm(
                setting,
                algorithm,
            )

            all_results.append(
                result
            )

            print(
                f"Return = "
                f"{result['average_return']:.2f}"
            )

            print(
                f"Success = "
                f"{result['success_rate']:.1f}%"
            )

            print(
                f"Steps = "
                f"{result['average_steps']:.2f}"
            )

    # Learning algorithms
    for setting in SETTINGS:

        for algorithm in LEARNING_ALGORITHMS:

            for seed in SEEDS:

                print()
                print(
                    "-" * 60
                )

                print(
                    f"{algorithm} | "
                    f"Setting {setting} | "
                    f"Seed {seed}"
                )

                result, history = (
                    run_learning_algorithm(
                        setting,
                        seed,
                        algorithm,
                    )
                )

                all_results.append(
                    result
                )

                all_histories.extend(
                    history
                )

                print(
                    f"Return = "
                    f"{result['average_return']:.2f}"
                )

                print(
                    f"Success = "
                    f"{result['success_rate']:.1f}%"
                )

                print(
                    f"Steps = "
                    f"{result['average_steps']:.2f}"
                )

    save_csv(
        os.path.join(
            RESULTS_DIR,
            "all_results.csv",
        ),
        all_results,
    )

    save_csv(
        os.path.join(
            RESULTS_DIR,
            "learning_history.csv",
        ),
        all_histories,
    )

    summary = build_summary(
        all_results
    )

    save_csv(
        os.path.join(
            RESULTS_DIR,
            "comparison_summary.csv",
        ),
        summary,
    )

    plot_learning_curves(
        all_histories
    )

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    for row in summary:

        print(
            f"Setting {row['setting']} | "
            f"{row['algorithm']:18s} | "
            f"Return: "
            f"{row['average_return']:7.2f} | "
            f"Success: "
            f"{row['success_rate']:6.1f}% | "
            f"Steps: "
            f"{row['average_steps']:6.2f}"
        )

    print()
    print(
        "Experiment completed."
    )


if __name__ == "__main__":
    main()