import os
import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from environment import SwitchGateEnv


CHECKPOINTS = [
    1,
    100,
    300,
    600,
    1000,
    2000,
    3000,
]

RESULTS_DIR = "results_q_learning"
OUTPUT_DIR = "training_visualizations"
MAX_STEPS = 150


def load_q(setting, episode):
    """
    Load Q-table từ đúng thư mục mà q_learning.py tạo ra.

    Cấu trúc:
    results_q_learning/
        setting_1/
            q_table_episode_1.npy
            q_table_episode_100.npy
            ...
    """

    path = os.path.join(
        RESULTS_DIR,
        f"setting_{setting}",
        f"q_table_episode_{episode}.npy",
    )

    if os.path.exists(path):
        return np.load(path)

    return None


def choose_greedy_action(q_values, rng):
    """
    Chọn action có Q-value lớn nhất.

    Nếu nhiều action có cùng Q-value,
    chọn ngẫu nhiên trong số các action tốt nhất.
    """

    max_q = np.max(q_values)

    best_actions = np.flatnonzero(
        np.isclose(
            q_values,
            max_q,
        )
    )

    return int(
        rng.choice(best_actions)
    )


def run(env, q_table, seed):
    """
    Chạy agent bằng Q-table đã học.

    Không exploration.
    Chỉ chọn action tốt nhất.
    """

    state, _ = env.reset(
        seed=seed
    )

    path = [
        tuple(
            map(int, state)
        )
    ]

    total_reward = 0.0
    success = False

    rng = np.random.default_rng(
        seed
    )

    for _ in range(MAX_STEPS):

        row, col, mask = map(
            int,
            state,
        )

        q_values = q_table[
            row,
            col,
            mask,
        ]

        action = choose_greedy_action(
            q_values,
            rng,
        )

        (
            state,
            reward,
            terminated,
            truncated,
            _,
        ) = env.step(action)

        path.append(
            tuple(
                map(int, state)
            )
        )

        total_reward += reward

        if terminated:

            success = True
            break

        if truncated:
            break

    return (
        path,
        total_reward,
        success,
    )


def draw(ax, env, path):
    """
    Vẽ map và đường đi của agent.
    """

    initial_mask = path[0][2]

    for r in range(env.rows):

        for c in range(env.cols):

            cell = env.grid[r][c]

            color = {
                "#": "#444444",
                "S": "#b8e6b8",
                "X": "#ffe08a",
                "A": "#b9d9ff",
                "B": "#b9d9ff",
            }.get(
                cell,
                "#f4f4f4",
            )

            # Hiển thị trạng thái cổng
            if cell in (
                "G",
                "H",
            ):

                if env.is_gate_open(
                    (r, c),
                    initial_mask,
                ):
                    color = "#a8e6cf"
                else:
                    color = "#f3aaaa"

            ax.add_patch(
                plt.Rectangle(
                    (c, r),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="gray",
                )
            )

            if cell != "#":

                ax.text(
                    c + 0.5,
                    r + 0.5,
                    cell,
                    ha="center",
                    va="center",
                    fontsize=9,
                )

    # Tọa độ đường đi
    xs = [
        state[1] + 0.5
        for state in path
    ]

    ys = [
        state[0] + 0.5
        for state in path
    ]

    # Vẽ đường đi
    ax.plot(
        xs,
        ys,
        linewidth=2,
        marker="o",
        markersize=3,
    )

    # Đánh dấu vị trí cuối
    ax.scatter(
        xs[-1],
        ys[-1],
        s=90,
        marker="*",
        zorder=5,
    )

    ax.set_xlim(
        0,
        env.cols,
    )

    ax.set_ylim(
        env.rows,
        0,
    )

    ax.set_aspect(
        "equal"
    )

    ax.axis("off")


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    for setting in range(
        1,
        5,
    ):

        env = SwitchGateEnv(
            setting=setting
        )

        try:

            available = []

            # ------------------------------------------
            # Tìm checkpoint thực tế
            # ------------------------------------------

            for episode in CHECKPOINTS:

                q_table = load_q(
                    setting,
                    episode,
                )

                if q_table is not None:

                    available.append(
                        (
                            episode,
                            q_table,
                        )
                    )

            if not available:

                print(
                    f"Setting {setting}: "
                    f"chưa có Q-table checkpoint."
                )

                continue

            print(
                f"\nSetting {setting}: "
                f"tìm thấy {len(available)} checkpoint."
            )

            # ------------------------------------------
            # Tạo figure
            # ------------------------------------------

            fig, axes = plt.subplots(
                1,
                len(available),
                figsize=(
                    4 * len(available),
                    5,
                ),
                squeeze=False,
            )

            # ------------------------------------------
            # Vẽ từng checkpoint
            # ------------------------------------------

            for i, (
                episode,
                q_table,
            ) in enumerate(
                available
            ):

                seed = (
                    1000
                    + setting * 10
                    + episode
                )

                (
                    trajectory,
                    reward,
                    success,
                ) = run(
                    env,
                    q_table,
                    seed,
                )

                ax = axes[0, i]

                draw(
                    ax,
                    env,
                    trajectory,
                )

                status = (
                    "Thành công"
                    if success
                    else "Chưa tới đích"
                )

                steps = (
                    len(trajectory)
                    - 1
                )

                ax.set_title(
                    f"Episode {episode}\n"
                    f"{status}\n"
                    f"Steps {steps}, "
                    f"Reward {reward:.1f}"
                )

            fig.suptitle(
                f"Setting {setting} "
                f"- Q-learning checkpoints"
            )

            fig.tight_layout()

            output_path = os.path.join(
                OUTPUT_DIR,
                f"q_learning_trajectory_setting{setting}.png",
            )

            fig.savefig(
                output_path,
                dpi=180,
                bbox_inches="tight",
            )

            plt.close(fig)

            print(
                "Đã lưu:",
                output_path,
            )

        finally:

            env.close()


if __name__ == "__main__":
    main()