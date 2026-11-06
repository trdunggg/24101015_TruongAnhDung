import os
import numpy as np

from environment import SwitchGateEnv

from policy_iteration import policy_iteration
from value_iteration import value_iteration


ARROWS = {
    0: "↑",
    1: "↓",
    2: "←",
    3: "→",
    4: "T",
}


def print_policy(
    env,
    policy,
    title,
):
    print()
    print("=" * 50)
    print(title)
    print("=" * 50)

    for r in range(env.rows):

        row_output = []

        for c in range(env.cols):

            cell = env.grid[r][c]

            if cell in (
                "#",
                "S",
                "X",
                "A",
                "B",
                "G",
                "H",
            ):
                row_output.append(cell)

            else:

                state = (
                    r,
                    c,
                    0,
                )

                action = policy.get(
                    state,
                    0,
                )

                row_output.append(
                    ARROWS.get(
                        action,
                        "?",
                    )
                )

        print(
            " ".join(row_output)
        )


def main():

    os.makedirs(
        "results_experiments",
        exist_ok=True,
    )

    for setting in range(1, 5):

        env = SwitchGateEnv(
            setting=setting
        )

        try:

            # -----------------------------
            # Policy Iteration
            # -----------------------------

            (
                _,
                policy,
                values,
            ) = policy_iteration(
                setting=setting,
                env=env,
            )

            print_policy(
                env,
                policy,
                f"Policy Iteration - Setting {setting}",
            )

            # -----------------------------
            # Value Iteration
            # -----------------------------

            (
                _,
                value_policy,
                value_values,
            ) = value_iteration(
                setting=setting,
                env=env,
            )

            print_policy(
                env,
                value_policy,
                f"Value Iteration - Setting {setting}",
            )

            # -----------------------------
            # Save Policy Iteration values
            # -----------------------------

            values_path = os.path.join(
                "results_experiments",
                f"policy_values_setting{setting}.npy",
            )

            np.save(
                values_path,
                np.array(
                    list(values.values())
                ),
            )

            print(
                f"Saved: {values_path}"
            )

            # -----------------------------
            # Save Value Iteration values
            # -----------------------------

            value_values_path = os.path.join(
                "results_experiments",
                f"value_values_setting{setting}.npy",
            )

            np.save(
                value_values_path,
                np.array(
                    list(value_values.values())
                ),
            )

            print(
                f"Saved: {value_values_path}"
            )

        finally:

            env.close()


if __name__ == "__main__":
    main()