from environment import SwitchGateEnv


GAMMA = 0.95
THETA = 1e-6
MAX_ITERATIONS = 1000


def get_next_state(env, state, action):
    """
    Lấy trạng thái kế tiếp, reward và trạng thái kết thúc.

    environment.py trả về:
    next_state, reward, terminated, truncated, info
    """

    (
        next_state,
        reward,
        terminated,
        truncated,
        _,
    ) = env.get_transition(
        state,
        action,
    )

    next_state = tuple(
        int(x)
        for x in next_state
    )

    done = (
        terminated
        or truncated
    )

    return (
        next_state,
        float(reward),
        done,
    )


def get_valid_states(env):
    """
    Lấy các trạng thái hợp lệ.

    Loại:
    - Tường
    - Trạng thái đang đứng trên cổng đóng
    """

    valid_states = []

    for state in env.get_all_states():

        row, col, mask = map(
            int,
            state,
        )

        cell = env.grid[row][col]

        if cell == "#":
            continue

        if cell in (
            "G",
            "H",
        ):
            if not env.is_gate_open(
                (row, col),
                mask,
            ):
                continue

        valid_states.append(
            (
                row,
                col,
                mask,
            )
        )

    return valid_states


def value_iteration(
    setting=1,
    env=None,
):
    """
    Value Iteration.

    Nếu truyền env vào thì dùng env đó.
    Nếu không truyền thì tự tạo environment.
    """

    if env is None:
        env = SwitchGateEnv(
            setting=setting
        )

    valid_states = get_valid_states(
        env
    )

    actions = range(
        env.action_space.n
    )

    values = {
        state: 0.0
        for state in valid_states
    }

    # ==========================================
    # VALUE ITERATION
    # ==========================================

    for iteration in range(
        MAX_ITERATIONS
    ):

        delta = 0.0

        new_values = values.copy()

        for state in valid_states:

            row, col, mask = state

            # Goal là trạng thái kết thúc.
            if (
                row,
                col,
            ) == env.goal_pos:

                new_values[state] = 0.0
                continue

            best_value = float(
                "-inf"
            )

            for action in actions:

                (
                    next_state,
                    reward,
                    done,
                ) = get_next_state(
                    env,
                    state,
                    action,
                )

                if done:

                    action_value = reward

                else:

                    action_value = (
                        reward
                        + GAMMA
                        * values.get(
                            next_state,
                            0.0,
                        )
                    )

                best_value = max(
                    best_value,
                    action_value,
                )

            new_values[state] = (
                best_value
            )

            delta = max(
                delta,
                abs(
                    new_values[state]
                    - values[state]
                ),
            )

        values = new_values

        print(
            f"Value Iteration - "
            f"Setting {setting} | "
            f"Iteration {iteration + 1} | "
            f"Delta: {delta:.8f}"
        )

        if delta < THETA:

            print(
                "Giá trị trạng thái đã hội tụ."
            )

            break

    # ==========================================
    # TRÍCH XUẤT POLICY
    # ==========================================

    policy = {}

    for state in valid_states:

        row, col, mask = state

        # Goal
        if (
            row,
            col,
        ) == env.goal_pos:

            policy[state] = 0
            continue

        best_action = 0

        best_value = float(
            "-inf"
        )

        for action in actions:

            (
                next_state,
                reward,
                done,
            ) = get_next_state(
                env,
                state,
                action,
            )

            if done:

                action_value = reward

            else:

                action_value = (
                    reward
                    + GAMMA
                    * values.get(
                        next_state,
                        0.0,
                    )
                )

            if action_value > best_value:

                best_value = (
                    action_value
                )

                best_action = action

        policy[state] = best_action

    return (
        env,
        policy,
        values,
    )


if __name__ == "__main__":

    env, policy, values = (
        value_iteration(
            setting=1
        )
    )

    start_state = (
        *env.start_pos,
        0,
    )

    print()
    print(
        "Value Iteration - Setting 1"
    )

    print(
        "Số trạng thái trong policy:",
        len(policy),
    )

    print(
        "Giá trị trạng thái bắt đầu:",
        values.get(
            start_state
        ),
    )

    print(
        "Hành động tại trạng thái bắt đầu:",
        policy.get(
            start_state
        ),
    )

    env.close()