import numpy as np


GAMMA = 0.95
THETA = 1e-6
MAX_ITERATIONS = 1000


# ==========================================================
# GET NEXT STATE
# ==========================================================

def get_next_state(env, state, action):
    """
    Lấy trạng thái kế tiếp từ environment.

    environment.py hiện tại trả về:

        (next_state, reward, terminated, truncated, info)

    Hàm này chuẩn hóa về:

        (next_state, reward, done)
    """

    # ------------------------------------------------------
    # Setting 4:
    # mask = 1 nghĩa là công tắc A đã được bật.
    #
    # special_switch_used là biến phụ trong environment.
    # Vì Policy Iteration chỉ truyền state nên cần đồng bộ
    # biến này từ mask.
    # ------------------------------------------------------

    old_special_used = env.special_switch_used

    if env.setting == 4:
        try:
            env.special_switch_used = bool(
                int(state[2]) & 1
            )

            result = env.get_transition(
                state,
                action
            )

        finally:
            env.special_switch_used = old_special_used

    else:
        result = env.get_transition(
            state,
            action
        )

    if (
        not isinstance(result, (tuple, list))
        or len(result) != 5
    ):
        raise ValueError(
            "env.get_transition() phải trả về 5 giá trị."
        )

    next_state, reward, terminated, truncated, info = result

    # Kiểm tra next_state
    if np.isscalar(next_state):
        raise TypeError(
            "next_state đang là số thay vì một trạng thái."
        )

    next_state = tuple(
        int(x)
        for x in next_state
    )

    done = bool(
        terminated or truncated
    )

    return (
        next_state,
        float(reward),
        done
    )


# ==========================================================
# GET STATES
# ==========================================================

def get_states(env):
    """
    Lấy toàn bộ trạng thái của môi trường.
    """

    states = env.get_all_states()

    return [
        tuple(int(x) for x in state)
        for state in states
    ]


# ==========================================================
# POLICY ITERATION
# ==========================================================

def policy_iteration(
    setting=1,
    env=None,
    gamma=GAMMA
):
    """
    Policy Iteration cho Switch-Gate Puzzle.

    Trả về:
        env
        policy
        values
    """

    # Nếu chưa truyền environment thì tự tạo
    if env is None:

        from environment import SwitchGateEnv

        env = SwitchGateEnv(
            setting=setting
        )

    states = get_states(env)

    actions = range(
        env.action_space.n
    )

    # ------------------------------------------------------
    # Khởi tạo Value Function
    # ------------------------------------------------------

    values = {
        state: 0.0
        for state in states
    }

    # ------------------------------------------------------
    # Khởi tạo Policy
    #
    # Mặc định chọn action 0 = Up
    # Sau đó Policy Improvement sẽ cải thiện.
    # ------------------------------------------------------

    policy = {
        state: 0
        for state in states
    }

    # ======================================================
    # MAIN POLICY ITERATION LOOP
    # ======================================================

    for iteration in range(
        MAX_ITERATIONS
    ):

        # ==================================================
        # 1. POLICY EVALUATION
        # ==================================================

        for _ in range(
            MAX_ITERATIONS
        ):

            delta = 0.0

            for state in states:

                row, col, mask = state

                # ------------------------------------------------
                # Goal là trạng thái kết thúc.
                # Giá trị tại goal = 0.
                # ------------------------------------------------

                if (
                    row,
                    col
                ) == env.goal_pos:

                    values[state] = 0.0
                    continue

                old_value = values[state]

                action = int(
                    policy[state]
                )

                (
                    next_state,
                    reward,
                    done
                ) = get_next_state(
                    env,
                    state,
                    action
                )

                # ----------------------------------------------
                # Tính giá trị tương lai
                # ----------------------------------------------

                future_value = 0.0

                if (
                    not done
                    and next_state in values
                ):

                    future_value = values[
                        next_state
                    ]

                new_value = (
                    reward
                    + gamma * future_value
                )

                values[state] = new_value

                delta = max(
                    delta,
                    abs(
                        old_value
                        - new_value
                    )
                )

            # Hội tụ
            if delta < THETA:
                break

        # ==================================================
        # 2. POLICY IMPROVEMENT
        # ==================================================

        policy_stable = True

        for state in states:

            row, col, mask = state

            # Goal không cần policy
            if (
                row,
                col
            ) == env.goal_pos:

                continue

            old_action = int(
                policy[state]
            )

            best_action = old_action
            best_value = -float("inf")

            # ----------------------------------------------
            # Thử tất cả action
            # ----------------------------------------------

            for action in actions:

                (
                    next_state,
                    reward,
                    done
                ) = get_next_state(
                    env,
                    state,
                    action
                )

                future_value = 0.0

                if (
                    not done
                    and next_state in values
                ):

                    future_value = values[
                        next_state
                    ]

                q_value = (
                    reward
                    + gamma * future_value
                )

                if q_value > best_value:

                    best_value = q_value
                    best_action = action

            policy[state] = int(
                best_action
            )

            # Kiểm tra policy có thay đổi không
            if old_action != best_action:

                policy_stable = False

        # ==================================================
        # CHECK CONVERGENCE
        # ==================================================

        if policy_stable:

            print(
                f"Policy Iteration setting {setting} "
                f"hội tụ sau {iteration + 1} vòng."
            )

            break

    return (
        env,
        policy,
        values
    )


# ==========================================================
# VALUE ITERATION
# ==========================================================

def value_iteration(
    setting=1,
    env=None,
    gamma=GAMMA
):
    """
    Value Iteration cho Switch-Gate Puzzle.

    Trả về:
        env
        policy
        values
    """

    # Nếu chưa có environment
    if env is None:

        from environment import SwitchGateEnv

        env = SwitchGateEnv(
            setting=setting
        )

    states = get_states(env)

    actions = range(
        env.action_space.n
    )

    # ------------------------------------------------------
    # Khởi tạo Value Function
    # ------------------------------------------------------

    values = {
        state: 0.0
        for state in states
    }

    # ======================================================
    # VALUE ITERATION
    # ======================================================

    for iteration in range(
        MAX_ITERATIONS
    ):

        delta = 0.0

        for state in states:

            row, col, mask = state

            # Goal
            if (
                row,
                col
            ) == env.goal_pos:

                values[state] = 0.0
                continue

            old_value = values[state]

            best_value = -float("inf")

            # ----------------------------------------------
            # Tìm action tốt nhất
            # ----------------------------------------------

            for action in actions:

                (
                    next_state,
                    reward,
                    done
                ) = get_next_state(
                    env,
                    state,
                    action
                )

                future_value = 0.0

                if (
                    not done
                    and next_state in values
                ):

                    future_value = values[
                        next_state
                    ]

                q_value = (
                    reward
                    + gamma * future_value
                )

                best_value = max(
                    best_value,
                    q_value
                )

            values[state] = best_value

            delta = max(
                delta,
                abs(
                    old_value
                    - best_value
                )
            )

        # --------------------------------------------------
        # Kiểm tra hội tụ
        # --------------------------------------------------

        if delta < THETA:

            print(
                f"Value Iteration setting {setting} "
                f"hội tụ sau {iteration + 1} vòng."
            )

            break

    # ======================================================
    # EXTRACT POLICY
    # ======================================================

    policy = {}

    for state in states:

        row, col, mask = state

        # Goal
        if (
            row,
            col
        ) == env.goal_pos:

            policy[state] = 0
            continue

        best_action = 0
        best_value = -float("inf")

        for action in actions:

            (
                next_state,
                reward,
                done
            ) = get_next_state(
                env,
                state,
                action
            )

            future_value = 0.0

            if (
                not done
                and next_state in values
            ):

                future_value = values[
                    next_state
                ]

            q_value = (
                reward
                + gamma * future_value
            )

            if q_value > best_value:

                best_value = q_value
                best_action = action

        policy[state] = int(
            best_action
        )

    return (
        env,
        policy,
        values
    )


# ==========================================================
# PRINT POLICY
# ==========================================================

def print_policy(
    env,
    policy
):
    """
    In policy ra màn hình.

    ↑ = Up
    ↓ = Down
    ← = Left
    → = Right
    T = Toggle
    """

    arrows = {
        0: "↑",
        1: "↓",
        2: "←",
        3: "→",
        4: "T",
    }

    print()
    print(
        f"===== POLICY - SETTING {env.setting} ====="
    )

    print(
        "Gate control:",
        env.gate_controller
    )

    print(
        "T = bật/tắt công tắc"
    )

    print()

    for r in range(
        env.rows
    ):

        row_output = []

        for c in range(
            env.cols
        ):

            cell = env.grid[r][c]

            # ----------------------------------------------
            # Giữ nguyên ký hiệu đặc biệt
            # ----------------------------------------------

            if cell in (
                "#",
                "X",
                "S",
                "A",
                "B",
                "G",
                "H",
            ):

                row_output.append(
                    cell
                )

            else:

                # Policy ở mask = 0
                state = (
                    r,
                    c,
                    0
                )

                action = policy.get(
                    state,
                    0
                )

                row_output.append(
                    arrows.get(
                        action,
                        "?"
                    )
                )

        print(
            " ".join(row_output)
        )


# ==========================================================
# TEST FILE
# ==========================================================

if __name__ == "__main__":

    for setting in range(1, 5):

        print()
        print(
            "================================"
        )
        print(
            f"SETTING {setting}"
        )
        print(
            "================================"
        )

        env, policy, values = policy_iteration(
            setting=setting
        )

        print_policy(
            env,
            policy
        )

        start_state = (
            env.start_pos[0],
            env.start_pos[1],
            0
        )

        print()
        print(
            "Giá trị trạng thái bắt đầu:",
            values.get(
                start_state,
                0.0
            )
        )