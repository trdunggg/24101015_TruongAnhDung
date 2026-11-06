import sys
import random

import pygame
import numpy as np

from environment import SwitchGateEnv
from policy_iteration import policy_iteration, value_iteration


# ==========================================================
# PYGAME SETUP
# ==========================================================

pygame.init()

WIDTH, HEIGHT = 1120, 800
FPS = 60

CELL = 62
BOARD_X = 28
BOARD_Y = 150

AI_MOVE_DELAY = 700
MAX_STEPS = 150


# ==========================================================
# COLORS
# ==========================================================

BG = (18, 24, 40)
PANEL = (29, 39, 60)
LIGHT = (48, 62, 88)

WHITE = (242, 246, 255)
MUTED = (164, 178, 201)

BLUE = (79, 151, 255)
GREEN = (90, 205, 142)
GOLD = (255, 202, 105)
PURPLE = (184, 143, 255)
TEAL = (66, 211, 190)
RED = (246, 104, 117)

FLOOR = (235, 240, 249)
WALL = (68, 79, 102)


# ==========================================================
# SCREEN
# ==========================================================

screen = pygame.display.set_mode((WIDTH, HEIGHT))

pygame.display.set_caption(
    "Switch-Gate Puzzle | Reinforcement Learning"
)

clock = pygame.time.Clock()


# ==========================================================
# ALGORITHMS
# ==========================================================

ALGORITHMS = [
    "Q-Learning",
    "SARSA",
    "Monte Carlo Control",
    "Policy Iteration",
    "Value Iteration",
]


# ==========================================================
# SETTING NAMES
# ==========================================================

SETTING_NAMES = {
    1: "1 Switch / 1 Gate",
    2: "2 Switches / 2 Gates",
    3: "1 Switch / Multiple Gates",
    4: "One-time Switch",
}


# ==========================================================
# ACTION NAMES
# ==========================================================

ACTIONS = [
    "Len",
    "Xuong",
    "Trai",
    "Phai",
    "Cong tac",
]


# ==========================================================
# GLOBAL STATE
# ==========================================================

setting = 1
algorithm = "Q-Learning"

env = None

q_table = None
policy = None

steps = 0
total_reward = 0.0
last_reward = None

episode_done = False
is_running = False
training = False

history = []

message = "Dang khoi tao..."
error_message = ""

last_ai_move = 0


# ==========================================================
# FONT
# ==========================================================

def font(size, bold=False):
    return pygame.font.SysFont(
        "segoeui",
        size,
        bold=bold
    )


# ==========================================================
# TEXT
# ==========================================================

def text(
    value,
    x,
    y,
    size=14,
    color=WHITE,
    bold=False
):
    image = font(
        size,
        bold
    ).render(
        str(value),
        True,
        color
    )

    screen.blit(
        image,
        (x, y)
    )


# ==========================================================
# BOX
# ==========================================================

def box(rect, color, radius=10):
    pygame.draw.rect(
        screen,
        color,
        rect,
        border_radius=radius
    )


# ==========================================================
# BUTTON
# ==========================================================

def draw_button(
    rect,
    label,
    active=False,
    size=12
):

    box(
        rect,
        TEAL if active else LIGHT,
        7
    )

    image = font(
        size,
        True
    ).render(
        label,
        True,
        BG if active else WHITE
    )

    screen.blit(
        image,
        image.get_rect(
            center=rect.center
        )
    )


# ==========================================================
# CHOOSE ACTION FROM Q-TABLE
# ==========================================================

def choose_q_action(table, state):

    r, c, mask = map(
        int,
        state
    )

    q_values = table[
        r,
        c,
        mask
    ]

    best = np.flatnonzero(
        np.isclose(
            q_values,
            q_values.max()
        )
    )

    return int(
        random.choice(
            best.tolist()
        )
    )


# ==========================================================
# TRAIN TABULAR RL
# ==========================================================

def train_tabular_agent():

    global q_table
    global policy
    global training
    global message
    global error_message

    training = True
    error_message = ""

    message = (
        f"Dang huan luyen {algorithm} "
        f"tren setting {setting}..."
    )

    draw()
    pygame.display.flip()

    try:

        gamma = 0.95
        alpha = 0.1

        episodes = 1500

        epsilon = 1.0
        epsilon_min = 0.05
        epsilon_decay = 0.995

        n_actions = env.action_space.n

        q = np.zeros(
            (
                env.rows,
                env.cols,
                env.num_masks,
                n_actions
            ),
            dtype=np.float64
        )

        rng = np.random.default_rng(
            42 + setting
        )

        # --------------------------------------------------
        # EPSILON-GREEDY
        # --------------------------------------------------

        def choose(state, eps):

            r, c, mask = map(
                int,
                state
            )

            if rng.random() < eps:

                return int(
                    rng.integers(
                        n_actions
                    )
                )

            row = q[
                r,
                c,
                mask
            ]

            best = np.flatnonzero(
                np.isclose(
                    row,
                    row.max()
                )
            )

            return int(
                rng.choice(best)
            )

        # --------------------------------------------------
        # MONTE CARLO COUNTS
        # --------------------------------------------------

        mc_count = np.zeros(
            (
                env.rows,
                env.cols,
                env.num_masks,
                n_actions
            ),
            dtype=np.int32
        )

        # ==================================================
        # TRAINING
        # ==================================================

        for episode in range(episodes):

            # =================================================
            # Q-LEARNING / SARSA
            # =================================================

            if algorithm in (
                "Q-Learning",
                "SARSA"
            ):

                state, _ = env.reset()

                action = choose(
                    state,
                    epsilon
                )

                for _ in range(MAX_STEPS):

                    (
                        next_state,
                        reward,
                        terminated,
                        truncated,
                        _
                    ) = env.step(action)

                    done = (
                        terminated
                        or truncated
                    )

                    r, c, mask = map(
                        int,
                        state
                    )

                    nr, nc, nmask = map(
                        int,
                        next_state
                    )

                    # --------------------------------------
                    # Q-LEARNING
                    # --------------------------------------

                    if algorithm == "Q-Learning":

                        if done:

                            target = reward

                        else:

                            target = (
                                reward
                                + gamma
                                * np.max(
                                    q[
                                        nr,
                                        nc,
                                        nmask
                                    ]
                                )
                            )

                    # --------------------------------------
                    # SARSA
                    # --------------------------------------

                    else:

                        next_action = choose(
                            next_state,
                            epsilon
                        )

                        if done:

                            target = reward

                        else:

                            target = (
                                reward
                                + gamma
                                * q[
                                    nr,
                                    nc,
                                    nmask,
                                    next_action
                                ]
                            )

                    # --------------------------------------
                    # UPDATE Q
                    # --------------------------------------

                    q[
                        r,
                        c,
                        mask,
                        action
                    ] += (
                        alpha
                        * (
                            target
                            - q[
                                r,
                                c,
                                mask,
                                action
                            ]
                        )
                    )

                    state = next_state

                    # --------------------------------------
                    # NEXT ACTION
                    # --------------------------------------

                    if (
                        algorithm == "SARSA"
                        and not done
                    ):

                        action = next_action

                    else:

                        action = choose(
                            state,
                            epsilon
                        )

                    if done:
                        break

            # =================================================
            # MONTE CARLO CONTROL
            # =================================================

            else:

                episode_data = []

                state, _ = env.reset()

                for _ in range(MAX_STEPS):

                    action = choose(
                        state,
                        epsilon
                    )

                    (
                        next_state,
                        reward,
                        terminated,
                        truncated,
                        _
                    ) = env.step(action)

                    episode_data.append(
                        (
                            tuple(
                                map(
                                    int,
                                    state
                                )
                            ),
                            int(action),
                            float(reward)
                        )
                    )

                    state = next_state

                    done = (
                        terminated
                        or truncated
                    )

                    if done:
                        break

                # ------------------------------------------
                # EVERY-VISIT MONTE CARLO
                # ------------------------------------------

                G = 0.0

                for state, action, reward in reversed(
                    episode_data
                ):

                    G = (
                        gamma * G
                        + reward
                    )

                    r, c, mask = state

                    mc_count[
                        r,
                        c,
                        mask,
                        action
                    ] += 1

                    count = mc_count[
                        r,
                        c,
                        mask,
                        action
                    ]

                    q[
                        r,
                        c,
                        mask,
                        action
                    ] += (
                        G
                        - q[
                            r,
                            c,
                            mask,
                            action
                        ]
                    ) / count

            # ------------------------------------------------
            # EPSILON DECAY
            # ------------------------------------------------

            epsilon = max(
                epsilon_min,
                epsilon * epsilon_decay
            )

            # ------------------------------------------------
            # SHOW PROGRESS
            # ------------------------------------------------

            if (episode + 1) % 300 == 0:

                message = (
                    f"{algorithm}: "
                    f"Episode {episode + 1}/"
                    f"{episodes} | "
                    f"Epsilon = "
                    f"{epsilon:.3f}"
                )

                draw()
                pygame.display.flip()

        q_table = q
        policy = None

        message = (
            f"Da huan luyen "
            f"{algorithm} xong."
        )

    except Exception as exc:

        error_message = str(exc)

        message = "Huan luyen that bai."

    training = False


# ==========================================================
# TRAIN POLICY / VALUE ITERATION
# ==========================================================

def train_dynamic_programming():

    global q_table
    global policy
    global training
    global message
    global error_message

    training = True
    error_message = ""

    message = (
        f"Dang tinh {algorithm} "
        f"tren setting {setting}..."
    )

    draw()
    pygame.display.flip()

    try:

        if algorithm == "Policy Iteration":

            (
                _,
                policy,
                _
            ) = policy_iteration(
                setting=setting,
                env=env,
                gamma=0.95
            )

        else:

            (
                _,
                policy,
                _
            ) = value_iteration(
                setting=setting,
                env=env,
                gamma=0.95
            )

        q_table = None

        message = (
            f"Da hoan thanh "
            f"{algorithm}."
        )

    except Exception as exc:

        error_message = str(exc)

        message = "Tinh toan that bai."

    training = False


# ==========================================================
# TRAIN CURRENT ALGORITHM
# ==========================================================

def train_agent():

    if algorithm in (
        "Q-Learning",
        "SARSA",
        "Monte Carlo Control"
    ):

        train_tabular_agent()

    else:

        train_dynamic_programming()


# ==========================================================
# CREATE / RECREATE CURRENT GAME
# ==========================================================

def new_game(
    new_setting=None,
    new_algorithm=None
):

    global env
    global setting
    global algorithm

    global steps
    global total_reward
    global last_reward

    global episode_done
    global is_running

    global history
    global message
    global error_message

    global last_ai_move

    is_running = False

    if new_setting is not None:
        setting = new_setting

    if new_algorithm is not None:
        algorithm = new_algorithm

    steps = 0
    total_reward = 0.0
    last_reward = None

    episode_done = False
    history = []

    error_message = ""

    try:

        if env is not None:
            env.close()

        env = SwitchGateEnv(
            setting=setting
        )

        env.reset()

        train_agent()

        is_running = False

        last_ai_move = pygame.time.get_ticks()

    except Exception as exc:

        error_message = str(exc)

        message = (
            "Loi tao game / huan luyen."
        )


# ==========================================================
# RESET CURRENT EPISODE
# ==========================================================

def reset_episode():

    global steps
    global total_reward
    global last_reward

    global episode_done
    global is_running

    global history
    global message

    global last_ai_move

    if env is None:
        return

    env.reset()

    steps = 0
    total_reward = 0.0
    last_reward = None

    episode_done = False
    is_running = False

    history = []

    last_ai_move = pygame.time.get_ticks()

    message = "Da reset luot choi."


# ==========================================================
# CHOOSE AI ACTION
# ==========================================================

def choose_action(state):

    if policy is not None:

        return int(
            policy.get(
                tuple(
                    map(int, state)
                ),
                0
            )
        )

    if q_table is not None:

        return choose_q_action(
            q_table,
            state
        )

    return 0


# ==========================================================
# PERFORM ACTION
# ==========================================================

def perform_action(action):

    global steps
    global total_reward
    global last_reward

    global episode_done
    global is_running

    global message

    if (
        env is None
        or episode_done
    ):
        return

    (
        state,
        reward,
        terminated,
        truncated,
        _
    ) = env.step(action)

    steps += 1

    total_reward += float(reward)
    last_reward = float(reward)

    r, c, mask = map(
        int,
        state
    )

    history.append(
        {
            "step": steps,
            "pos": f"({r},{c})",
            "action": ACTIONS[action],
            "reward": float(reward),
        }
    )

    if terminated:

        episode_done = True
        is_running = False

        message = (
            f"Da toi dich! "
            f"Reward = "
            f"{total_reward:.1f}"
        )

    elif truncated:

        episode_done = True
        is_running = False

        message = "Het gioi han buoc."

    else:

        message = (
            f"{ACTIONS[action]} | "
            f"Reward {reward:.1f}"
        )


# ==========================================================
# DRAW
# ==========================================================

def draw():

    screen.fill(BG)

    # ======================================================
    # HEADER
    # ======================================================

    box(
        (0, 0, WIDTH, 80),
        PANEL,
        0
    )

    text(
        "SWITCH-GATE PUZZLE",
        24,
        10,
        24,
        WHITE,
        True
    )

    text(
        algorithm,
        27,
        44,
        12,
        MUTED
    )

    if env is None:

        pygame.display.flip()
        return

    # ======================================================
    # BOARD
    # ======================================================

    board_w = env.cols * CELL
    board_h = env.rows * CELL

    box(
        (
            BOARD_X - 7,
            BOARD_Y - 7,
            board_w + 14,
            board_h + 14
        ),
        PANEL
    )

    r, c, mask = map(
        int,
        env.state
    )

    # ======================================================
    # DRAW CELLS
    # ======================================================

    for rr in range(env.rows):

        for cc in range(env.cols):

            cell = env.grid[rr][cc]

            if cell == "#":

                color = WALL

            elif cell == "S":

                color = GREEN

            elif cell == "X":

                color = GOLD

            elif cell in ("A", "B"):

                color = PURPLE

            elif cell in ("G", "H"):

                opened = env.is_gate_open(
                    (rr, cc),
                    mask
                )

                color = (
                    TEAL
                    if opened
                    else RED
                )

            else:

                color = FLOOR

            rect = pygame.Rect(
                BOARD_X + cc * CELL,
                BOARD_Y + rr * CELL,
                CELL - 4,
                CELL - 4
            )

            box(
                rect,
                color,
                7
            )

            label = font(
                18,
                True
            ).render(
                cell,
                True,
                BG if cell != "#" else WHITE
            )

            screen.blit(
                label,
                label.get_rect(
                    center=rect.center
                )
            )

            if cell in ("G", "H"):

                opened = env.is_gate_open(
                    (rr, cc),
                    mask
                )

                status = (
                    "OPEN"
                    if opened
                    else "CLOSED"
                )

                small = font(
                    8,
                    True
                ).render(
                    status,
                    True,
                    BG if opened else WHITE
                )

                screen.blit(
                    small,
                    small.get_rect(
                        center=(
                            rect.centerx,
                            rect.bottom - 9
                        )
                    )
                )

    # ======================================================
    # PLAYER
    # ======================================================

    center = (
        BOARD_X
        + c * CELL
        + (CELL - 4) // 2,

        BOARD_Y
        + r * CELL
        + (CELL - 4) // 2
    )

    pygame.draw.circle(
        screen,
        BLUE,
        center,
        16
    )

    pygame.draw.circle(
        screen,
        WHITE,
        center,
        16,
        2
    )

    # ======================================================
    # LEGEND
    # ======================================================

    legend_y = (
        BOARD_Y
        + board_h
        + 7
    )

    text(
        "S: Start",
        BOARD_X,
        legend_y,
        10,
        GREEN
    )

    text(
        "X: Goal",
        BOARD_X + 85,
        legend_y,
        10,
        GOLD
    )

    text(
        "A/B: Switch",
        BOARD_X + 165,
        legend_y,
        10,
        PURPLE
    )

    text(
        "G/H: Gate",
        BOARD_X + 275,
        legend_y,
        10,
        TEAL
    )

    control_text = []

    for gate, switch in env.gate_controller.items():

        control_text.append(
            f"{switch}->{gate}"
        )

    text(
        " | ".join(control_text),
        BOARD_X + 370,
        legend_y,
        10,
        WHITE,
        True
    )

    # ======================================================
    # RIGHT PANEL
    # ======================================================

    px = 640
    pw = 455

    # ======================================================
    # ALGORITHM
    # ======================================================

    box(
        (px, 105, pw, 151),
        PANEL
    )

    text(
        "THUAT TOAN",
        px + 12,
        112,
        11,
        MUTED,
        True
    )

    alg_buttons = [
        (
            px + 10,
            132,
            210,
            29,
            "Q-Learning"
        ),

        (
            px + 230,
            132,
            215,
            29,
            "SARSA"
        ),

        (
            px + 10,
            166,
            210,
            29,
            "Monte Carlo Control"
        ),

        (
            px + 230,
            166,
            215,
            29,
            "Policy Iteration"
        ),

        (
            px + 10,
            200,
            435,
            29,
            "Value Iteration"
        ),
    ]

    for bx, by, bw, bh, name in alg_buttons:

        draw_button(
            pygame.Rect(
                bx,
                by,
                bw,
                bh
            ),
            name,
            active=(
                algorithm == name
            ),
            size=11
        )

    # ======================================================
    # SETTING
    # ======================================================

    box(
        (px, 264, pw, 95),
        PANEL
    )

    text(
        "SETTING - CO CHE CONG / SWITCH",
        px + 12,
        271,
        11,
        MUTED,
        True
    )

    for i in range(1, 5):

        bx = (
            px
            + 10
            + (i - 1) * 111
        )

        draw_button(
            pygame.Rect(
                bx,
                294,
                104,
                30
            ),
            f"Setting {i}",
            active=(
                setting == i
            ),
            size=10
        )

    text(
        SETTING_NAMES[setting],
        px + 12,
        333,
        10,
        WHITE
    )

    # ======================================================
    # STATISTICS
    # ======================================================

    box(
        (px, 367, pw, 87),
        PANEL
    )

    text(
        "THONG KE",
        px + 12,
        374,
        11,
        MUTED,
        True
    )

    text(
        f"Buoc: {steps}",
        px + 12,
        396,
        14,
        BLUE,
        True
    )

    text(
        f"Reward: {total_reward:.1f}",
        px + 130,
        396,
        14,
        GOLD,
        True
    )

    text(
        f"Switch mask: {mask}",
        px + 285,
        396,
        12,
        PURPLE,
        True
    )

    if last_reward is not None:

        text(
            f"Reward buoc gan nhat: "
            f"{last_reward:.1f}",
            px + 12,
            426,
            10,
            MUTED
        )

    # ======================================================
    # CONTROL
    # ======================================================

    box(
        (px, 462, pw, 119),
        PANEL
    )

    text(
        "DIEU KHIEN",
        px + 12,
        469,
        11,
        MUTED,
        True
    )

    draw_button(
        pygame.Rect(
            px + 10,
            490,
            210,
            34
        ),
        (
            "TAM DUNG"
            if is_running
            else "START AI"
        ),
        active=is_running
    )

    draw_button(
        pygame.Rect(
            px + 230,
            490,
            215,
            34
        ),
        "RESET"
    )

    # ======================================================
    # HISTORY
    # ======================================================

    box(
        (px, 590, pw, 177),
        PANEL
    )

    text(
        "LICH SU DI CHUYEN",
        px + 12,
        598,
        11,
        MUTED,
        True
    )

    text(
        "Buoc",
        px + 12,
        621,
        9,
        MUTED,
        True
    )

    text(
        "Vi tri",
        px + 70,
        621,
        9,
        MUTED,
        True
    )

    text(
        "Hanh dong",
        px + 155,
        621,
        9,
        MUTED,
        True
    )

    text(
        "Reward",
        px + 340,
        621,
        9,
        MUTED,
        True
    )

    for i, item in enumerate(history[-5:]):

        yy = 641 + i * 21

        text(
            item["step"],
            px + 15,
            yy,
            10
        )

        text(
            item["pos"],
            px + 70,
            yy,
            10
        )

        text(
            item["action"],
            px + 155,
            yy,
            10,
            TEAL
        )

        text(
            f'{item["reward"]:.1f}',
            px + 340,
            yy,
            10,
            GOLD
        )

    # ======================================================
    # MESSAGE
    # ======================================================

    text(
        message[:100],
        28,
        HEIGHT - 20,
        11,
        MUTED
    )

    # ======================================================
    # ERROR
    # ======================================================

    if error_message:

        box(
            (
                24,
                HEIGHT - 82,
                WIDTH - 48,
                48
            ),
            (85, 35, 47)
        )

        text(
            "LOI: " + error_message[:125],
            34,
            HEIGHT - 67,
            10,
            WHITE
        )

    pygame.display.flip()


# ==========================================================
# HANDLE MOUSE CLICK
# ==========================================================

def handle_click(pos):

    global is_running
    global message
    global last_ai_move

    x, y = pos

    px = 640

    # ------------------------------------------------------
    # ALGORITHM BUTTONS
    # ------------------------------------------------------

    alg_buttons = [
        (
            px + 10,
            132,
            210,
            29,
            "Q-Learning"
        ),

        (
            px + 230,
            132,
            215,
            29,
            "SARSA"
        ),

        (
            px + 10,
            166,
            210,
            29,
            "Monte Carlo Control"
        ),

        (
            px + 230,
            166,
            215,
            29,
            "Policy Iteration"
        ),

        (
            px + 10,
            200,
            435,
            29,
            "Value Iteration"
        ),
    ]

    for bx, by, bw, bh, name in alg_buttons:

        if (
            bx <= x <= bx + bw
            and by <= y <= by + bh
        ):

            new_game(
                new_algorithm=name
            )

            return

    # ------------------------------------------------------
    # SETTING BUTTONS
    # ------------------------------------------------------

    if (
        px + 10 <= x <= px + 447
        and 294 <= y <= 324
    ):

        index = (
            x - (px + 10)
        ) // 111 + 1

        if 1 <= index <= 4:

            new_game(
                new_setting=int(index)
            )

        return

    # ------------------------------------------------------
    # START / PAUSE
    # ------------------------------------------------------

    if (
        px + 10 <= x <= px + 220
        and 490 <= y <= 524
    ):

        if error_message:

            message = (
                "Hay kiem tra loi truoc."
            )

        elif episode_done:

            message = (
                "Nhan RESET de choi lai."
            )

        else:

            is_running = not is_running

            last_ai_move = pygame.time.get_ticks()

            message = (
                "AI dang chay."
                if is_running
                else "Da tam dung."
            )

        return

    # ------------------------------------------------------
    # RESET
    # ------------------------------------------------------

    if (
        px + 230 <= x <= px + 445
        and 490 <= y <= 524
    ):

        reset_episode()

        return


# ==========================================================
# INITIALIZE
# ==========================================================

new_game(
    new_setting=1,
    new_algorithm="Q-Learning"
)


# ==========================================================
# MAIN LOOP
# ==========================================================

running = True

while running:

    clock.tick(FPS)

    # ------------------------------------------------------
    # EVENTS
    # ------------------------------------------------------

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            running = False

        elif (
            event.type == pygame.MOUSEBUTTONDOWN
            and event.button == 1
        ):

            handle_click(event.pos)

        elif event.type == pygame.KEYDOWN:

            # ESC
            if event.key == pygame.K_ESCAPE:

                running = False

            # RESET
            elif event.key == pygame.K_r:

                reset_episode()

            # START / PAUSE
            elif event.key == pygame.K_SPACE:

                if (
                    not error_message
                    and not episode_done
                ):

                    is_running = not is_running

                    last_ai_move = (
                        pygame.time.get_ticks()
                    )

            # SETTINGS 1-4
            elif event.key in (
                pygame.K_1,
                pygame.K_2,
                pygame.K_3,
                pygame.K_4,
            ):

                selected = {
                    pygame.K_1: 1,
                    pygame.K_2: 2,
                    pygame.K_3: 3,
                    pygame.K_4: 4,
                }[event.key]

                new_game(
                    new_setting=selected
                )

    # ======================================================
    # AI MOVEMENT
    # ======================================================

    if (
        is_running
        and not episode_done
        and not training
        and not error_message
    ):

        now = pygame.time.get_ticks()

        if (
            now - last_ai_move
            >= AI_MOVE_DELAY
        ):

            action = choose_action(
                env.state
            )

            perform_action(action)

            last_ai_move = now

    # ======================================================
    # DRAW
    # ======================================================

    draw()


# ==========================================================
# CLOSE
# ==========================================================

if env is not None:
    env.close()

pygame.quit()
sys.exit()