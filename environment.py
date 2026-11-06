import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SwitchGateEnv(gym.Env):
    """
    Switch-Gate Puzzle Environment

    Settings:
        1: 1 switch (A) controls 1 gate (G).
        2: 2 switches (A, B) control 2 gates (G, H).
        3: 1 switch (A) controls 2 gates (G, H).
        4: 1 switch (A) controls 1 gate (G), and A can only be activated once.

    Symbols:
        S = Start
        X = Goal
        A/B = Switch
        G/H = Gate
        # = Wall
        . = Floor

    Actions:
        0 = Up
        1 = Down
        2 = Left
        3 = Right
        4 = Toggle switch

    State:
        (row, column, switch_mask)
    """

    metadata = {"render_modes": ["human"]}

    # --------------------------------------------------
    # MAPS
    # --------------------------------------------------

    MAPS = {
        # Setting 1:
        # A controls G
        1: [
            "S..#...",
            ".#A#...",
            "...G...",
            ".###...",
            "....#X.",
        ],

        # Setting 2:
        # A controls G
        # B controls H
        2: [
            "S..######",
            "#A.######",
            "##.G...##",
            "####B.H.#",
            "######..#",
            "####.##.#",
            "####...X#",
        ],

        # Setting 3:
        # A controls both G and H
        3: [
            "S..######",
            ".#A######",
            "...G..H..",
            "###.###..",
            "#######.X",
        ],

        # Setting 4:
        # A controls G
        # A can only be activated once
        4: [
            "S..######",
            ".#A######",
            "...G.....",
            "###.####.",
            "#######.X",
        ],
    }

    # --------------------------------------------------
    # REWARD CONFIGURATION
    # --------------------------------------------------

    REWARD_CONFIGS = {
        "default": {
            "step": -1.0,
            "goal": 10.0,
            "invalid_toggle": -2.0,
        },
        "alternative": {
            "step": -0.2,
            "goal": 10.0,
            "invalid_toggle": -2.0,
        },
    }

    # --------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------

    def __init__(
        self,
        setting=1,
        render_mode=None,
        reward_config="default",
    ):
        super().__init__()

        if setting not in self.MAPS:
            raise ValueError(
                "setting phải là 1, 2, 3 hoặc 4"
            )

        if reward_config not in self.REWARD_CONFIGS:
            raise ValueError(
                "reward_config phải là 'default' hoặc 'alternative'"
            )

        self.setting = setting
        self.render_mode = render_mode
        self.reward_config = reward_config

        # Reward values
        self.step_reward = self.REWARD_CONFIGS[
            reward_config
        ]["step"]

        self.goal_reward = self.REWARD_CONFIGS[
            reward_config
        ]["goal"]

        self.invalid_toggle_reward = self.REWARD_CONFIGS[
            reward_config
        ]["invalid_toggle"]

        # Load fixed map
        self.grid = [
            list(row)
            for row in self.MAPS[setting]
        ]

        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

        # Check map dimensions
        if any(
            len(row) != self.cols
            for row in self.grid
        ):
            raise ValueError(
                "Các hàng trong bản đồ phải có cùng độ dài"
            )

        # --------------------------------------------------
        # FIND IMPORTANT POSITIONS
        # --------------------------------------------------

        self.start_pos = None
        self.goal_pos = None

        self.switch_positions = {}
        self.gate_positions = {}

        for r in range(self.rows):
            for c in range(self.cols):

                cell = self.grid[r][c]

                if cell == "S":
                    self.start_pos = (r, c)

                elif cell == "X":
                    self.goal_pos = (r, c)

                elif cell in ("A", "B"):
                    self.switch_positions[cell] = (
                        r,
                        c,
                    )

                elif cell in ("G", "H"):
                    self.gate_positions.setdefault(
                        cell,
                        [],
                    ).append((r, c))

        if self.start_pos is None:
            raise ValueError(
                "Bản đồ phải có điểm bắt đầu S"
            )

        if self.goal_pos is None:
            raise ValueError(
                "Bản đồ phải có điểm đích X"
            )

        # --------------------------------------------------
        # SWITCH INFORMATION
        # --------------------------------------------------

        self.switch_names = sorted(
            self.switch_positions.keys()
        )

        self.switch_bits = {
            name: 1 << index
            for index, name in enumerate(
                self.switch_names
            )
        }

        self.num_switches = len(
            self.switch_names
        )

        self.num_masks = 2 ** self.num_switches

        # --------------------------------------------------
        # GATE CONTROL
        # --------------------------------------------------

        if setting == 1:
            # A -> G
            self.gate_controller = {
                "G": "A"
            }

        elif setting == 2:
            # A -> G
            # B -> H
            self.gate_controller = {
                "G": "A",
                "H": "B",
            }

        elif setting == 3:
            # A -> G and H
            self.gate_controller = {
                "G": "A",
                "H": "A",
            }

        else:
            # A -> G
            self.gate_controller = {
                "G": "A"
            }

        # --------------------------------------------------
        # GYM SPACES
        # --------------------------------------------------

        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.MultiDiscrete(
            [
                self.rows,
                self.cols,
                self.num_masks,
            ]
        )

        # --------------------------------------------------
        # STATE
        # --------------------------------------------------

        self.state = None
        self.steps = 0

        # Setting 4:
        # A can only be activated once.
        self.special_switch_used = False

    # --------------------------------------------------
    # RESET
    # --------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.special_switch_used = False

        self.state = (
            self.start_pos[0],
            self.start_pos[1],
            0,
        )

        observation = np.array(
            self.state,
            dtype=np.int64,
        )

        info = {
            "position": self.start_pos,
            "mask": 0,
            "setting": self.setting,
            "reward_config": self.reward_config,
        }

        return observation, info

    # --------------------------------------------------
    # CHECK GATE
    # --------------------------------------------------

    def is_gate_open(self, position, mask=None):

        if mask is None:

            if self.state is None:
                mask = 0
            else:
                mask = int(self.state[2])

        r, c = position
        cell = self.grid[r][c]

        # Not a gate
        if cell not in ("G", "H"):
            return True

        # Find which switch controls this gate
        switch_name = self.gate_controller.get(
            cell
        )

        if switch_name is None:
            return True

        bit = self.switch_bits.get(
            switch_name,
            0,
        )

        return bool(mask & bit)

    # --------------------------------------------------
    # CHECK VALID POSITION
    # --------------------------------------------------

    def _is_valid_position(
        self,
        row,
        col,
        mask,
    ):

        if not (
            0 <= row < self.rows
            and 0 <= col < self.cols
        ):
            return False

        cell = self.grid[row][col]

        # Wall
        if cell == "#":
            return False

        # Closed gate
        if cell in ("G", "H"):
            if not self.is_gate_open(
                (row, col),
                mask,
            ):
                return False

        return True

    # --------------------------------------------------
    # STEP
    # --------------------------------------------------

    def step(self, action):

        if self.state is None:
            raise RuntimeError(
                "Hãy gọi reset() trước khi gọi step()"
            )

        if not self.action_space.contains(
            action
        ):
            raise ValueError(
                "action phải nằm trong khoảng 0 đến 4"
            )

        row, col, mask = map(
            int,
            self.state,
        )

        reward = self.step_reward
        terminated = False
        truncated = False

        # ==================================================
        # ACTION 4: TOGGLE SWITCH
        # ==================================================

        if action == 4:

            cell = self.grid[row][col]

            # Player is standing on a switch
            if cell in self.switch_positions:

                bit = self.switch_bits[cell]

                # ------------------------------------------
                # SETTING 4
                # A can only be activated once
                # ------------------------------------------

                if (
                    self.setting == 4
                    and cell == "A"
                ):

                    if self.special_switch_used:

                        reward = (
                            self.invalid_toggle_reward
                        )

                    else:

                        # Activate A
                        mask ^= bit

                        self.special_switch_used = True

                        reward = -0.5

                # ------------------------------------------
                # NORMAL SWITCH
                # ------------------------------------------

                else:

                    mask ^= bit

                    reward = -0.5

            # Toggle pressed somewhere else
            else:

                reward = (
                    self.invalid_toggle_reward
                )

        # ==================================================
        # MOVEMENT ACTIONS
        # ==================================================

        else:

            moves = {
                0: (-1, 0),  # Up
                1: (1, 0),   # Down
                2: (0, -1),  # Left
                3: (0, 1),   # Right
            }

            dr, dc = moves[action]

            new_row = row + dr
            new_col = col + dc

            # Check whether movement is valid
            if self._is_valid_position(
                new_row,
                new_col,
                mask,
            ):

                row = new_row
                col = new_col

        # ==================================================
        # GOAL CHECK
        # ==================================================

        if (row, col) == self.goal_pos:

            reward = self.goal_reward
            terminated = True

        # ==================================================
        # UPDATE STATE
        # ==================================================

        self.steps += 1

        if (
            self.steps >= 150
            and not terminated
        ):
            truncated = True

        self.state = (
            row,
            col,
            mask,
        )

        observation = np.array(
            self.state,
            dtype=np.int64,
        )

        info = {
            "position": (row, col),
            "mask": mask,
            "goal_reached": terminated,
            "setting": self.setting,
            "reward_config": self.reward_config,
            "steps": self.steps,
        }

        # Optional human rendering
        if self.render_mode == "human":
            self.render()

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    # --------------------------------------------------
    # GET ALL STATES
    # --------------------------------------------------

    def get_all_states(self):

        states = []

        for row in range(self.rows):

            for col in range(self.cols):

                # Wall is not a state
                if self.grid[row][col] == "#":
                    continue

                for mask in range(
                    self.num_masks
                ):

                    states.append(
                        (
                            row,
                            col,
                            mask,
                        )
                    )

        return states

    # --------------------------------------------------
    # GET TRANSITION
    # --------------------------------------------------

    def get_transition(
        self,
        state,
        action,
    ):
        """
        Dùng cho Policy Iteration.

        Lấy trạng thái kế tiếp mà không
        thay đổi state thật của môi trường.
        """

        old_state = self.state
        old_steps = self.steps
        old_special_used = (
            self.special_switch_used
        )

        try:

            self.state = tuple(
                map(int, state)
            )

            next_state, reward, terminated, truncated, info = self.step(
                action
            )

            return (
                next_state,
                reward,
                terminated,
                truncated,
                info,
            )

        finally:

            self.state = old_state
            self.steps = old_steps
            self.special_switch_used = (
                old_special_used
            )

    # --------------------------------------------------
    # RENDER
    # --------------------------------------------------

    def render(self):

        if self.state is None:
            print(
                "Hãy gọi reset() trước."
            )
            return

        row, col, mask = map(
            int,
            self.state,
        )

        print()
        print(
            f"===== SETTING {self.setting} ====="
        )

        print(
            f"Position: ({row}, {col})"
        )

        print(
            f"Switch mask: {mask}"
        )

        print(
            f"Reward config: {self.reward_config}"
        )

        print(
            "Gate control:",
            self.gate_controller,
        )

        print()

        for r in range(self.rows):

            line = []

            for c in range(self.cols):

                cell = self.grid[r][c]

                # Player
                if (r, c) == (row, col):

                    line.append("P")

                # Gate
                elif cell in ("G", "H"):

                    if self.is_gate_open(
                        (r, c),
                        mask,
                    ):
                        line.append(cell)
                    else:
                        line.append(
                            cell.lower()
                        )

                else:

                    line.append(cell)

            print(" ".join(line))

        print()

    # --------------------------------------------------
    # CLOSE
    # --------------------------------------------------

    def close(self):
        pass


# ======================================================
# QUICK TEST
# ======================================================

if __name__ == "__main__":

    for setting in range(1, 5):

        env = SwitchGateEnv(
            setting=setting
        )

        observation, info = env.reset(
            seed=42
        )

        print()
        print(
            f"========== SETTING {setting} =========="
        )

        print(
            "Observation:",
            observation,
        )

        env.render()

        env.close()