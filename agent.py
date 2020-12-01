import numpy as np

action2name = {
    0: "N",
    1: "W",
    2: "S",
    3: "E",
}

name2action = {v: k for k, v in action2name.items()}

action2delta = {
    0: (0, 1),  # x,y format
    1: (-1, 0),
    2: (0, -1),
    3: (1, 0),
}

delta2action = {v: k for k, v in action2delta.items()}


class Agent:
    def __init__(
        self,
        init_position=[0, 0],
        env_shape=(100, 100),
        x_bounds=range(2, 97),
        y_bounds=range(2, 97),
    ):

        self._env_shape = env_shape
        self.position = np.array(init_position)
        self._x_bounds = x_bounds
        self._y_bounds = y_bounds
        self._actions, self._traj = self.generate_trajectory()
        self._found_fire = False
        self._view = None

    def random_policy(self):
        Q = np.random.rand(4)
        return Q

    def nn_policy(self):
        return self.random_policy()

    def action(self):
        """
        sample and return action
        """
        # until we find fire, sample from trajectory
        if not self._found_fire:
            action = self._actions.pop(0)

        # sample from policy
        else:
            Q = self.random_policy()
            viable = self.viable_actions()
            action = np.argmax(Q[viable])

        delta = action2delta[action]
        self.position += delta

        return action

    @property
    def observation(self):
        return self._view

    @observation.setter
    def observation(self, value):
        self._view = value
        # bool addition

    #         self._found_fire = or

    def viable_actions(self):
        """
        returns a mask
        """

        viable = []
        x, y = self.position

        # check North
        if y + 1 in self._y_bounds:
            viable.append("N")

        # check South
        if y - 1 in self._y_bounds:
            viable.append("S")

        # check West
        if x - 1 in self._x_bounds:
            viable.append("W")

        # check East
        if x + 1 in self._x_bounds:
            viable.append("E")

        # convert to indices
        viable = [name2action[v] for v in viable]

        is_viable = np.zeros(4, dtype=np.bool)
        is_viable[viable] = True

        return is_viable

    def reset(position=[0, 0]):
        self.position = np.array(position)

    def extinguish(self, env, range_xy=(3, 3)):
        x, y = self.position
        dx, dy = range_xy
        env._state_map[y - dy : y + dy, x - dx : x + dx] = 0

    def observe(self, env, fov=(3, 3)):
        view = env._state_map
        self._view = env._state_map[y - dy : y + dy, x - dx : x + dx]

    def generate_trajectory(self, step_width=10):

        actions = []
        traj = []

        for x_step in self._x_bounds[::10]:

            midway = x_step + int(step_width / 2)

            # go north
            for y in self._y_bounds:
                traj.append((x_step, y))
                actions.append(name2action["N"])

            # go east
            for x in range(x_step, midway):
                traj.append((x, y))
                actions.append(name2action["E"])

            # go down
            for y in reversed(self._y_bounds):
                traj.append((midway, y))
                actions.append(name2action["S"])

            # go west
            for x in range(midway, x_step + step_width):
                traj.append((x, y))
                actions.append(name2action["E"])

        return actions, traj
