import numpy as np
import torch.nn
from videoio import VideoWriter

action2name = {
    0: "N",
    1: "W",
    2: "S",
    3: "E",
}

name2action = {v: k for k, v in action2name.items()}

action2delta = {
    # NN output to (y,x) format
    0: (0, 1),
    1: (-1, 0),
    2: (0, -1),
    3: (1, 0),
}

delta2action = {v: k for k, v in action2delta.items()}


class Agent:
    def __init__(
        self,
        init_position=[50, 50],
        env_shape=(100, 100),
        fov_shape=(7, 7),
        fov_extinguish=(5, 5),
        lr=1e-6,
    ):

        self._env_shape = env_shape
        self.position = np.array(init_position)
        self._x_bounds = range(fov_shape[1] + 1, env_shape[1] - fov_shape[0] - 1)
        self._y_bounds = range(fov_shape[0] + 1, env_shape[0] - fov_shape[1] - 1)
        self._actions, self._traj = self.generate_trajectory()
        self._fov_shape = fov_shape
        self._fov_extinguish = fov_extinguish
        self._view = np.zeros(fov_shape)

        self._policy = torch.nn.Sequential(
            torch.nn.Linear(
                fov_shape[0] * fov_shape[1], 2048
            ),  # input: (the 7x7 segment view)
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048, 4),  # output: (the values of N, S, W, E actions)
            torch.nn.Sigmoid(),
        )

        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr)

    def random_policy(self):
        Q = np.random.rand(4)
        return Q

    def action(self, epsilon):
        """
        sample and return action using epsilon-greedy approach
        epsilon is the fraction of random actions
        """
        # sample randomly
        Q_rand = self.random_policy()

        # sample form policy
        view = self._view.flatten()  # e.g. 5x5 view of environment state
        self._tensor_in = torch.tensor(view, dtype=torch.float32)
        self._tensor_out = self._policy(self._tensor_in)
        Q_policy = (
            self._tensor_out.detach().numpy()
        )  # Q function values approximated by

        # epsilon-greedy
        Q = Q_rand if np.random.rand() < epsilon else Q_policy

        # assign -inf reward to actions that take us to infeasible states
        viable = self.viable_actions()
        Q[~viable] = -np.inf
        action = np.argmax(Q)

        # update position based on chosen action
        delta = action2delta[action]
        self.position += delta

        return action

    @property
    def observation(self):
        return self._view

    @observation.setter
    def observation(self, value):
        self._view = value

    def backprop(self, action, reward):
        self._optimizer.zero_grad()

        tensor_reward = torch.zeros(4, dtype=torch.float32)
        tensor_reward[action] = reward
        self._tensor_out.backward(tensor_reward)

        self._optimizer.step()

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

    @staticmethod
    def save_video(filename, frame_buffer):
        writer = VideoWriter(filename, resolution=(500, 500), fps=60)
        for frame in frame_buffer:
            writer.write(np.array(frame))
        writer.close()
