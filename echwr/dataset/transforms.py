import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d

__all__ = ['AddNoise', 'Drift', 'Dropout', 'TimeWarp']


class AddNoise:
    '''Add random noise to time series.

    The noise added to every time point of a time series is independent and
    identically distributed.

    Args:
        loc: Mean of the random noise. If float, all noise values are sampled
            with the same mean. If tuple/list, the mean is sampled randomly
            for each series/channel. Defaults to 0.
        scale: Standard deviation of the random noise. If float, used for all.
            If tuple/list, sampled randomly per series/channel. Defaults to
            0.1.
        distr: Distribution of the random noise. Must be one of 'gaussian',
            'laplace', or 'uniform'. Defaults to 'gaussian'.
        kind: How the noise is added. Must be 'additive' or 'multiplicative'.
            Defaults to 'additive'.
        per_channel: Whether to sample independent noise values for each
            channel. Defaults to True.

    Attributes:
        loc (float | tuple | list): Configured mean parameter.
        scale (float | tuple | list): Configured scale parameter.
        kind (str): Mode of noise addition.
        per_channel (bool): Flag for channel independence.
        gen_noise (callable): The noise generation function.
    '''

    def __init__(
        self,
        loc: float | tuple[float, float] | list[float] = 0,
        scale: float | tuple[float, float] | list[float] = 0.1,
        distr: str = 'gaussian',
        kind: str = 'additive',
        per_channel: bool = True,
    ) -> None:
        self.loc = loc
        self.scale = scale
        self.kind = kind
        self.per_channel = per_channel

        # set up noise generator
        if distr == 'gaussian':
            self.gen_noise = lambda size: np.random.normal(0.0, 1.0, size)
        elif distr == 'laplace':
            self.gen_noise = lambda size: np.random.laplace(0.0, 1.0, size)
        elif distr == 'uniform':
            self.gen_noise = lambda size: np.random.uniform(0.0, 1.0, size)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''Applies noise to the input sequence.

        Args:
            x: Input sequence. Shape: (len_seq, num_chan).

        Returns:
            Processed output sequence. Shape: (len_seq, num_chan).
        '''
        L, C = x.shape

        if isinstance(self.loc, (float, int)):
            loc = self.loc
        elif isinstance(self.loc, tuple):
            loc = np.random.uniform(low=self.loc[0], high=self.loc[1])
        elif isinstance(self.loc, list):
            loc = np.random.choice(self.loc)

        if isinstance(self.scale, (float, int)):
            scale = self.scale
        elif isinstance(self.scale, tuple):
            scale = np.random.uniform(low=self.scale[0], high=self.scale[1])
        elif isinstance(self.scale, list):
            scale = np.random.choice(self.scale)

        if self.per_channel:
            noise = self.gen_noise((L, C))
        else:
            noise = self.gen_noise((L, 1))
            noise = np.repeat(noise, C, axis=1)

        noise = noise * scale + loc

        if self.kind == 'additive':
            x = x + noise
        else:
            x = x * (1.0 + noise)

        return x


class Drift:
    '''Drift the value of time series.

    The augmenter drifts the value of time series from its original values
    randomly and smoothly. The extent of drifting is controlled by the maximal
    drift and the number of drift points.

    Args:
        max_drift: The maximal amount of drift added. If float, fixed for all.
            If tuple, sampled from interval. Defaults to 0.5.
        n_drift_points: The number of time points a new drifting trend is
            defined. If int, fixed. If list, sampled randomly. Defaults to 3.
        kind: 'additive' or 'multiplicative'. Defaults to 'additive'.
        per_channel: Whether to sample independent drifts for each channel.
            Defaults to True.

    Attributes:
        max_drift (float | tuple): Configured drift magnitude.
        kind (str): Mode of drift addition.
        per_channel (bool): Flag for channel independence.
        n_drift_points (set): Set of possible drift point counts.
    '''

    def __init__(
        self,
        max_drift: float | tuple[float, float] = 0.5,
        n_drift_points: int | list[int] = 3,
        kind: str = 'additive',
        per_channel: bool = True,
    ) -> None:
        self.max_drift = max_drift
        self.kind = kind
        self.per_channel = per_channel

        if isinstance(n_drift_points, int):
            self.n_drift_points = set([n_drift_points])
        else:
            self.n_drift_points = set(n_drift_points)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''Applies drift to the input sequence.

        Args:
            x: Input sequence. Shape: (len_seq, num_chan).

        Returns:
            Processed output sequence. Shape: (len_seq, num_chan).
        '''
        L, C = x.shape
        ind = np.random.choice(
            len(self.n_drift_points), C if self.per_channel else 1
        )  # map series to n_drift_points
        drift = np.zeros((C if self.per_channel else 1, L))

        for i, n in enumerate(self.n_drift_points):
            if not (ind == i).any():
                continue

            anchors = np.cumsum(
                np.random.normal(size=((ind == i).sum(), n + 2)), axis=1
            )
            interpFuncs = CubicSpline(
                np.linspace(0, L, n + 2), anchors, axis=1
            )
            drift[ind == i, :] = interpFuncs(np.arange(L))

        drift = drift.reshape((-1, L)).swapaxes(0, 1)
        drift = drift - drift[0, :].reshape(1, -1)
        drift = drift / (abs(drift).max(axis=1, keepdims=True) + 1e-6)

        if isinstance(self.max_drift, (float, int)):
            drift = drift * self.max_drift
        else:
            drift = drift * np.random.uniform(
                low=self.max_drift[0],
                high=self.max_drift[1],
                size=(1, C if self.per_channel else 1),
            )

        if self.kind == 'additive':
            x = x + drift
        else:
            x = x * (1 + drift)

        return x


class Dropout:
    '''Dropout values of some random time points in time series.

    Single time points or sub-sequences could be dropped out.

    Args:
        p: Probability of a time point being dropped out. Defaults to 0.05.
        size: Size of dropped out units. If int, fixed size. If tuple/list,
            sampled randomly. Defaults to 1.
        fill: Filling strategy ('ffill', 'bfill', 'mean') or a fixed float
            value. Defaults to 'ffill'.
        per_channel: Whether to sample independent dropout masks for each
            channel. Defaults to False.

    Attributes:
        p (float | tuple | list): Dropout probability configuration.
        fill (str | float): Fill strategy configuration.
        per_channel (bool): Flag for channel independence.
        size (list): List of possible dropout block sizes.
    '''

    def __init__(
        self,
        p: float | tuple[float, float] | list[float] = 0.05,
        size: int | tuple[int, int] | list[int] = 1,
        fill: str | float = 'ffill',
        per_channel: bool = False,
    ) -> None:
        self.p = p
        self.fill = fill
        self.per_channel = per_channel

        if isinstance(size, int):
            self.size = [size]
        elif isinstance(size, tuple):
            self.size = list(range(size[0], size[1]))
        elif isinstance(size, list):
            self.size = size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''Applies dropout to the input sequence.

        Args:
            x: Input sequence. Shape: (len_seq, num_chan).

        Returns:
            Processed output sequence. Shape: (len_seq, num_chan).
        '''
        L, C = x.shape

        if isinstance(self.p, (float, int)):
            p = np.ones(C if self.per_channel else 1) * self.p
        elif isinstance(self.p, tuple):
            p = np.random.uniform(p[0], p[1], C if self.per_channel else 1)
        elif isinstance(self.p, list):
            p = np.random.choice(self.p, C if self.per_channel else 1)

        x = x.swapaxes(0, 1)

        if isinstance(self.fill, str) and (self.fill == 'mean'):
            fill_value = x.mean(axis=0)

        for s in self.size:
            # sample dropout blocks
            if self.per_channel:
                drop = (
                    np.random.uniform(size=(C, L - s))
                    <= p.reshape(-1, 1) / len(self.size) / s
                )
            else:
                drop = (
                    np.random.uniform(size=(L - s))
                    <= p.reshape(-1, 1) / len(self.size) / s
                )
                drop = np.repeat(drop, C, axis=0)

            ind = np.argwhere(drop)  # position of dropout blocks

            if ind.size > 0:
                if isinstance(self.fill, str) and (self.fill == 'ffill'):
                    i = np.repeat(ind[:, 0], s)
                    j0 = np.repeat(ind[:, 1], s)
                    j1 = j0 + np.tile(np.arange(1, s + 1), len(ind))
                    # clip index to avoid out of bounds
                    j1 = np.clip(j1, 0, L - 1)
                    x[i, j1] = x[i, j0]
                elif isinstance(self.fill, str) and (self.fill == 'bfill'):
                    i = np.repeat(ind[:, 0], s)
                    j0 = np.repeat(ind[:, 1], s) + s
                    j1 = j0 - np.tile(np.arange(1, s + 1), len(ind))
                    j0 = np.clip(j0, 0, L - 1)
                    x[i, j1] = x[i, j0]
                elif isinstance(self.fill, str) and (self.fill == 'mean'):
                    i = np.repeat(ind[:, 0], s)
                    j = np.repeat(ind[:, 1], s) + np.tile(
                        np.arange(1, s + 1), len(ind)
                    )
                    j = np.clip(j, 0, L - 1)
                    x[i, j] = fill_value[i]
                elif isinstance(self.fill, (float, int)):
                    i = np.repeat(ind[:, 0], s)
                    j = np.repeat(ind[:, 1], s) + np.tile(
                        np.arange(1, s + 1), len(ind)
                    )
                    j = np.clip(j, 0, L - 1)
                    x[i, j] = self.fill

        x = x.reshape(C, L).T

        return x


class TimeWarp:
    '''Random time warping.

    The augmenter randomly changes the speed of the timeline. The time warping
    is controlled by the number of speed changes and the maximal ratio of
    max/min speed.

    Args:
        n_speed_change: The number of speed changes in each series. Defaults
            to 3.
        max_speed_ratio: The maximal ratio of max/min speed in the warped time
            line. Higher values mean significant warping. Defaults to 3.0.

    Attributes:
        n_speed_change (int): Number of speed change points.
        max_speed_ratio (float | tuple | list): Configured max speed ratio.
    '''

    def __init__(
        self,
        n_speed_change: int = 3,
        max_speed_ratio: float | tuple[float, float] | list[float] = 3.0,
    ) -> None:
        self.n_speed_change = n_speed_change
        self.max_speed_ratio = max_speed_ratio

    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''Applies time warping to the input sequence.

        Args:
            x: Input sequence. Shape: (len_seq, num_chan).

        Returns:
            Processed output sequence. Shape: (len_seq, num_chan).
        '''
        L, _ = x.shape

        idx = np.arange(L)
        anchors = np.arange(
            0,
            1 + 1 / (self.n_speed_change + 1) / 2,
            1 / (self.n_speed_change + 1),
        ) * (L - 1)

        if isinstance(self.max_speed_ratio, (float, int)):
            max_speed_ratio = float(self.max_speed_ratio)
        elif isinstance(self.max_speed_ratio, tuple):
            max_speed_ratio = np.random.uniform(
                low=self.max_speed_ratio[0], high=self.max_speed_ratio[1]
            )
        elif isinstance(self.max_speed_ratio, list):
            max_speed_ratio = np.random.choice(self.max_speed_ratio)

        # generate random speeds at anchors
        anchor_values = np.random.uniform(
            low=0.0, high=1.0, size=self.n_speed_change + 1
        )
        
        # normalize anchor values to respect the max_speed_ratio constraint
        # note: simplified logic to prevent division by zero
        if max_speed_ratio > 1.0 + 1e-8:
            denom = (anchor_values.max() - anchor_values.min())
            
            if denom < 1e-8:
                 # edge case: all random values are identical
                 anchor_values = anchor_values * 0 + 1.0
            else:
                anchor_values = (anchor_values - anchor_values.min()) / denom
                anchor_values = anchor_values * (max_speed_ratio - 1.0) + 1.0
        else:
             anchor_values = anchor_values * 0 + 1.0

        # convert speed to cumulative distance (time map)
        anchor_values = anchor_values.cumsum() 
        anchor_values = anchor_values / anchor_values[-1] * (L - 1)
        anchor_values = np.insert(anchor_values, 0, 0)

        warp = PchipInterpolator(x=anchors, y=anchor_values, axis=0)(idx)

        x = interp1d(
            idx, x, axis=0, fill_value='extrapolate', assume_sorted=True
        )(warp)

        return x