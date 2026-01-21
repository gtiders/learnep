"""最远点采样 (Farthest Point Sampling) 算法实现。"""

from typing import Callable
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist


class FarthestPointSampler:
    """最远点采样器。

    使用贪心策略迭代选取距离已选集合最远的点，直到满足停止条件。
    典型应用场景：从大量候选结构描述符中筛选出最具代表性的子集。

    Example:
        >>> data = np.random.randn(1000, 30)  # 1000个候选点，每个30维
        >>> sampler = FarthestPointSampler(min_distance=0.1)
        >>> indices = sampler.select(data)
    """

    def __init__(
        self,
        min_distance: float = 0.1,
        metric: str | Callable = "euclidean",
        **metric_kwargs,
    ) -> None:
        """初始化采样器。

        Args:
            min_distance: 停止阈值。当最远候选点的距离低于此值时停止采样。
            metric: 距离度量方式，支持 scipy.spatial.distance.cdist 的所有度量,
                如 'euclidean', 'cosine', 'minkowski' 等。
            **metric_kwargs: 传递给度量函数的额外参数。
        """
        self.min_distance = min_distance
        self.metric = metric
        self.metric_kwargs = metric_kwargs

    def select(
        self,
        candidates: ArrayLike,
        selected: ArrayLike | None = None,
        *,
        min_distance: float | None = None,
        min_select: int = 1,
        max_select: int | None = None,
    ) -> list[int]:
        """从候选集中选取距离已选集合最远的点。

        Args:
            candidates: 候选点集，形状 (N, D)，N 为点数，D 为特征维度。
            selected: 已选点集，形状 (M, D)。若为 None 则从空集开始。
            min_distance: 本次采样的停止阈值，覆盖实例默认值。
            min_select: 最少选取数量，即使距离已低于阈值也会继续选。
            max_select: 最多选取数量。

        Returns:
            被选中点在 candidates 中的索引列表。
        """
        candidates = np.asarray(candidates)
        n_candidates = len(candidates)

        if n_candidates == 0:
            return []

        min_distance = min_distance if min_distance is not None else self.min_distance
        max_select = max_select if max_select is not None else n_candidates

        result: list[int] = []

        # 初始化已选集合
        if selected is None or len(selected) == 0:
            result.append(0)
            current_selected = candidates[[0]]
        else:
            current_selected = np.asarray(selected)

        # 计算每个候选点到已选集合的最短距离
        distances = self._min_distances(candidates, current_selected)

        # 贪心迭代：每次选最远的点
        while len(result) < max_select:
            if distances.max() <= min_distance and len(result) >= min_select:
                break

            idx = int(np.argmax(distances))
            result.append(idx)

            # 增量更新距离：只需比较到新点的距离
            new_distances = self._min_distances(candidates, candidates[[idx]])
            distances = np.minimum(distances, new_distances)

        return result

    def _min_distances(self, points: NDArray, reference: NDArray) -> NDArray:
        """计算 points 中每个点到 reference 集合的最短距离。"""
        dist_matrix = cdist(points, reference, metric=self.metric, **self.metric_kwargs)
        return dist_matrix.min(axis=1)
