"""孤立森林 (Isolation Forest) 异常检测模块。

基于 sklearn.ensemble.IsolationForest 实现，用于从数据集中筛选或剔除离群点（Outliers）。
"""

from typing import List, Literal
import numpy as np
from numpy.typing import ArrayLike
from sklearn.ensemble import IsolationForest


class OutlierDetector:
    """基于孤立森林的异常检测器。

    使用 sklearn 的 IsolationForest 算法识别数据集中的异常点。
    适用于高维数据的异常检测，如原子环境描述符或结构描述符。

    Example:
        >>> detector = OutlierDetector(contamination=0.05)
        >>> data = np.random.randn(100, 30)
        >>> # 返回正常样本的索引
        >>> valid_indices = detector.detect(data, return_type='inliers')
        >>> # 或者返回布尔掩码
        >>> mask = detector.detect(data, return_type='mask')
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | Literal["auto"] = "auto",
        max_samples: float | int | Literal["auto"] = "auto",
        random_state: int | None = 42,
        **kwargs,
    ) -> None:
        """初始化孤立森林检测器。

        Args:
            n_estimators: 森林中树的数量。默认 100。
            contamination: 预期的异常值比例。
                如果是 'auto'，算法会自己决定。
                如果是浮点数 (0, 0.5]，表示数据中异常点的比例。
            max_samples: 每棵树抽样的样本数量。
            random_state: 随机种子，保证结果可复现。默认 42。
            **kwargs: 传递给 sklearn.ensemble.IsolationForest 的其他参数。
        """
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
            **kwargs,
        )
        self.fitted = False

    def detect(
        self,
        data: ArrayLike,
        return_type: Literal["indices", "mask", "outliers"] = "indices",
    ) -> List[int] | np.ndarray:
        """执行异常检测。

        Args:
            data: 输入数据，形状 (N, D)。
            return_type: 返回值类型。
                - 'indices': 返回被判定为 **正常 (Inliers)** 样本的索引列表。
                - 'outliers': 返回被判定为 **异常 (Outliers)** 样本的索引列表。
                - 'mask': 返回一个布尔数组，True 表示正常，False 表示异常。

        Returns:
            根据 return_type 返回相应结果。
        """
        data = np.asarray(data)

        pred = self.model.fit_predict(data)
        self.fitted = True

        if return_type == "mask":
            return pred == 1

        elif return_type == "indices":
            return np.where(pred == 1)[0].tolist()

        elif return_type == "outliers":
            return np.where(pred == -1)[0].tolist()

        else:
            raise ValueError(f"Unknown return_type: {return_type}")

    def score_samples(self, data: ArrayLike) -> np.ndarray:
        """获取样本的异常分数。

        分数越低，越可能是异常点。

        Args:
            data: 输入数据。

        Returns:
            异常分数数组。
        """
        if not self.fitted:
            # 如果之前没 fit 过，先 fit 一下
            self.model.fit(data)
            self.fitted = True

        return self.model.score_samples(data)
