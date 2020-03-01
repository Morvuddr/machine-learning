from enum import Enum

class ReductionType(Enum):
    naive = "naive"
    one_hot = "one_hot"

class DistanceFuncType(Enum):
    MANHATTAN = "manhattan"
    EUCLIDEAN = "euclidean"
    CHEBYSHEV = "chebyshev"


class KernelFuncType(Enum):
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    EPANECHNIKOV = "epanechnikov"
    QUARTIC = "quartic"
    TRIWEIGHT = "triweight"
    TRICUBE = "tricube"
    GAUSSIAN = "gaussian"
    COSINE = "cosine"
    LOGISTIC = "logistic"
    SIGMOID = "sigmoid"


class WindowType(Enum):
    FIXED = "fixed"
    VARIABLE = "variable"