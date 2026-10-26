"""一维时序双分支融合，只在指定训练区拟合参数。"""
from dataclasses import dataclass
import numpy as np

@dataclass
class FusionParameters:
    # 参考论文的分支拼接及通道注意力，在此对应输出投影或互补标量权重。
    mode: str
    coefficient: np.ndarray
    bias: np.ndarray
    is_complex: bool
    fit_start: int
    fit_stop: int

def _vector(values):
    x = np.asarray(values)
    if x.ndim == 1:
        return x
    if x.ndim == 2 and 1 in x.shape:
        return x.ravel()
    raise ValueError("需要一维时序信号或单列向量")

def fusion_layer(linear, nonlinear, target=None, *, mode="projection",
                 train_slice=None, parameters=None, ridge=1e-6, verbose=True):
    """拟合或复用冻结融合参数；长度不同时截断到两路最短长度。"""
    linear, nonlinear = _vector(linear), _vector(nonlinear)
    count = min(len(linear), len(nonlinear))
    linear, nonlinear = linear[:count], nonlinear[:count]
    complex_signal = np.iscomplexobj(linear) or np.iscomplexobj(nonlinear)
    if complex_signal:
        features = np.column_stack((linear.real, linear.imag, nonlinear.real, nonlinear.imag))
    else:
        features = np.column_stack((linear, nonlinear))
    if parameters is None:
        if target is None or train_slice is None:
            raise ValueError("拟合融合参数必须显式提供训练目标和训练切片")
        desired = _vector(target)
        start, stop = train_slice
        if not 0 <= start < stop <= min(count,len(desired)):
            raise ValueError("训练区必须完整包含在有效对齐区间内")
        # 只读取训练区目标，不用测试目标选择模式、系数或正则强度。
        if mode == "weighted_sum":
            delta = linear[start:stop] - nonlinear[start:stop]
            residual = desired[start:stop] - nonlinear[start:stop]
            power = float(np.vdot(delta,delta).real)
            weight = .5 if power < 1e-20 else float(np.clip(np.vdot(delta,residual).real/power,0,1))
            coefficient = np.array([weight,1-weight])
            bias = np.zeros(1)
        elif mode == "projection":
            x = features[start:stop]
            y = (np.column_stack((desired[start:stop].real,desired[start:stop].imag))
                 if complex_signal else desired[start:stop].real[:,None])
            xm, ym = x.mean(axis=0), y.mean(axis=0)
            xc, yc = x-xm, y-ym
            penalty = ridge*max(float(np.trace(xc.T@xc))/x.shape[1],1e-12)
            coefficient = np.linalg.solve(xc.T@xc+penalty*np.eye(x.shape[1]),xc.T@yc)
            bias = ym-xm@coefficient
        else:
            raise ValueError("融合模式只能为projection或weighted_sum")
        parameters = FusionParameters(mode,coefficient,bias,complex_signal,start,stop)
    if complex_signal != parameters.is_complex:
        raise ValueError("拟合与推理必须采用相同实数或复数语义")
    if parameters.mode == "weighted_sum":
        result = parameters.coefficient[0]*linear+parameters.coefficient[1]*nonlinear
    else:
        projected = features@parameters.coefficient+parameters.bias
        result = projected[:,0]+1j*projected[:,1] if complex_signal else projected[:,0]
    if not np.isfinite(result).all():
        raise FloatingPointError("融合输出出现非有限数值")
    if verbose:
        print(f"融合：线性{linear.shape}，非线性{nonlinear.shape}，拼接{features.shape}，输出{result.shape}，模式={parameters.mode}",flush=True)
    return result,parameters
