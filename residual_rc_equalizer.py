"""线性主体加残差RC修正：参考信号只在训练前缀生成监督目标。"""
import numpy as np
from fusion_prefix_backend import fast_linear,fast_rc
from linear_rc_equalizer import _linear_rc_equalize
from rclzl_w2s import rc_wave_equalize

def residual_rc_equalize(waveform,target,*,linear_train_samples,residual_train_samples,
                         linear_params,nonlinear_params=None,native=False,verbose=True):
    """先冻结线性主体，再拟合理想参考减线性输出，最后直接相加。"""
    waveform=np.asarray(waveform).ravel()
    target=np.asarray(target).ravel()
    # 只需训练参考前缀，推理波形可以比参考更长，不要求提供测试真值。
    count=len(waveform)
    if not 0<linear_train_samples<=residual_train_samples<=count:
        raise ValueError("两级训练边界必须位于有效输入内，残差训练边界不早于线性训练边界")
    if len(target)<residual_train_samples:
        raise ValueError("参考至少需要覆盖声明的训练前缀，不需要测试参考")
    # 级联居中窗口叠加：线性输出自身有前视，残差拟合边界必须相应退让。
    lookahead=linear_params["taps"]//2
    residual_fit_samples=residual_train_samples-lookahead
    if residual_fit_samples<=linear_params["taps"]:
        raise ValueError("保护后残差训练区过短")
    nonlinear_params=dict(linear_params if nonlinear_params is None else nonlinear_params)
    # 不向线性拟合器传递测试参考，即使外部误传了测试标签也不会使用。
    linear_teacher=np.zeros(count)
    linear_teacher[:linear_train_samples]=target[:linear_train_samples].real
    if native:
        linear=_linear_rc_equalize(waveform,linear_teacher,train_samples=linear_train_samples,
                                   **linear_params)[0].real
    else:
        linear=fast_linear(waveform,linear_teacher,linear_params,linear_train_samples).real
    # 修正目标的符号必须为参考减预测；预测减参考则最终需要减去残差。
    correction_teacher=np.zeros(count)
    correction_teacher[:residual_fit_samples]=(
        target[:residual_fit_samples].real-linear[:residual_fit_samples])
    scale=float(np.sqrt(np.mean(correction_teacher[:residual_fit_samples]**2)))
    # 真正级联：非线性RC输入只来自冻结的线性输出，不输入理想参考或真实残差。
    if scale<=1e-14:
        correction=np.zeros(count)
    elif native:
        correction=rc_wave_equalize(linear,correction_teacher,
            train_samples=residual_fit_samples,**nonlinear_params)[0]
    else:
        correction=fast_rc(linear,correction_teacher,nonlinear_params,residual_fit_samples).real
    recovered=linear+correction
    if not np.isfinite(recovered).all():
        raise FloatingPointError("残差级联输出包含非有限数值")
    # 这里只统计训练区误差，不从测试误差调整修正强度。
    train=slice(0,residual_fit_samples)
    metadata={"linear_train_samples":linear_train_samples,
        "residual_train_samples":residual_train_samples,
        "residual_fit_samples":residual_fit_samples,"cascade_lookahead_guard":lookahead,
        "target_definition":"reference_minus_linear","output_definition":"linear_plus_correction",
        "nonlinear_input":"linear_output_only","correction_gain":1.,
        "train_linear_mse":float(np.mean(correction_teacher[train]**2)),
        "train_corrected_mse":float(np.mean((correction_teacher[train]-correction[train])**2))}
    if verbose:
        print(f"残差级联：输入{waveform.shape}→线性{linear.shape}→残差RC{correction.shape}→相加{recovered.shape}",flush=True)
    return recovered,linear,correction,metadata
