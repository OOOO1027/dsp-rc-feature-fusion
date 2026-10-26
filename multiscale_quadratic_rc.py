"""不同时间尺度的独立储备池、各自完整二次读出及一维波形投影。"""
import numpy as np
from fusion_prefix_backend import linear_features,features_kernel

def build_features(waveform,target,params,train_samples,normalization=None):
    """短窗identity与长窗tanh均从同一原始波形取输入，互不串接。"""
    wave=np.asarray(waveform).reshape(-1)
    if np.any(wave.imag):raise ValueError("当前CAP波形前端要求实数输入")
    taps=int(params["taps"]);direct=int(params["passthrough_taps"])
    if min(taps,direct)<=0 or taps%2!=1 or direct%2!=1:raise ValueError("储备池和直通抽头须为正奇数")
    margin=max(taps,direct)//2
    if not 2*margin+1<train_samples<=len(wave):raise ValueError("有效输入或训练区不足，禁止补造参考")
    if normalization is None:
        if target is None or len(target)<train_samples:raise ValueError("参考必须覆盖拟合区")
        teacher=np.asarray(target)[:train_samples]
        if np.any(teacher.imag):raise ValueError("波形参考必须为实数")
        rms=float(np.sqrt(np.mean(abs(wave[:train_samples])**2)))
        scale=float(np.sqrt(np.mean(abs(teacher)**2)))
        if min(rms,scale)<1e-14:raise ValueError("训练RMS过小")
    else:rms=normalization["input_rms"];scale=normalization["target_rms"]
    x=wave.real/rms;half=taps//2
    window=np.lib.stride_tricks.sliding_window_view(np.pad(x,(half,half)),taps)
    input_scale=max(float(np.max(abs(window[:train_samples-half]))),1e-12) if normalization is None else normalization["input_window_scale"]
    inputs=np.ascontiguousarray(np.column_stack((window,np.zeros_like(window)))/input_scale)
    rng=np.random.default_rng(params["seed"])
    recurrent=rng.uniform(-1,1,(params["nodes"],params["nodes"]))
    recurrent*=params["spectral_radius"]/np.max(abs(np.linalg.eigvals(recurrent)))
    win=rng.uniform(-1,1,(params["nodes"],2*taps+1))
    gamma=params["gamma"]
    if not 0<gamma<=1:raise ValueError("gamma须位于(0,1]")
    activation=params["activation"]
    if activation=="identity":
        effective=(1-gamma)*np.eye(params["nodes"])+gamma*recurrent
        if np.max(abs(np.linalg.eigvals(effective)))>=1:raise ValueError("线性递归不稳定，拒绝训练")
        state_features=linear_features(inputs,recurrent,win,gamma)
    elif activation=="tanh":state_features=features_kernel(inputs,recurrent,win,gamma)
    else:raise ValueError("激活只能是identity或tanh")
    # 直通窗口与递归输入窗口独立，保留既有CH1=25、CH3=97线性ISI能力。
    dh=direct//2
    direct_window=np.lib.stride_tricks.sliding_window_view(np.pad(x,(dh,dh)),direct)/input_scale
    base=np.column_stack((state_features[:,:params["nodes"]],direct_window,np.zeros_like(direct_window),np.ones(len(wave))))
    if not np.isfinite(base).all():raise FloatingPointError("储备池状态含非有限数值")
    norm=dict(input_rms=rms,input_window_scale=input_scale,target_rms=scale)
    return base,scale,slice(margin+1,train_samples-margin),norm

def fit_projection(short_wave,long_wave,target,fit_slice,alpha=1e-6):
    """训练区拟合两个实波形的2→1投影；不是门控，不引入图像结构。"""
    count=min(len(short_wave),len(long_wave),len(target))
    if fit_slice.stop>count or fit_slice.stop<=fit_slice.start:raise ValueError("投影拟合区不足")
    x=np.column_stack((short_wave[:count],long_wave[:count]))[fit_slice]
    y=np.asarray(target).real[:count][fit_slice]
    mean=x.mean(axis=0);std=np.maximum(x.std(axis=0),1e-8);ym=float(y.mean())
    z=(x-mean)/std
    coefficient=np.linalg.solve(z.T@z/len(z)+alpha*np.eye(2),z.T@(y-ym)/len(z))/std
    return dict(coefficient=coefficient.tolist(),bias=float(ym-mean@coefficient),source="仅波形拟合区",alpha=alpha)

def fusion_layer(short_wave,long_wave,projection):
    """两路自动截断到最短长度；不reshape、不补造样本。"""
    count=min(len(short_wave),len(long_wave))
    a,b=projection["coefficient"]
    result=a*np.asarray(short_wave[:count])+b*np.asarray(long_wave[:count])+projection["bias"]
    if not np.isfinite(result).all():raise FloatingPointError("投影输出非有限数值")
    return result

def convex_projection(weight):
    return dict(coefficient=[float(weight),float(1-weight)],bias=0.,source="校准BER选择的凸投影")
