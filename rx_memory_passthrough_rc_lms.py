"""读取验证已冻结的双通道配置，保持公共链路不变，不进行扫描或训练。"""
import json
from threadpoolctl import threadpool_limits
from memory_passthrough_sweep import LOCK,NN,evaluate,save

if __name__=="__main__":
    if not LOCK.exists():raise FileNotFoundError("请先运行memory_passthrough_sweep.py完成训练内验证扫描")
    selection=json.loads(LOCK.read_text(encoding="utf-8"))
    path=NN/"memory_passthrough_summary.json"
    previous=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    with threadpool_limits(limits=8):result=evaluate(selection)
    if previous.get("selection")==selection:
        for name in ("elapsed_seconds","protected_files_unchanged"):
            if name in previous:result[name]=previous[name]
    save(path,result)
