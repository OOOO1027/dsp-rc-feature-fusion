"""使用训练内验证已冻结的CH3配置，CH1保留原40节点二次岭读出。"""
import json
from threadpoolctl import threadpool_limits
from ch3_capacity_sweep import LOCK,evaluate

if __name__=="__main__":
    if not LOCK.exists():
        raise FileNotFoundError("尚无冻结参数，请先运行ch3_capacity_sweep.py")
    selection=json.loads(LOCK.read_text(encoding="utf-8"))
    with threadpool_limits(limits=8):
        evaluate(selection,include_candidates=False)
