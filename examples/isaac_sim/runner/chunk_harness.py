"""Blend overlapping forecasts in a common task frame and slew joint targets at physics rate."""
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from arm import action_rotation, tf

class ChunkHarness:
    def __init__(self):
        self.history=deque(maxlen=3)
    def add(self,start,actions,anchor,lift):
        self.history.append((start,actions,anchor,lift[:3,:3]))
    def target(self,t,side):
        off=0 if side=="left" else 10
        seq=[]
        for start,act,anc,rl in self.history:
            idx=t-start
            if 0<=idx<len(act):
                row=act[idx,off:off+10]
                base=anc[side]
                seq.append((np.exp(-idx/13.),base[:3,3]+rl@row[:3],
                            rl@action_rotation(row[3:9])@rl.T@base[:3,:3],row[9]))
        w=np.array([v[0] for v in seq]); w/=w.sum()
        p=w@np.array([v[1] for v in seq])
        rotation=Rotation.from_matrix(np.array([v[2] for v in seq])).mean(weights=w).as_matrix()
        jaw=float(w@np.array([v[3] for v in seq]))
        return tf(p,rotation),jaw

def measured_lift_state(x,rec):
    fixed=np.linalg.inv(np.asarray(rec["chassis_T_lift_native"]))
    x=x.copy()
    for off,side in ((0,"left"),(10,"right")):
        wrist=fixed@np.asarray(rec["chassis_T_gripper_base_candidate"][side])
        x[off:off+3]=wrist[:3,3]
        x[off+3:off+9]=wrist[:2,:3].reshape(-1)
    return x
