"""Lift-frame target tracking for the EX001 V6 URDF. Distances in metres, angles in radians."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation


def tf(p, r):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = r, p
    return t


def limited(v, maximum):
    return v * min(1., maximum / max(np.linalg.norm(v), 1.e-9))


def action_rotation(values):
    a, b = np.array(values, dtype=float).reshape(2, 3)
    if np.linalg.norm(a) < .05:
        raise ValueError("degenerate first rotation axis")
    a /= np.linalg.norm(a)
    b -= b.dot(a)*a
    if np.linalg.norm(b) < .05:
        raise ValueError("degenerate second rotation axis")
    b /= np.linalg.norm(b)
    return np.array([a, b, np.cross(a,b)])


class Arm:
    def __init__(self, urdf, side):
        self.names = [f"{side}_arm_joint{i}" for i in range(1,7)]
        links = {j.find("child").get("link"):j for j in ET.parse(urdf).getroot().findall("joint")}
        n, reverse = f"{side}_arm_gripper_base_link", []
        while n != "chassis_base_link":
            j = links[n]
            origin = j.find("origin")
            def xyz(tag, default):
                if origin is None:
                    return np.fromstring(default, sep=" ")
                return np.fromstring(origin.get(tag,default), sep=" ")
            transform = tf(xyz("xyz","0 0 0"), Rotation.from_euler("xyz",xyz("rpy","0 0 0")).as_matrix())
            axis = j.find("axis")
            axis = np.fromstring("1 0 0" if axis is None else axis.get("xyz"),sep=" ")
            reverse.append((j.get("name"),j.get("type"),transform,axis,j.find("limit")))
            n = j.find("parent").get("link")
        self.chain = list(reversed(reverse))
        limits = {name:li for name,_,_,_,li in self.chain if name in self.names}
        self.lo, self.hi, self.vmax = np.array([[float(limits[n].get(k)) for k in ("lower","upper","velocity")] for n in self.names]).T

    def kinematics(self, q, lift):
        t = np.eye(4)
        origins, axes = [], []
        for name, typ, fixed, axis, _ in self.chain:
            t = t @ fixed
            if name in self.names:
                i = self.names.index(name)
                origins.append(t[:3,3].copy())
                axes.append(t[:3,:3] @ axis)
                t = t @ tf([0,0,0],Rotation.from_rotvec(q[i]*axis).as_matrix())
            elif typ == "prismatic":
                assert name == "lift_joint"
                t = t @ tf(lift*axis,np.eye(3))
        jac = np.vstack([np.cross(axes,t[:3,3]-np.array(origins)).T,np.array(axes).T])
        return t, jac

    def step(self,q,lift,desired,dt):
        now,j=self.kinematics(q,lift)
        p=limited(desired[:3,3]-now[:3,3], .32*dt)
        w=limited(Rotation.from_matrix(desired[:3,:3]@now[:3,:3].T).as_rotvec(), 1.8*dt)
        weighted=j.copy()
        weighted[3:]*=.12
        d=np.r_[p,.12*w]
        dq=weighted.T @ np.linalg.solve(weighted@weighted.T+.018**2*np.eye(6),d)
        return np.clip(q+np.clip(dq,-self.vmax*dt*.55,self.vmax*dt*.55),self.lo+.001,self.hi-.001)

    def initial(self,q,lift,p):
        q=np.array(q,dtype=float)
        goal=self.kinematics(q,lift)[0].copy()
        goal[:3,3]=p
        for _ in range(140):
            q=self.step(q,lift,goal,.035)
            if np.linalg.norm(self.kinematics(q,lift)[0][:3,3]-p)<.0008:
                break
        return q
