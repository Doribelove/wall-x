"""A task-space reference for D1/D2 in simulation. Uses measured scene-object poses."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

from arm import tf,limited

DOWN=np.array([[0.,1.,0.],[0.,0.,-1.],[-1.,0.,0.]])
OPEN=.033


@dataclass
class Phase:
    name: str
    xyz: np.ndarray
    gap: float
    dwell: float
    tolerance: float
    free: bool


class TaskServo:
    def __init__(self,task,props,home):
        self.task=task; self.origin=props;self.track=np.zeros(3);self.yaw=0.;self.retries={};self.failure=""
        self.phases=[]
        self.ix=0;self.at=0.;self.steady=0.;self.age=0.
        self.events=[];self.stop=""
        self.home=np.array(home)
        self.left=self.home.copy()
        self.left_gap=.026
        ap=self.phases.append
        p=lambda n: np.array(props[n][0],float)
        def add(name,xyz,gap,dwell=.18,tol=.006,free=False):
            ap(Phase(name,np.array(xyz,float),gap,dwell,tol,free))
        def w(xy,z):return np.r_[np.array(xy)[:2],z]
        if task=="D1":
            self.names=("red2","red0","blue4")
            bowls=("yellow","yellow","mint")
            for i,(obj,destination) in enumerate(zip(self.names,bowls)):
                src=p(obj);bowl=p(destination)
                xy=bowl[:2]+np.asarray([[-.017,0.],[.017,0.],[0.,0.]][i])
                add(obj+"_align",w(src,.91),.021,free=True)
                add(obj+"_descend",w(src,.798),.021,.24,.0035)
                add(obj+"_grip",w(src,.798),.006,.72,.0045)
                add(obj+"_lift",w(src,.90),.003,.22,.007,True)
                add(obj+"_transfer",w(xy,.90),.003,.2,.006,True)
                add(obj+"_lower",w(xy,.847),.003,.24,.004)
                add(obj+"_release",w(xy,.847),OPEN,.55,.006)
                add(obj+"_withdraw",w(xy,.91),OPEN,.18,.008,True)
            mint=p("mint")
            rot=Rotation.from_quat(np.roll(props["mint"][1],-1)).as_matrix()
            lug=mint+(rot@np.array([0.,-.058,.029]))
            dd=np.array([.540,-.408])-mint[:2]
            add("bowl_align",w(lug,.910),OPEN,free=True)
            add("bowl_descend",w(lug,lug[2]+.150),OPEN,.3,.004)
            add("bowl_grip",w(lug,lug[2]+.150),.006,.8,.005)
            add("bowl_lift",w(lug,lug[2]+.19),.006,.2,.006)
            add("bowl_deliver",w(lug[:2]+dd,lug[2]+.19),.006,.3,.007,True)
            add("bowl_setdown",w(lug[:2]+dd,lug[2]+.152),.006,.4,.005)
            add("bowl_release",w(lug[:2]+dd,lug[2]+.152),OPEN,.55,.006)
            add("retreat",w(lug[:2]+dd,.93),OPEN,.5,.01,True)
        elif task=="D2":
            e=p("EraserProxy");b=p("Board");rot=Rotation.from_quat(np.roll(props["Board"][1],-1)).as_matrix()
            mark=lambda x:b+rot@np.array([x,-.232,.0062])
            m0=mark(-.024);m1=mark(.026)
            right_z=e[2]+.160
            self.left=w(b+rot@np.array([0.,.180,0.]),b[2]+.159)
            self.left_gap=.033
            add("align_eraser",w(e,right_z+.055),OPEN,.2,.009,True)
            add("lower_to_handle",w(e,right_z),OPEN,.30,.0038)
            add("close_on_handle",w(e,right_z),.002,.70,.005)
            add("lift_eraser",w(e,right_z+.050),.002,.25,.005,True)
            add("above_black_marks",w(m0,right_z+.050),.002,.2,.005,True)
            wipe_z=b[2]+.006+.005+.160-.0004
            add("establish_contact",w(m0,wipe_z),.002,.40,.0045)
            for i in range(4):
                add("wipe_out_"+str(i),w(m1,wipe_z),.002,.20,.004)
                add("wipe_back_"+str(i),w(m0,wipe_z),.002,.20,.004)
            add("lift_from_board",w(m0,right_z+.058),.002,.25,.006)
            add("return_to_dock",w(e,right_z+.056),.002,.25,.006,True)
            add("lower_into_dock",w(e,right_z),.002,.32,.005)
            add("release_tool",w(e,right_z),OPEN,.5,.006)
            add("retreat",w(e,right_z+.08),OPEN,.45,.008,True)
        else: raise ValueError(task)

    def goals(self,row,wrists,elapsed,props):
        def goal(phase):
            pt=phase.xyz.copy()
            prefix=phase.name.split("_")[0]
            if self.task=="D1" and prefix in ("red2","red0","blue4") and phase.name.endswith(("_align","_descend","_grip","_lift")):
                if not phase.name.endswith("_lift"):
                    xy=np.asarray(props[prefix][0])[:2]
                    self.track=np.clip(xy-np.asarray(self.origin[prefix][0])[:2],-.035,.035)
                pt[:2]+=self.track[:2]
            if self.task=="D2" and (phase.name.startswith("wipe_") or phase.name=="establish_contact"):
                b0,q0=self.origin["Board"];b,q=props["Board"]
                R0=Rotation.from_quat(np.roll(q0,-1)).as_matrix();R=Rotation.from_quat(np.roll(q,-1)).as_matrix()
                pt=np.asarray(b)+R@R0.T@(pt-b0)
                eraser=np.asarray(props["EraserProxy"][0])
                # Use the live tool center and board pose to remove lateral compliance during wiping.
                pt[:2]+=np.clip(wrists["right"][:2,3]-eraser[:2],-.025,.025)
            return pt

        phase=self.phases[min(self.ix,len(self.phases)-1)]
        error=float(np.linalg.norm(wrists["right"][:3,3]-goal(phase)))
        self.steady=self.steady+.04 if error<=phase.tolerance else 0.
        self.age+=.04
        if self.ix<len(self.phases) and (self.steady>=phase.dwell or self.age>4.0):
            if self.task=="D1" and phase.name in ("red2_lift","red0_lift","blue4_lift"):
                obj=phase.name.split("_")[0]; height=props[obj][0][2]-self.origin[obj][0][2]
                if height<.030 and self.retries.get(obj,0)<1:
                    self.retries[obj]=1; self.ix-=3; self.age=0.;self.steady=0.
                    self.events.append(dict(phase=phase.name,outcome="retry_after_empty_lift",rise_m=height,physics_s=elapsed))
                    phase=self.phases[self.ix]
                    return {"left":(tf(self.left,DOWN),self.left_gap),"right":(tf(goal(phase),Rotation.from_euler("z",self.yaw).as_matrix()@DOWN),phase.gap)},phase.name,np.zeros(3)
            self.events.append(dict(phase=phase.name,physics_s=elapsed,position_error_m=error,
                                    outcome="arrived" if self.steady>=phase.dwell else "timeout"))
            self.ix+=1
            self.steady=0.;self.age=0.
        phase=self.phases[min(self.ix,len(self.phases)-1)]
        if self.task=="D1":
            prefix=phase.name.split("_")[0]
            if prefix in ("red2","red0","blue4"):
                self.yaw=(np.pi*.5 if prefix=="red2" else 0.)
            else:self.yaw=0.
        if self.task=="D2":
            self.left_gap=.0465 if phase.name in ("align_eraser","lower_to_handle") or phase.name in ("release_tool","retreat") else .033
            if phase.name=="retreat":self.left[2]=self.origin["Board"][0][2]+.23
        residual=limited(row[10:13]*.18,.0025 if phase.free else .0003)
        left_pose=tf(self.left,DOWN)
        return {"left":(left_pose,self.left_gap),"right":(tf(goal(phase)+residual,Rotation.from_euler("z",self.yaw).as_matrix()@DOWN),phase.gap)},phase.name,residual
