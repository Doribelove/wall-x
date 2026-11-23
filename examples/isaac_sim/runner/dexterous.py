"""Bimanual task-space waypoints. Coordinates come from simulated objects, with measured hand feedback."""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from arm import tf, limited
from task_improved import DOWN, OPEN

def zrot(yaw):
    return Rotation.from_euler("z", yaw).as_matrix()

def quat(q):
    return Rotation.from_quat(np.roll(q,-1)).as_matrix()

@dataclass
class Move:
    name: str
    left: tuple
    right: tuple
    dwell: float
    tolerance: float
    loose: bool

class Bimanual:
    def __init__(self,task,props,home):
        self.task=task;self.phases=[];self.ix=0;self.dwell=0.;self.age=0.;self.events=[]
        self.cloth_pick={};self.cloth_on=set()
        def hand(p,r=DOWN,g=.029):return (tf(p,r),g)
        self.l=hand([.39,.15,.94]);self.r=hand([.39,-.14,.94])
        def step(n,l=None,r=None,d=.22,tol=.005,loose=False):
            if l is not None:self.l=l
            if r is not None:self.r=r
            self.phases.append(Move(n,self.l,self.r,d,tol,loose))
        def p(n):return np.array(props[n][0])
        if task=="D3":
            b=p("Bottle");c=p("Cap");dst=np.array([.600,.116, b[2]])
            sideways=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
            def bottle(pos,opening):
                return hand(pos-sideways@np.array([.146,0.,0.]),sideways,opening)
            def cap(pos,gap,angle=0):
                R=zrot(np.deg2rad(angle))@DOWN
                return hand(pos-R@np.array([.148,0.,0.]),R,gap)
            dz=lambda v:np.array([0,0,v])
            step("approach_parts",bottle(b+[0,.065,.045],.038),cap(c+dz(.065),.032),loose=True)
            step("grasp_bottle",bottle(b,.038),cap(c,.032),d=.30)
            step("grip_both",bottle(b,.018),cap(c,.010),d=.85)
            step("test_lift_cap",None,cap(c+dz(.025),.010),d=.35)
            step("clear_bottle_neck",None,cap(c+dz(.113),.010))
            top=b+dz(.0547+.010)
            step("align_cap_over_neck",None,cap(top+dz(.030),.010),d=.35)
            step("approach_threads",None,cap(top,.010),d=.4,tol=.003)
            step("reverse_find_thread",None,cap(top,.010,10),d=.4)
            for i in range(16):
                z=.0547+max(0.,.006-(i+1)*.00038)
                at=b+dz(z)
                step("clockwise_"+str(i+1),None,cap(at,.010,-57),d=.18,tol=.003)
                if i<15:
                    step("regrip_open_"+str(i+1),None,cap(at,.031,-57),d=.30)
                    step("index_wrist_"+str(i+1),None,cap(at+dz(.002),.031,0),d=.13)
                    step("regrip_"+str(i+1),None,cap(at,.010,0),d=.35)
            step("release_cap",None,cap(b+dz(.0547),.035,-57),d=.5)
            step("clear_cap",None,cap(b+dz(.135),.035),d=.4)
            step("lift_closed_bottle",bottle(b+dz(.035),.018),hand([.45,-.18,.94]),d=.35)
            step("deliver_bottle",bottle(dst+dz(.035),.018),None,loose=True)
            step("set_bottle_down",bottle(dst,.018),None,d=.45)
            step("release_bottle",bottle(dst,.038),None,d=.55)
            step("freestanding_check",bottle(dst+[0,.075,.040],.038),None,d=1.20)
        elif task=="D4":
            nodes=np.array(props["cloth_samples_world_m"]);lo=np.min(nodes,axis=0);hi=np.max(nodes,axis=0)
            x0,x1=lo[0]+.002,hi[0]-.002;y0,y1=lo[1]+.002,hi[1]-.002
            floor=lo[2];xm=(x0+x1)/2;ym=(y0+y1)/2
            def cloth(x,y,z,gap):return hand(np.array([x,y,z+.1624]),DOWN,gap)
            def both(n,xyL,xyR,z,g,d=.25,tol=.005):
                step(n,cloth(*xyL,z,g),cloth(*xyR,z,g),d,tol)
            l=(x0,y1);r=(x0,y0)
            both("find_orange_edge",l,r,floor+.056,.022)
            both("touch_first_two_corners",l,r,floor+.001,.018,.38)
            both("pinch_first_edge",l,r,floor+.001,.002,.55)
            self.cloth_pick["pinch_first_edge"]=("left","right")
            for a in np.linspace(18,180,12):
                t=np.deg2rad(a)
                x=xm-(xm-x0)*np.cos(t);z=floor+.005+.13*np.sin(t)
                both("fold_x_"+str(round(a)),(x,y1),(x,y0),z,.002,.12,.008)
            both("release_first_fold",(x1,y1),(x1,y0),floor+.005,.024,.55)
            both("clear_first_fold",(x1,y1),(x1,y0),floor+.095,.027)
            l=(x1-.004,y0+.002);r=(xm+.006,y0+.002)
            both("approach_folded_edge",l,r,floor+.095,.024)
            both("touch_second_edge",l,r,floor+.002,.018,.35)
            both("pinch_second_edge",l,r,floor+.002,.002,.6)
            self.cloth_pick["pinch_second_edge"]=("left","right")
            for a in np.linspace(16,180,12):
                t=np.deg2rad(a);y=ym-(ym-y0)*np.cos(t);z=floor+.008+.136*np.sin(t)
                both("fold_y_"+str(round(a)),(l[0],y),(r[0],y),z,.002,.14,.009)
            l=(l[0],y1);r=(r[0],y1)
            both("release_second_fold",l,r,floor+.008,.024,.65)
            both("clear_second_fold",l,r,floor+.090,.026)
            # Pinch both layers near the outside edge, then lift and set down in the marked area.
            l=(x1-.008,y1-.006);r=(xm+.008,y1-.006)
            both("touch_folded_stack",l,r,floor+.004,.018,.35)
            both("grip_folded_stack",l,r,floor+.004,.002,.55)
            self.cloth_pick["grip_folded_stack"]=("left","right")
            both("lift_folded_stack",l,r,floor+.065,.002)
            delta=np.array([.61-(x1+xm)/2,.20-(y1+ym)/2])
            l=np.array(l)+delta;r=np.array(r)+delta
            both("transfer_to_storage",l,r,floor+.068,.002,.3,.01)
            both("set_stack_down",l,r,floor+.008,.002,.4,.010)
            both("release_in_storage",l,r,floor+.010,.027,.55,.01)
            both("inspect_free_stack",l,r,floor+.095,.028,1.1,.013)
        elif task=="D5":
            male=p("BenchAssembly/ServiceDrive"); female=p("BenchAssembly/FemaleShell")
            fm=quat(props["BenchAssembly/FemaleShell"][1]);mm=quat(props["BenchAssembly/ServiceDrive"][1])
            # Female local +Z points inward. The service-drive origin is 6 mm behind its tip.
            face=female;axis=fm[:,2];worldz=np.array([0.,0.,1.])
            angle=np.arctan2(fm[1,0],fm[0,0]);start_angle=np.arctan2(mm[1,0],mm[0,0])
            lrot=zrot(angle+np.pi/2)@DOWN;r0=zrot(start_angle+np.pi/2)@DOWN;rf=zrot(angle+np.pi/2)@DOWN
            left_location=female+fm@np.array([0.,0.,.026])
            def lh(p,g):return hand(p+worldz*.149,lrot,g)
            def rh(p,g,ori):return hand(p+worldz*.149,ori,g)
            plug=male+mm@np.array([0.,0.,-.013])
            step("locate_matching_parts",lh(left_location+worldz*.067,.025),rh(plug+worldz*.065,.026,r0),loose=True)
            step("descend_to_handles",lh(left_location,.022),rh(plug,.022,r0),d=.35)
            step("grip_socket_and_drive",lh(left_location,.003),rh(plug,.002,r0),d=.70)
            step("lift_both",lh(left_location+worldz*.025,.003),rh(plug+worldz*.035,.002,r0),d=.35)
            target=face+worldz*.025
            # Target gripper location refers to the male tail, 19 mm behind the tip.
            step("align_keyed_drive",None,rh(target-axis*.054,.002,rf),d=.3)
            step("at_socket_mouth",None,rh(target-axis*.019,.002,rf),d=.4,tol=.002)
            for i in range(1,6):
                step("insert_"+str(i),None,rh(target-axis*(.019-.002*i),.002,rf),d=.18,tol=.0025)
            step("support_assembly",lh(left_location,.003),rh(face-axis*.009,.002,rf),d=.55)
            step("release_on_bench",lh(left_location,.026),rh(face-axis*.009,.026,rf),d=.65)
            step("retention_check",lh(left_location+worldz*.070,.026),rh(face-axis*.009+worldz*.068,.026,rf),d=1.20)
        else:raise ValueError(task)

    def goals(self,row,wrists,elapsed,props=None):
        i=min(self.ix,len(self.phases)-1);ph=self.phases[i]
        errors=[np.linalg.norm(wrists[s][:3,3]-v[0][:3,3]) for s,v in (("left",ph.left),("right",ph.right))]
        angles=[np.linalg.norm(Rotation.from_matrix(v[0][:3,:3]@wrists[s][:3,:3].T).as_rotvec()) for s,v in (("left",ph.left),("right",ph.right))]
        self.dwell=self.dwell+.04 if max(errors)<=ph.tolerance and max(angles)<.13 else 0.
        self.age+=.04
        if self.ix<len(self.phases) and (self.dwell>=ph.dwell or self.age>=2.7):
            self.events.append(dict(phase=ph.name,physics_s=elapsed,position_error_m=float(max(errors)),outcome="arrived" if self.dwell>=ph.dwell else "timeout"))
            self.ix+=1;self.dwell=0.;self.age=0.
        ph=self.phases[min(self.ix,len(self.phases)-1)]
        residual=limited(row[10:13]*.12,.0015 if ph.loose else .00012)
        dest=ph.right[0].copy();dest[:3,3]+=residual
        return {"left":ph.left,"right":(dest,ph.right[1])},ph.name,residual
