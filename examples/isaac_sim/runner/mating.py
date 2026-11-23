"""Physical connector guides activated after measured 3-D alignment; only real relative motion advances a thread."""
import numpy as np
from scipy.spatial.transform import Rotation
from dexterous import quat,zrot,DOWN
from arm import tf,limited

def transform(x):return tf(np.asarray(x[0]),quat(x[1]))
def wrap(x):return (x+np.pi)%(2*np.pi)-np.pi
def yaw(R):return float(np.arctan2(R[1,0],R[0,0]))
def gd(R):
    from pxr import Gf
    q=Rotation.from_matrix(R).as_quat();return Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3].tolist()))
def unit(v):return v/max(np.linalg.norm(v),1.e-9)

class Assembly:
    def __init__(self,task,props,home,st,bodies):
        self.task=task;self.usb=task=='D5';self.st=st;self.bodies=bodies
        self.a,self.b=("BenchAssembly/FemaleShell","BenchAssembly/ServiceDrive") if self.usb else ("Bottle","Cap")
        self.A0=transform(props[self.a]); self.B0=transform(props[self.b])
        self.events=[];self.ix=0;self.time=0.;self.steady=0.;self.thread=None;self.spin=0.;self.last_yaw=None
        self.z0=0.;self.seated=False;self.grip_offsets={};self.phases=[];self.seen="";self.turn_index=0;self.start_yaw=0.
        self.liftedA=None;self.clearB=None;self.heldbase=None;self.heldangle=None
        self.log=[];self.angle_demand=0.;self.active_turn=None;self.twist_at_start=0.;self.demand_at_start=0.
        names=['locate','approach','close','test_lift','clear','approach_above','align','enter']
        if self.usb:names+=['press_'+str(j) for j in range(1,7)]+['release_drive','lower_assembly','release_base','inspect']
        else:
            for j in range(27):
                names+=['turn_'+str(j),'loosen_jaws_'+str(j),'reset_wrist_'+str(j),'regrip_'+str(j)]
            names+=['release_cap','withdraw','lift_bottle','deliver','set_down','release_base','inspect']
        self.phases=names
    def connector(self,A,B,name,t):
        from pxr import UsdPhysics,Gf
        rel=np.linalg.inv(A)@B; tip=rel[:3,3]+(rel[:3,:3]@np.array([0.,0.,.006]) if self.usb else 0.)
        radial=float(np.linalg.norm(tip[:2])); tilt=float(np.arccos(np.clip(rel[2,2],-1,1)));ang=yaw(rel[:3,:3]);z=float(rel[2,3]);self.metrics=(radial,tilt,z,float(tip[2]))
        if self.thread is None:
            ok=(abs(wrap(ang))<np.deg2rad(5) and -.0028<tip[2]<.003) if self.usb else (name=='enter' and .0583<z<.0628)
            if name not in ('align','enter') or not ok or radial>(.0017 if self.usb else .0022) or tilt>np.deg2rad(4):return
            j=UsdPhysics.Joint.Define(self.st,"/World/ContactMatingGuide")
            j.CreateExcludeFromArticulationAttr(True);j.CreateCollisionEnabledAttr(False)
            j.CreateLocalPos0Attr(Gf.Vec3f(0.,0.,z));j.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
            j.CreateLocalRot0Attr(gd(np.eye(3)));j.CreateLocalRot1Attr(gd(zrot(0. if self.usb else -ang)))
            for ax in ('transX','transY','rotX','rotY')+(('rotZ',) if self.usb else ()):
                lim=UsdPhysics.LimitAPI.Apply(j.GetPrim(),ax);lim.CreateLowAttr(1.);lim.CreateHighAttr(-1.)
            lim=UsdPhysics.LimitAPI.Apply(j.GetPrim(),'transZ');lim.CreateLowAttr(-.003 if self.usb else -.018);lim.CreateHighAttr(.0098-.006-z if self.usb else .0025)
            j.CreateBody0Rel().SetTargets([self.bodies[self.a].GetPath()]);j.CreateBody1Rel().SetTargets([self.bodies[self.b].GetPath()])
            self.zdrive=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'transZ')
            self.zdrive.CreateTypeAttr('force');self.zdrive.CreateStiffnessAttr(0. if self.usb else 7500.);self.zdrive.CreateDampingAttr(0. if self.usb else 18.)
            self.zdrive.CreateMaxForceAttr(12.);self.zdrive.CreateTargetPositionAttr(0.)
            if not self.usb:
                self.brake=UsdPhysics.DriveAPI.Apply(j.GetPrim(),'rotZ')
                self.brake.CreateTypeAttr('force');self.brake.CreateStiffnessAttr(.055);self.brake.CreateDampingAttr(.012)
                self.brake.CreateMaxForceAttr(.16);self.brake.CreateTargetPositionAttr(0.)
            self.thread=j;self.z0=z;self.last_yaw=ang
            self.log.append(dict(event="measured_mating_entry",time=t,phase=name,radial_m=radial,tilt_deg=float(np.rad2deg(tilt)),relative_z=z))
        if not self.usb and not self.seated:
            self.spin+=wrap(ang-self.last_yaw); self.last_yaw=ang
            intended=max(.0547,self.z0+self.spin*.002/(2*np.pi))
            self.zdrive.GetTargetPositionAttr().Set(float(intended-self.z0))
            complete=intended<=.0548 and abs(z-.0547)<.0013
        else:complete=bool(self.usb and name.startswith('press_') and tip[2]>=.00945 and radial<.0011)
        if complete and not self.seated:
            self.seated=True
            self.zdrive.GetStiffnessAttr().Set(5000.)
            self.zdrive.GetDampingAttr().Set(15.)
            self.zdrive.GetTargetPositionAttr().Set(float(z-self.z0))
            if self.usb:self.zdrive.GetMaxForceAttr().Set(1.2)
            if not self.usb:
                rot=UsdPhysics.DriveAPI.Apply(self.thread.GetPrim(),'rotZ')
                rot.CreateTypeAttr('force');rot.CreateStiffnessAttr(.12);rot.CreateDampingAttr(.02)
                rot.CreateMaxForceAttr(.16);rot.CreateTargetPositionAttr(float(np.rad2deg(wrap(ang-self.start_entry_yaw))))
            self.log.append(dict(event="seat",time=t,turns=float(-self.spin/(2*np.pi)),depth=float(tip[2]),radial_m=radial))
    def goals(self,row,wrists,t,props):
        A=transform(props[self.a]);B=transform(props[self.b]);W=wrists
        name=self.phases[min(self.ix,len(self.phases)-1)]
        if self.thread is None:self.start_entry_yaw=yaw((np.linalg.inv(A)@B)[:3,:3])
        self.connector(A,B,name,t)
        if self.thread is not None and not self.usb and not self.seated:
            # Self-locking thread friction holds the last angular reference during a regrasp.
            jaw_axis=A[:3,:3].T@W['right'][:3,1];twist=np.arctan2(jaw_axis[1],jaw_axis[0])
            if name.startswith('turn_'):
                if self.active_turn!=name:
                    self.active_turn=name;self.twist_at_start=twist;self.demand_at_start=self.angle_demand
                delta=float(np.clip(wrap(twist-self.twist_at_start),-np.deg2rad(54),0.))
                self.angle_demand=self.demand_at_start+delta
            self.brake.GetTargetPositionAttr().Set(float(np.rad2deg(wrap(self.angle_demand))))

        if self.seated and not self.usb and name.startswith(('turn_','loosen_jaws_','reset_wrist_','regrip_')):
            self.ix=self.phases.index('release_cap');name='release_cap'
        self.time+=.04
        if name!=self.seen:
            self.seen=name;self.time=.04;self.steady=0.
            if self.usb and name.startswith('press_'):
                self.feed_origin=float((np.linalg.inv(A)@B)[2,3])
            if name.startswith('turn_') or name=='enter':self.start_yaw=yaw((np.linalg.inv(A)@B)[:3,:3])
            if name in ('test_lift','lift_bottle','lower_assembly'):
                self.grip_offsets={s:np.linalg.inv(w)@obj for s,w,obj in [('left',W['left'],A),('right',W['right'],B)]}
            if name.startswith('turn_'):self.turn_wrist=W['right'].copy()
            if name.startswith('turn_') or name=='approach_above':
                self.grip_offsets['right']=np.linalg.inv(W['right'])@B
        usb=self.usb;up=np.array([0.,0.,1.])
        def tool(P,R,offset):return tf(P[:3,3]+P[:3,:3]@offset-R@np.array([.156 if usb else .148,0.,0.]),R)
        def from_obj(side,T,actual):
            # Recompute the observed grasp transform to compensate asymmetric pads and compliance.
            return T@np.linalg.inv(actual)@W[side]
        if usb:
            axis=unit(self.A0[:3,2]*[1,1,0]);width=unit(self.A0[:3,0]*[1,1,0]);rv=unit(-.65*axis-.76*up)
            yl=unit(width-rv*np.dot(rv,width)); RL=np.stack([rv,yl,np.cross(rv,yl)],axis=1)
            RR=zrot(yaw(self.B0[:3,:3]))@DOWN;ro=np.array([0,0,-.018]);lo=np.array([0,0,.026])
            ll=.003;rr=.002;looseL=.033;looseR=.029
        else:
            rv=unit(np.array([0.,-.82,-.57]));yl=np.array([1.,0.,0.]);RL=np.stack([rv,yl,np.cross(rv,yl)],axis=1)
            RR=zrot(np.deg2rad(110.))@DOWN;ro=np.array([0.,0.,.004]);lo=np.zeros(3)
            ll=.019;rr=.011;looseL=.037;looseR=.033
        l0=tool(self.A0,RL,lo);r0=tool(self.B0,RR,ro)
        hl=tool(A,RL,lo);hr=tool(B,RR,ro)
        if usb:
            # Use the distal 5 mm of the straight finger for a thin drive resting flat.
            hr[:3,3]+=up*.005;r0[:3,3]+=up*.005
            hl[:3,3]+=up*.008;l0[:3,3]+=up*.008
        L=l0.copy();R=r0.copy();gl=ll;gr=rr;dwell=.28;tol=.006;gate=True;end=False
        if name=='locate':
            L=hl;R=hr;L[:3,3]+=up*.065;R[:3,3]+=up*.065;gl=looseL;gr=looseR
        elif name in ('approach','close'):
            L=hl;R=hr
            if name=='approach':gl=looseL;gr=looseR
            else:dwell=.65
            tol=.012
        elif name=='test_lift':
            L=self.anchor_left.copy(); R=self.anchor_right.copy();R[:3,3]+=up*.025
            if usb:L[:3,3]+=up*.025
            gate=B[2,3]>self.B0[2,3]+.010;dwell=.32
        elif name=='clear':
            R=self.anchor_right.copy();R[:3,3]+=up*(.060 if usb else .126)
            L=self.fixedL.copy()
        else:
            L=self.fixedL.copy()
            BA=np.eye(4);BA[:3,:3]=A[:3,:3]
            if not usb:BA[:3,:3]=A[:3,:3]@zrot(self.initial_cap_yaw)
            if usb:
                depth={'approach_above':-.050,'align':-.012,'enter':-.0058}.get(name,.0037)
                if name.startswith('press_'):
                    idx=int(name.split('_')[1]);end_depth=(-.0058+.00172*idx if idx<6 else .0068)
                    depth=min(end_depth,self.feed_origin+.0018*self.time)
                    if self.thread is not None and not self.seated:
                        self.zdrive.GetStiffnessAttr().Set(600. if idx<6 else 1450.)
                        self.zdrive.GetDampingAttr().Set(6.)
                        self.zdrive.GetMaxForceAttr().Set(3.5)
                        self.zdrive.GetTargetPositionAttr().Set(float(depth-self.z0))
                BA[:3,3]=A[:3,3]+A[:3,:3]@np.array([0.,0.,depth])
            else:
                height={'approach_above':.089,'align':.072,'enter':.0618}.get(name,max(.0547,self.z0+self.spin*.002/(2*np.pi)) if self.thread else .0618)
                BA[:3,3]=A[:3,3]+A[:3,:3]@np.array([0.,0.,height])
            if name in ('approach_above','align','enter') or name.startswith('press_'):
                R=from_obj('right',BA,B);tol=.010
                err=np.linalg.norm(B[:3,3]-BA[:3,3]);axis_err=np.arccos(np.clip(np.dot(B[:3,2],BA[:3,2]),-1.,1.))
                gate=(err<(.003 if name in ('enter','align') else .005) and axis_err<.08)
                if name=='enter':gate=self.thread is not None
                if usb and name.startswith('press_'):
                    gate=gate and depth>=end_depth-.0001
                if name=='press_6':gate=self.seated;dwell=.45
            elif name.startswith('turn_'):
                BA[:3,:3]=A[:3,:3]@zrot(self.start_yaw-np.deg2rad(52))
                R=BA@np.linalg.inv(self.grip_offsets['right'])
                deltaR=A[:3,:3]@zrot(np.deg2rad(-52))@A[:3,:3].T
                R[:3,:3]=deltaR@self.turn_wrist[:3,:3]
                R[:3,3]=BA[:3,3]-R[:3,:3]@self.grip_offsets['right'][:3,3]
                tol=.014
                rot_done=-wrap(yaw((np.linalg.inv(A)@B)[:3,:3])-self.start_yaw)
                gate=rot_done>np.deg2rad(47);dwell=.12
            elif name.startswith(('loosen_jaws_','reset_wrist_','regrip_')):
                R=tool(B,RR,ro)
                if name.startswith('loosen_jaws_'):
                    R[:3,:3]=W['right'][:3,:3];gr=looseR
                elif name.startswith('reset_wrist_'):gr=looseR;R[:3,3]+=up*.003
                dwell=.28;tol=.014
            elif name in ('release_drive','release_cap','withdraw'):
                R=W['right'].copy() if name.startswith('release') else self.parkR.copy();gr=looseR;dwell=.45;tol=.015
                if name=='release_drive':dwell=2.20
                if name=='withdraw':R[:3,3]+=up*.070
            elif name in ('lift_bottle','deliver','set_down','lower_assembly','release_base','inspect'):
                R=self.parkR.copy();R[:3,3]+=up*(.123 if usb else .075);gr=looseR
                if name in ('lower_assembly','lift_bottle','deliver','set_down'):
                    desired=self.A0.copy()
                    if name=='lift_bottle':desired[:3,3]=A[:3,3];desired[2,3]=self.A0[2,3]+.035
                    if name in ('deliver','set_down'):desired[:3,3]=[.60,.116,self.A0[2,3]+(.035 if name=='deliver' else 0.)]
                    L=from_obj('left',desired,A);gate=np.linalg.norm(desired[:3,3]-A[:3,3])<.006;tol=.012
                    if name in ('set_down','lower_assembly'):self.finalL=W['left'].copy()
                else:
                    L=self.finalL.copy();gl=looseL;dwell=.55
                    if name=='inspect':L[:3,3]+=up*(.123 if usb else .07);dwell=2.15;tol=.016
        if name=='close':self.anchor_left=W['left'].copy();self.anchor_right=W['right'].copy(); self.initial_cap_yaw=yaw((np.linalg.inv(A)@B)[:3,:3])
        if name=='test_lift':self.fixedL=W['left'].copy()
        if name in ('release_cap','release_drive'):self.parkR=W['right'].copy();self.finalL=W['left'].copy()
        err=float(max(np.linalg.norm(W['right'][:3,3]-R[:3,3]),np.linalg.norm(W['left'][:3,3]-L[:3,3])))
        self.steady=self.steady+.04 if gate and err<tol else 0.
        timeout=4.5 if name=='press_6' else 3.1
        if self.steady>=dwell or self.time>timeout:
            self.events.append(dict(phase=name,physics_s=t,position_error_m=err,outcome='arrived' if self.steady>=dwell else 'timeout'))
            self.ix+=1
            if name in ('test_lift','enter','press_6') and self.time>timeout and not gate:
                self.events[-1]['outcome']='failed_lift_or_engagement';self.ix=len(self.phases)
        residual=limited(row[10:13]*.1,.0015 if name in ('locate','clear') else .00010)
        R=R.copy();R[:3,3]+=residual
        return {'left':(L,gl),'right':(R,gr)},name,residual
