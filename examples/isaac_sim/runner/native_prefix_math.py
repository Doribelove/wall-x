"""EX001 body-EF→position-drive conversion for one native right-arm-only model row. No simulator or hardware imports."""
from pathlib import Path
import math,xml.etree.ElementTree as ET
import numpy as np
WALL_GRIP_G_MAX=1.526
FING_G_M_PER_RAD=.00852694
MAX_XYZ=.016
MAX_ANG=math.radians(8)
DT=.004
MIN_DELTA=.0007
N_TICKS=25
TABLE_Z=.630
RIGHT=tuple(f'right_arm_joint{i}' for i in range(1,7))
TCP='right_arm_gripper_base_link'
def need(b,r):
 if not b:raise RuntimeError(r)
def rigid(T):
 T=np.asarray(T,float)
 need(T.shape==(4,4) and np.isfinite(T).all() and np.allclose(T[3],[0,0,0,1],atol=1e-6) and np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-4) and np.linalg.det(T[:3,:3])>.999,'BAD_SE3')
 return T
def state26(rec):
 values=[];jaw={}
 for side in ('left','right'):
  t=rigid(rec['chassis_T_gripper_base_candidate'][side]);enc=rec['measured_encoder']
  p=np.array([enc[f'{side}_arm_gripper_{j}_joint'] for j in ('left','right')],float)
  need(np.isfinite(p).all() and p[0]>-.0001 and p[1]<.0001 and abs(p.sum())<.0004,'SIGNED_JAW_ENCODERS_DISAGREE')
  v=float(.5*(p[0]-p[1])/FING_G_M_PER_RAD)
  need(0<=v<=WALL_GRIP_G_MAX+1e-5,'JAW_STATE_OUTSIDE_G_WIRE_RANGE')
  jaw[side]=p.tolist();values.extend([*t[:3,3],*t[:2,:3].reshape(6),v])
 values.extend([0.,0.,0.,enc['lift_joint'],enc['head_yaw_joint'],enc['head_pitch_joint']])
 v=np.asarray(values,np.float32);need(v.shape==(26,) and np.isfinite(v).all(),'BAD_NATIVE_STATE_26')
 return v,jaw
def signed_goal(g):
 need(np.isfinite(g) and 0<=g<=WALL_GRIP_G_MAX,'G_ACTION_OUTSIDE_VENDOR_PROFILE')
 return np.array([1.,-1.])*FING_G_M_PER_RAD*g
def axes(v,k):
 if v is None:return np.array(k,float)
 return np.fromstring(v,sep=' ')
def rx(k,a):
 x,y,z=k/np.linalg.norm(k);m=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
 return np.eye(3)+np.sin(a)*m+(1-np.cos(a))*(m@m)
def tf(p,r):
 x=np.eye(4);x[:3,3]=p;x[:3,:3]=r;return x
def rot6(v):
 a,b=np.asarray(v,float).reshape(2,3);an=np.linalg.norm(a)
 need(.05<an<2.,'DEGENERATE_R6_AXIS_A')
 a=a/an;b=b-(b@a)*a;bn=np.linalg.norm(b);need(.05<bn<2.,'DEGENERATE_R6_AXIS_B')
 b=b/bn;return np.stack([a,b,np.cross(a,b)])
def angle(R):
 return float(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
class Rig:
 def __init__(self,urdf):
  rt=ET.parse(urdf).getroot();cm={j.find('child').get('link'):j for j in rt.findall('joint')}
  for side in ('left','right'):
   for name,sign in [('left',1),('right',-1)]:
    j=next(z for z in cm.values() if z.get('name')==f'{side}_arm_gripper_{name}_joint');m=j.find('mimic')
    need(j.get('type')=='prismatic' and m.get('joint')==f'{side}_arm_gripper' and abs(float(m.get('multiplier'))-sign*FING_G_M_PER_RAD)<1e-10 and float(m.get('offset','0'))==0.,'MIMIC_GEOMETRY_DISAGREES')
  n=TCP;self.joints=[];a={}
  while n!='chassis_base_link':
   j=cm[n];nm=j.get('name');tp=j.get('type');o=j.find('origin');ax=j.find('axis')
   p=axes(o.get('xyz'),[0,0,0]) if o is not None else np.zeros(3)
   th=axes(o.get('rpy'),[0,0,0]) if o is not None else np.zeros(3)
   A=axes(ax.get('xyz'),[1,0,0]) if ax is not None else np.array([1.,0,0.])
   M=tf(p,rx([0.,0.,1.],th[2])@rx([0.,1.,0.],th[1])@rx([1.,0.,0.],th[0]));self.joints.append((nm,tp,M,A))
   if nm in RIGHT:
    jm=j.find('limit');a[nm]=[float(jm.get(k)) for k in ('lower','upper','velocity')]
   n=j.find('parent').get('link')
  self.joints.reverse()
  self.lo,self.hi,self.vmax=np.array([a[k] for k in RIGHT]).T
 def fk(self,q,lift):
  E=dict(zip(RIGHT,q));E['lift_joint']=lift;T=np.eye(4)
  for n,tp,f,ax in self.joints:
   T=T@f
   if tp=='prismatic':T=T@tf(E[n]*ax,np.eye(3))
   elif tp in ('revolute','continuous'):T=T@tf([0.,0,0],rx(ax,E[n]))
  return T
 def ik(self,p,R,seed,lift):
  def res(q):
   x=self.fk(q,lift);return np.r_[x[:3,3]-p,.105*(x[:3,:3]-R).reshape(9)]
  q=np.asarray(seed,float).copy();diff=.00015*np.eye(6)
  for _ in range(60):
   cur=self.fk(q,lift);y=res(q)
   if np.linalg.norm(y[:3])<.00015 and angle(R@cur[:3,:3].T)<.002:return q
   jac=np.column_stack([(res(q+e)-res(q-e))/.0003 for e in diff])
   q=q+np.linalg.solve(jac.T@jac+2e-6*np.eye(6),-jac.T@y).clip(-.03,.03)
   need(np.all(q>self.lo+.009) and np.all(q<self.hi-.009),'IK_LIMIT_OR_BRANCH_JUMP')
  raise RuntimeError('NO_TOLERANT_IK_FOR_EXACT_MODEL_TARGET')
def one_prefix(rig,rec,row):
 r=np.asarray(row,float)
 need(r.shape==(26,) and np.isfinite(r).all(),'INVALID_MODEL_ROW')
 enc=rec['measured_encoder'];q=np.array([enc[k] for k in RIGHT]);lift=float(enc['lift_joint'])
 T=rigid(rec['chassis_T_gripper_base_candidate']['right'])
 r0=rig.fk(q,lift)
 need(np.linalg.norm(T[:3,3]-r0[:3,3])<=.002 and angle(T[:3,:3]@r0[:3,:3].T)<=.007,'NATIVE_KINEMATIC_FRAME_PARITY_FAIL')
 delta=r[10:13];ang=angle(rot6(r[13:19]))
 need(np.linalg.norm(delta)<=MAX_XYZ,'MODEL_DELTA_EXCEEDS_DECLARED_PREFIX_BOUND')
 need(np.linalg.norm(delta)>=MIN_DELTA,'MODEL_MEANINGFUL_MOTION_NOT_REQUESTED')
 abs_T=rigid(tf(T[:3,3]+delta,T[:3,:3]))
 need(.25<=abs_T[0,3]<=.72 and -.57<=abs_T[1,3]<=.23 and abs_T[2,3]>=TABLE_Z+.175 and abs_T[2,3]<1.13,'OUTSIDE_RIGHT_APPROACH_ONLY_WORKSPACE')
 target=rig.ik(abs_T[:3,3],abs_T[:3,:3],q,lift)
 need(np.max(np.abs(target-q))<=.075,'NONLOCAL_IK_BRANCH')
 # Time interpolation only. Do not clip or replace the requested endpoint. No objects as input.
 ticks=np.linspace(q,target,N_TICKS+1)[1:,:]
 need(np.all(np.abs(np.diff(np.vstack([q,ticks]),axis=0))/DT<=rig.vmax*.25),'VELOCITY_LIMIT')
 jaw_measured=float(.5*(enc['right_arm_gripper_left_joint']-enc['right_arm_gripper_right_joint']))
 need(.0<=jaw_measured<.0131,'MODEL_G_PROFILE_JAW_HOLD_RANGE')
 return {'start_q_rad':q.tolist(),'targets_q_rad':ticks.tolist(),'model_right_delta_xyz_m':delta.tolist(),'model_right_delta_angle_rad_not_enabled':ang,
    'model_right_ee_absolute_chassis':abs_T.tolist(),'model_right_gripper_proposal_not_enabled':float(r[19]),
    'frozen_signed_jaw_half_opening_m':jaw_measured,'state_anchor_lift_m':lift,
    'muted_by_declared_approach_only_profile':['right_arm_orientation','left_arm','left_gripper','right_gripper','base','head','lift']}
