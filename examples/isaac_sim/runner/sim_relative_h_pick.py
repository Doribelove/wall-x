"""Custom fixed-lift V6 adapter based on official reset-relative semantics; not SDK hardware or benchmark certification."""
import math
import numpy as np
from native_prefix_math import Rig,RIGHT,DT,need,rigid,angle,rot6,tf
M_PER_RAD=.00852694
H_EXACT_SIM_MAX=5.717175   # Official H legacy root and V6 pinned URDF
H_LAB_APERTURE_MAX=.033  # independently verified D1 native slider cap; veto proposals above, do not clamp
H_INIT_RAD=3.75
FRAMES='MANAENV_PREVIEW_HOME_RELATIVE_LIFT_LOCAL_V6_H_PRISMATIC_COMPATIBILITY_R1'
KEEP_FROZEN=('left_arm','left_gripper','base','head','lift')
INSTRUCTION='Pick up the red cube directly underneath your right gripper with your right hand, and put it into the yellow bowl. Do not move any of the other pieces.'
def g_to_signed(g):
 need(np.isfinite(g) and 0<=g<=H_EXACT_SIM_MAX,'H_GRIPPER_PROFILE_RANGE')
 return np.asarray([1.,-1.])*float(g)*M_PER_RAD
def ee_in_lift(rec):
 inv=np.linalg.inv(rigid(rec['chassis_T_lift_native']))
 return {s:rigid(inv@rigid(rec['chassis_T_gripper_base_candidate'][s])) for s in ('left','right')}
def attach_home(initial,record):
 # Next frames receive ONLY the freeze of original exact measured lift-local EE; no per-query re-zeroing.
 init=ee_in_lift(initial)
 need(type(initial['physics_tick']) is int and record['physics_tick']>=initial['physics_tick'],'reset reference tick')
 record.update(relative_protocol=FRAMES,initial_home_tick=initial['physics_tick'],
  fixed_home_lift_T_ee={s:T.tolist() for s,T in init.items()})
def state(rec):
 need(rec['relative_protocol']==FRAMES and rec['initial_home_tick']<=rec['physics_tick'],'diagnostic packet lacks frozen home')
 now=ee_in_lift(rec); e=rec['measured_encoder'];a=[];j={}
 for s in ('left','right'):
  h=rigid(rec['fixed_home_lift_T_ee'][s]);v=now[s]
  # Official ManaEnv relative position and left-multiplied rotation, relative in lift_local for wholebody.
  p=v[:3,3]-h[:3,3];R=v[:3,:3]@h[:3,:3].T
  m=np.array([e[f'{s}_arm_gripper_{q}_joint'] for q in ('left','right')],float)
  need(np.isfinite(m).all() and m[0]>=-.00005 and m[1]<=.00005 and max(abs(m))<=.0488,'UNSUPPORTED H ENCODER SIGN/BOUNDS')
  # Author reads the passive right mimic to measure contact, not gripper motor command.
  g=float(-m[1]/M_PER_RAD)
  need(-.006<=g<=H_EXACT_SIM_MAX+.006,'BAD H MIMIC INVERSE')
  g=max(0.,min(H_EXACT_SIM_MAX,g));j[s]=m.tolist()
  a.extend([*p,*R[:2].reshape(-1),g])
 a.extend([0.,0.,0.,e['lift_joint'],e['head_yaw_joint'],e['head_pitch_joint']])
 out=np.asarray(a,np.float32);need(out.shape==(26,) and np.isfinite(out).all(),'invalid H-relative state')
 return out,j
def plan(rig,rec,row,allow_qualified_open_cap=False):
 row=np.asarray(row,dtype=float)
 need(row.shape==(26,) and np.isfinite(row).all(),'INVALID_RAW_ROW')
 X,j=state(rec);L=rigid(rec['chassis_T_lift_native']);home=rigid(rec['fixed_home_lift_T_ee']['right']);now=ee_in_lift(rec)['right']
 q=np.array([rec['measured_encoder'][k] for k in RIGHT]);hl=float(rec['measured_encoder']['lift_joint'])
 A=rigid(rec['chassis_T_gripper_base_candidate']['right']);oracle=rig.fk(q,hl)
 need(np.linalg.norm(A[:3,3]-oracle[:3,3])<.002 and angle(A[:3,:3]@oracle[:3,:3].T)<.007,'SIM_EEF_FK_PARITY')
 dP=row[10:13];dR=rot6(row[13:19]);p_rel=(now[:3,3]-home[:3,3])+dP;R_rel=dR@(now[:3,:3]@home[:3,:3].T)
 target_T=rigid(L@tf(home[:3,3]+p_rel,R_rel@home[:3,:3])); delta_r=angle(dR)
 need(np.linalg.norm(dP)<.024 and delta_r<math.radians(12),'RAW_MODEL_SE3_EXCEEDS_LAB_BOUNDS')
 need(.26<=target_T[0,3]<=.70 and -.47<=target_T[1,3]<=.03 and .787<=target_T[2,3]<=1.05,'OUTSIDE_LAB_REACH_ENVELOPE')
 j0=np.array(j['right'],float);jg=g_to_signed(row[19]);raw_jg=jg.copy()
 need(np.max(np.abs(j0))<=H_LAB_APERTURE_MAX+.0015,'NATIVE_MEASURED_JAW_EXCEEDS_QUALIFIED_COMPLIANCE')
 clipped=bool(np.max(np.abs(jg))>H_LAB_APERTURE_MAX)
 if clipped:
  need(allow_qualified_open_cap and np.max(np.abs(jg))>np.max(np.abs(j0)),'AUTHOR_H_POSE_VALID_BUT_OUTSIDE_QUALIFIED_D1_PAD_TRAVEL')
  # Bound only a legal monotonic opening to previously demonstrated actuator stroke. No closed/loaded substitute.
  jg=np.array([H_LAB_APERTURE_MAX,-H_LAB_APERTURE_MAX])
 need(np.linalg.norm(dP)>=.0006 or delta_r>=.004 or np.max(np.abs(jg-j0))>=.00035,'NO_MEASURABLE_INTENT')
 goal=rig.ik(target_T[:3,3],target_T[:3,:3],q,hl)
 need(np.max(np.abs(goal-q))<=.11,'NONLOCAL_IK_BRANCH')
 # Exact pose/jaw endpoints, with fixed time interpolation (not targets replacements).
 ticks=max(25,int(math.ceil(np.max(np.abs(jg-j0))/(DT*.045))),int(math.ceil(np.max(np.abs(goal-q)/(DT*rig.vmax*.20)))))
 need(ticks<=160,'QUALIFIED_CAPPED_ENDPOINT_CANNOT_BE_REACHED_IN_BOUNDED_TIME')
 qpath=np.linspace(q,goal,ticks+1);gpath=np.linspace(j0,jg,ticks+1)
 # Measured start may be slightly beyond a previously issued signed command from compliance.
 # Sweep uses exact measured initial point; commanded points may never exceed checked actuator travel.
 gpath[1:]=np.clip(gpath[1:],-H_LAB_APERTURE_MAX,H_LAB_APERTURE_MAX)
 need(np.all(np.abs(np.diff(qpath,axis=0))/DT<=rig.vmax*.25) and np.max(np.abs(np.diff(gpath[1:],axis=0))/DT)<=.060,'DRIVE_VELOCITY_LIMIT')
 return {'q_start':q.tolist(),'qpath_after_start':qpath[1:].tolist(),'jaw_start':j0.tolist(),'jawpath_after_start':gpath[1:].tolist(),
   'gross_right_SE3_delta_m':dP.tolist(),'gross_right_rot_rad':delta_r,
   'requested_jaw_motor_rad':float(row[19]),'requested_signed_jaw_m':raw_jg.tolist(),'applied_signed_jaw_m':jg.tolist(),'qualified_open_cap_applied':clipped,'goal_chassis_T_ee':target_T.tolist(),
   'relative_protocol':FRAMES,'muted_model_axes':list(KEEP_FROZEN)}
