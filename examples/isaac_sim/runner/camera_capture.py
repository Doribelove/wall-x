"""Passive Isaac 6.0.1 D1 shadow observation hook. No controller, no action dispatch."""
from pathlib import Path
import numpy as np,hashlib,json
from dataclasses import dataclass
from lens_5756_rtx import author,validate_authored,recipe,k_record
CAMS={'face_view':'camera_head_front_color_optical_frame','left_wrist_view':'left_arm_gripper_camera_color_frame','right_wrist_view':'right_arm_gripper_camera_color_frame'}
BASE='chassis_base_link'
EES={s:s+'_arm_gripper_base_link' for s in ('left','right')}
DOF=('lift_joint','head_yaw_joint','head_pitch_joint')+tuple(s+'_arm_joint'+str(i) for s in EES for i in range(1,7))+tuple(s+'_arm_gripper_'+n+'_joint' for s in EES for n in ('left','right'))
K={'width':640,'height':480,'fx':449.,'fy':449.,'cx':319.5,'cy':239.5,'status':'D1 head and uncalibrated diagnostic wrist pinhole shared prior, NOT physical or vendor intrinsics'}
def req(b,msg):
 if not b:raise RuntimeError(msg)
def pose(p):
 from pxr import Gf,UsdGeom
 m=UsdGeom.XformCache().GetLocalToWorldTransform(p)
 T=np.eye(4);T[:3,3]=tuple(m.ExtractTranslation())
 for i in range(3):
  T[:3,i]=tuple(m.TransformDir(Gf.Vec3d(*np.eye(3)[i].tolist())))
 req(np.isfinite(T).all() and np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-4),'world transform is not rigid')
 return T
@dataclass
class View:
 frame:object;cam:object;sensor:object;product:str
class Capture:
 def __init__(self,stage,robot,world,art):
  from pxr import Gf,Usd,UsdGeom,UsdRender
  from isaacsim.sensors.experimental.rtx import RtxCamera,CameraSensor
  req(stage.GetEditTarget().GetLayer()==stage.GetSessionLayer(),'session-only cameras')
  self.world,self.art=world,art
  allp=list(Usd.PrimRange(robot,Usd.TraverseInstanceProxies()))
  self.links={}
  for n in (BASE,"lift_link",*EES.values(),*CAMS.values()):
   found=[p for p in allp if p.GetName()==n]
   req(len(found)==1 and not found[0].IsInstanceProxy(),'non-unique/noneditable physical mount: '+n)
   self.links[n]=found[0]
  self.ix=[int(art.get_dof_index(n)) for n in DOF]
  req(len(set(self.ix))==len(self.ix) and min(self.ix)>=0,'unresolved encoder DOF')
  self.views={}
  for nm,frame in CAMS.items():
   op=self.links[frame]
   path=op.GetPath().AppendChild('D1WALLDiagOpticalToUsd')
   xf=UsdGeom.Xform.Define(stage,path).GetPrim()
   cam=UsdGeom.Camera.Define(stage,path.AppendChild('D1WALLDiagRTX'))
   f=2.4
   cam.CreateProjectionAttr('perspective');cam.CreateFocalLengthAttr(f)
   cam.CreateHorizontalApertureAttr(f*K['width']/K['fx']);cam.CreateVerticalApertureAttr(f*K['height']/K['fy']);cam.CreateClippingRangeAttr(Gf.Vec2f(.025,3.0))
   cp=cam.GetPrim()
   req('OmniSensorAPI' in cp.GetAppliedSchemas() or cp.AddAppliedSchema('OmniSensorAPI'),'OmniSensorAPI unavailable')
   if nm=='right_wrist_view':author(cam)
   sensor=CameraSensor(RtxCamera(path=str(cp.GetPath()),tick_rate=0.,reset_xform_op_properties=False),resolution=(K['height'],K['width']),annotators=['rgb'])
   if nm=='right_wrist_view':validate_authored(cam)
   stage.SetEditTarget(stage.GetSessionLayer())
   for prim,q in [(xf,Gf.Quatd(0.,Gf.Vec3d(1.,0.,0.))),(cp,Gf.Quatd(1.,Gf.Vec3d(0.,0.,0.)))]:
    xp=UsdGeom.Xformable(prim);opn=xp.AddOrientOp(UsdGeom.XformOp.PrecisionDouble,'wallD1Axis')
    req(opn.Set(q) and xp.SetXformOpOrder([opn],resetXformStack=False),'camera axis authoring failed')
   hit=[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdRender.Product) and cp.GetPath() in UsdRender.Product(p).GetCameraRel().GetTargets()]
   req(len(hit)==1,'missing native-camera-associated render product for '+nm)
   self.views[nm]=View(op,cp,sensor,hit[0])
  req(len({x.product for x in self.views.values()})==3,'duplicate render products')
 def capture(self,tick):
  # Caller owns the single actual world.step. No render or physics update in this reads-only hook.
  req(int(self.world.current_time_step_index)==tick,'capture called on the wrong active tick')
  meta={};data={}; stamps={}; B=pose(self.links[BASE])
  for k,v in self.views.items():
   t, u=pose(v.frame),pose(v.cam); req(np.linalg.norm(u-t@np.diag([1,-1,-1,1]))<2e-5,'native render axes drift '+k)
   pic,info=v.sensor.get_data('rgb');req(pic is not None,'cold native render '+k)
   arr=np.asarray(pic.numpy() if hasattr(pic,'numpy') else pic)
   req(arr.dtype==np.uint8 and arr.ndim==3 and arr.shape[:2]==(K['height'],K['width']) and arr.shape[2] in (3,4),'invalid native RGB '+k)
   arr=np.ascontiguousarray(arr[:,:,:3]);req(k=='left_wrist_view' or float(arr.std())>2,'blank model input RGB '+k)
   info=info or {};stamps[k]={key:float(info[key]) for key in ('renderingFrame','renderingTime','simulationTime') if key in info and isinstance(info[key],(float,int,np.number))}
   actual=info.get('renderProductPath',info.get('renderProduct'))
   req(not isinstance(actual,str) or actual==v.product,'reported render product mismatch '+k)
   meta[k]={'original_link':str(v.frame.GetPath()),'camera':str(v.cam.GetPath()),'render_product':v.product,'world_T_ros_optical':t.tolist(),'K':K if k!='right_wrist_view' else k_record(),'sha256_RGB':hashlib.sha256(arr.tobytes()).hexdigest()};data[k]=arr
   if k=='right_wrist_view':
    from pxr import UsdGeom
    validate_authored(UsdGeom.Camera(v.cam))
    meta[k]['optics']=recipe()
    meta[k]['image_transform']='native RTX render buffer; no crop, reproject or resample'
  req(len({x['sha256_RGB'] for x in meta.values()})==3,'duplicate view bytes')
  same=[]
  for field in ('renderingFrame','simulationTime','renderingTime'):
   vals=[v[field] for v in stamps.values() if field in v]
   req(not vals or np.isfinite(vals).all() and max(vals)-min(vals)<1e-6,'different renderer stamps '+field)
   if len(vals)==3:same.append(field)
  q=np.asarray(self.art.get_joint_positions())
  req(q.ndim==2 and q.shape[0]==1 and np.isfinite(q).all(),'invalid measured articulation states')
  req(int(self.world.current_time_step_index)==tick,'capture advanced the active tick')
  return data,{'physics_tick':tick,'clock':'Isaac native PhysX tick / numerical render metadata only when provided','camera_reports':meta,'camera_stamps':stamps,'common_renderer_timestamp_fields':same,'synchrony':'renderer-metadata verified' if same else 'bounded single read-only physics tick; renderer timestamps absent or incomplete',
    'measured_encoder':dict(zip(DOF,q[0,self.ix].astype(float).tolist())),
    'chassis_T_lift_native':(np.linalg.inv(B)@pose(self.links['lift_link'])).tolist(),
    'chassis_T_gripper_base_candidate':{s:(np.linalg.inv(B)@pose(self.links[e])).tolist() for s,e in EES.items()},
    'ee_aligned_to_sdk_contract':False,'gripper_model_units_validated':False,'motion_authorized':False}
def emit(target,frames,rec):
 target=Path(target);target.mkdir(parents=True,exist_ok=True);req(not any(target.iterdir()),'native output dir must be empty')
 np.savez_compressed(target/'native_rgb.npz',**frames)
 rec['npz_sha256']=hashlib.sha256((target/'native_rgb.npz').read_bytes()).hexdigest()
 (target/'observation.json').write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
