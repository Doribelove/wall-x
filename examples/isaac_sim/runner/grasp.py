"""Contact-region grasp constraints for simulated grippers, with explicit engagement and release events."""
import numpy as np
from scipy.spatial.transform import Rotation

class Grasp:
    def __init__(self,st,robot_bodies,prop_prims,task,cloth=None):
        self.st=st;self.bodies=robot_bodies;self.props=prop_prims
        self.task=task;self.held={};self.nodes={};self.events=[];self.cloth=cloth
    def patch_cloth(self,phase,side,W,gap,jaws,t):
        from pxr import Gf,Vt
        grip=gap<.010 and phase in ("pinch_first_edge","pinch_second_edge","grip_folded_stack","pinch_supported","secure_corners","pinch_hinge")
        augment=phase in ("secure_corners","grip_folded_stack")
        if side in self.nodes and gap>.016:
            parts=self.nodes.pop(side);ids=[i for prim,g in parts for i in g]
            for prim,group in parts:prim.GetAttribute("omniphysics:attachmentEnabled").Set(False)
            self.events.append(dict(event="release_nodes",side=side,time=t,count=len(ids)))
        if (side not in self.nodes or augment) and grip and max(abs(jaws))<.012:
            pv=self.cloth.get_simulation_nodal_positions()
            pv=pv.detach().cpu() if hasattr(pv,"detach") else pv
            pts=np.asarray(pv.numpy() if hasattr(pv,"numpy") else pv)[0]
            local=(pts-W[:3,3])@W[:3,:3]
            ok=(np.abs(local[:,1])<=max(.012,max(abs(jaws))+.006)) & (np.abs(local[:,2])<=.013) & (local[:,0]>=.139) & (local[:,0]<=.169)
            existing={i for obj,group in self.nodes.get(side,[]) for i in group}
            ids=[i for i in np.nonzero(ok)[0].tolist() if i not in existing]
            if ids:
                prim=self.st.DefinePrim("/World/FingerNodeAttachment"+str(len(self.events)), "OmniPhysicsVtxXformAttachment")
                prim.GetAttribute("omniphysics:vtxIndicesSrc0").Set(Vt.IntArray(ids))
                prim.GetAttribute("omniphysics:localPositionsSrc1").Set(Vt.Vec3fArray([Gf.Vec3f(*map(float,a)) for a in local[ids]]))
                prim.GetRelationship("omniphysics:src0").SetTargets([self.props["D4Coupon/Cloth"].GetPath()])
                prim.GetRelationship("omniphysics:src1").SetTargets([self.bodies[side+"_arm_gripper_base_link"].GetPath()])
                self.nodes.setdefault(side,[]).append((prim,ids))
                self.events.append(dict(event="pinch_nodes",side=side,time=t,phase=phase,count=len(ids),indices=ids))
    def apply(self,phase,goals,wrists,measured,t):
        if self.cloth is not None:
            for i,side in enumerate(("left","right")):
                self.patch_cloth(phase,side,wrists[side],goals[side][1],measured[12+2*i:14+2*i],t)
            return
        # Only capture a geometric grasp inside opposed finger pads. D1's pinch remains pure friction.
        lookup={"D2":{"left":("Board",(0.,.180,0.),.0395,.045),"right":("EraserProxy",(0.,0.,.014),.012,.020)},
                "D3":{"left":("Bottle",(0.,0.,0.),.028,.035),"right":("Cap",(0.,0.,.004),.022,.030)},
                "D5":{"left":("BenchAssembly/FemaleShell",(0.,0.,.025),.010,.020),
                      "right":("BenchAssembly/ServiceDrive",(0.,0.,-.018),.008,.016)}}
        if self.task not in lookup:return
        from pxr import UsdPhysics,Gf
        from native_capture_h_5756 import pose
        for i,side in enumerate(("left","right")):
            if side not in lookup[self.task]:continue
            name,offset,radius,release=lookup[self.task][side]
            gap=goals[side][1];jaws=measured[12+2*i:14+2*i]
            if side in self.held:
                if gap>=release:
                    self.held.pop(side).GetJointEnabledAttr().Set(False)
                    self.events.append(dict(event="release",side=side,object=name,time=t,phase=phase))
                continue
            if gap>=release or max(abs(jaws))>radius+(.018 if self.task in ("D3","D5") else .008):continue
            if self.task=="D2" and side=="left" and max(abs(jaws))>.0415: continue
            O=pose(self.props[name]);W=wrists[side]
            local=W[:3,:3].T@(O[:3,3]+O[:3,:3]@offset-W[:3,3])
            if abs(local[1]-np.mean(jaws))>min(radius+.003,.016) or abs(local[2])>.017 or abs(local[0]-.146)>.024:continue
            prim=self.bodies[side+"_arm_gripper_base_link"]
            trueW=pose(prim);rel=np.linalg.inv(trueW)@O
            q=Rotation.from_matrix(rel[:3,:3]).as_quat()
            path="/World/TaskGrip_"+side+"_"+str(len(self.events))
            joint=UsdPhysics.FixedJoint.Define(self.st,path)
            joint.CreateExcludeFromArticulationAttr(True)
            joint.CreateLocalPos0Attr(Gf.Vec3f(*rel[:3,3].tolist()))
            joint.CreateLocalRot0Attr(Gf.Quatf(float(q[3]),Gf.Vec3f(*q[:3].tolist())))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0,0,0));joint.CreateLocalRot1Attr(Gf.Quatf(1.,Gf.Vec3f(0,0,0)))
            joint.CreateBreakForceAttr(65.);joint.CreateBreakTorqueAttr(2.5)
            joint.CreateBody0Rel().SetTargets([prim.GetPath()]);joint.CreateBody1Rel().SetTargets([self.props[name].GetPath()])
            self.held[side]=joint
            self.events.append(dict(event="engage_grasp_region",side=side,object=name,time=t,phase=phase,pad_frame_center=local.tolist(),measured_jaws_m=jaws.tolist()))
