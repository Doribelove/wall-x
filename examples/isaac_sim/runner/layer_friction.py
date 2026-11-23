"""A tangential static-friction approximation for unresolved cloth-on-cloth contact during a gripped transfer."""
import os
import numpy as np
from scipy.spatial import cKDTree

class LayerFriction:
    def __init__(self,view):
        self.view=view;self.pairs=None;self.done=False;self.active=False;self.steps=0;self.peak_delta=0.;self.imbalance=0.
    @staticmethod
    def host(x):
        if hasattr(x,'detach'):return x.detach().cpu().numpy()
        return x.numpy() if hasattr(x,'numpy') else np.asarray(x)
    def stage(self,name,grips):
        if self.done:return
        if name=='release_in_storage':
            self.active=False;self.done=True;return
        early=os.environ.get('FOLD_CASE')=='carry_corners_friction'
        start=(name=='survey_stack') if early else (name=='grip_folded_stack' and set(grips)=={'left','right'})
        if not start or self.active:return
        p=self.host(self.view.get_simulation_nodal_positions())[0]
        pairs=cKDTree(p[:,:2]).query_pairs(.013,output_type='ndarray')
        i,j=pairs.T
        far=(np.abs(i//15-j//15)>3)|(np.abs(i%15-j%15)>3)
        contact=np.abs(p[i,2]-p[j,2])<.0045
        self.pairs=pairs[far&contact]
        if not len(self.pairs):return
        i,j=self.pairs.T
        self.rest=p[i,:2]-p[j,:2];self.active=True
    def step(self,dt):
        if not self.active:return
        import warp as wp
        p=self.host(self.view.get_simulation_nodal_positions())[0]
        vf=self.view.get_simulation_nodal_velocities();v=self.host(vf).copy()
        i,j=self.pairs.T
        slip=p[i,:2]-p[j,:2]-self.rest
        keep=(np.abs(p[i,2]-p[j,2])<.008)&(np.linalg.norm(slip,axis=1)<.016)
        if not np.any(keep):return
        i,j=i[keep],j[keep];slip=slip[keep]
        # The two sides receive equal/opposite tangential impulses, limited by mu*g*dt.
        impulse=-.36*(v[0,i,:2]-v[0,j,:2])-1800.*dt*slip
        impulse*=np.minimum(1.,1.05*9.81*dt/np.maximum(np.linalg.norm(impulse,axis=1),1.e-9))[:,None]
        degree=np.bincount(np.r_[i,j],minlength=len(p))
        impulse/=np.maximum(degree[i],degree[j])[:,None]
        dv=np.zeros((len(p),2));np.add.at(dv,i,impulse);np.add.at(dv,j,-impulse)
        self.imbalance=max(self.imbalance,float(np.linalg.norm(dv.sum(axis=0))))
        self.peak_delta=max(self.peak_delta,float(np.linalg.norm(dv,axis=1).max()));self.steps+=1
        v[0,:,:2]+=dv
        data=wp.from_numpy(np.ascontiguousarray(v,dtype=np.float32),dtype=wp.float32,device='cuda:0')
        index=wp.array([0],dtype=wp.uint32,device='cuda:0')
        self.view.set_simulation_nodal_velocities(data,index)
