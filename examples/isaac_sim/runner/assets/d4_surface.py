"""D4 topology preflight + isolated native USD authoring. CLI is CPU-only, no Isaac launch."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXACT_OBJ = 'd986f9c9d793e49b89ba228eef4311c2902758bebb98d8c86cc4f56d0526b7b7'
OBJ = HERE/'towel.obj'
CLOTH = '/World/D4Coupon/Cloth'
TABLE = '/World/D4Coupon/Table'
SCENE = '/World/PhysicsScene'

@dataclass(frozen=True)
class Candidate:
    # Development coordinates only; not measured cotton parameters.
    # 285 g/m2 × .300 × .300 m²; measured specimen mass overrides after recording.
    mass_kg: float = .02565
    thickness_m: float = .002
    young_pa: float = 200000.
    poisson: float = .30
    bend_pa: float = 18000.
    mu_dynamic: float = .60
    cloth_rest_m: float = .0006
    cloth_contact_m: float = .0020
    table_rest_m: float = .0
    table_contact_m: float = .0015
    solver_positions: int = 12
    hz: int = 250
    gravity_z: float = -9.81
    table_top_m: float = .72
    cloth_drop_gap_m: float = .018
    overhang_m: float = .042

def need(ok, msg):
    if not ok: raise ValueError(msg)

def audited_mesh(path=OBJ):
    b=path.read_bytes()
    need(sha256(b).hexdigest()==EXACT_OBJ,'not the frozen D4 rest mesh')
    p=[]; f=[]
    for l in b.decode().splitlines():
        z=l.split()
        if not z: continue
        if z[0]=='v':
            need(len(z)==4,'not xyz')
            p.append(tuple(map(float,z[1:])))
        elif z[0]=='f':
            need(len(z)==4,'not triangles')
            f.append(tuple(int(a)-1 for a in z[1:]))
        else: raise ValueError('unexpected OBJ content')
    need(len(set(p))==len(p)==225 and len(f)==len({frozenset(x) for x in f})==392,'frozen topology mismatch')
    low=tuple(min(q) for q in zip(*p)); high=tuple(max(q) for q in zip(*p))
    need(low==(-.15,-.15,0.) and high==(.15,.15,0.),'not the original 300 mm square')
    outer=Counter(); incidence=defaultdict(list); links=defaultdict(set); area=0.; sign=None
    for tri in f:
        need(len(set(tri))==3 and min(tri)>=0 and max(tri)<len(p),'bad triangle vertex')
        a,b,c=[p[i] for i in tri]
        twice=(b[0]-a[0])*(c[1]-a[1])-(c[0]-a[0])*(b[1]-a[1])
        need(twice and (sign is None or (twice>0)==sign),'zero area or mixed triangle winding')
        sign=twice>0; area+=abs(twice)/2
        for i,j in zip(tri,(*tri[1:],tri[0])):
            pair=tuple(sorted((i,j))); incidence[pair].append((i,j)); links[i].add(j); links[j].add(i)
    for e,d in incidence.items():
        need(len(d)==1 or len(d)==2 and d[0]==d[1][::-1],'nonmanifold or mismatched winding')
        if len(d)==1: outer.update(e)
    seen={0}; trail=[0]
    while trail:
        for v in links[trail.pop()]-seen: seen.add(v); trail.append(v)
    need(len(seen)==len(p) and len(outer)==56 and set(outer.values())=={2} and
         len(p)-len(incidence)+len(f)==1 and abs(area-.09)<1e-9,'not a connected single sheet')
    corners=[p.index((x,y,0.)) for x,y in ((-.15,-.15),(.15,-.15),(.15,.15),(-.15,.15))]
    return p,f,{'obj_sha256':EXACT_OBJ,'length_width_m':[.3,.3],'vertices':len(p),'triangles':len(f),
                'edges':len(incidence),'boundary_vertices':len(outer),'surface_area_m2':area,
                'rest_corner_vertex_ids_ccw':corners,'physical_run_performed':False}

def recipe(candidate=Candidate(),path=OBJ):
    cfg=asdict(candidate)
    need(all(math.isfinite(v) for v in cfg.values()),'non-finite calibration candidate')
    need(0<candidate.mass_kg<.2 and 0<candidate.thickness_m<.01 and
         0<candidate.young_pa and 0<candidate.bend_pa and 0<candidate.poisson<.5 and
         0<candidate.mu_dynamic and 0<=candidate.cloth_rest_m<candidate.cloth_contact_m and
         0<=candidate.table_rest_m<candidate.table_contact_m and
         candidate.solver_positions>0 and candidate.hz in (250,500) and
         candidate.gravity_z==-9.81 and candidate.cloth_drop_gap_m>candidate.cloth_contact_m+candidate.table_contact_m and
         0<candidate.overhang_m<.15,'inadmissible development candidate')
    p,f,top=audited_mesh(path)
    rest_center_x=.15-candidate.overhang_m
    shift=(rest_center_x,0.,candidate.table_top_m+candidate.cloth_drop_gap_m)
    top.update({'cloth_path':CLOTH,'table_path':TABLE,'physics_scene_path':SCENE,
                'cloth_initial_xform_translation_m':shift,'table_top_m':candidate.table_top_m,
                'table_x_span_m':[0.,.66],'footprint_supported_x_fraction_at_spawn':(.3-candidate.overhang_m)/.3,
                'solver_scope':'GPU PhysX surface deformable API design; native parse and runtime not tested',
                'claim_scope':'isolated gravity/contact coupon spawn only; not a D4 motion or video','candidate_not_calibrated':cfg})
    return p,f,top

def author_coupon(stage, *, candidate=Candidate(), obj=OBJ):
    """Author ONLY on an already-open new empty stage with Isaac 6.0.1 schemas available.

    Does not create an app or advance time, does not attach, pin, drive, animate or override any vertex.
    A future operator must verify effective GPU scene/runtime independently.
    """
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt
    p, f, plan=recipe(candidate,obj)
    need(not any(stage.Traverse()),'fresh empty isolated stage only; do not patch a robot or task scene')
    def set_attr(attr,value):
        need(bool(attr) and attr.Set(value),'missing/incompatible schema attribute: '+str(attr))
    def api(prim,name):
        need(prim.ApplyAPI(name) and prim.HasAPI(name),'6.0.1 schema not registered: '+name)
    def attrs(prim,values):
        for name,val in values.items(): set_attr(prim.GetAttribute(name),val)
    UsdGeom.SetStageMetersPerUnit(stage,1.); UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    world=UsdGeom.Xform.Define(stage,'/World')
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(stage,'/World/D4Coupon')

    scene=UsdPhysics.Scene.Define(stage,SCENE)
    set_attr(scene.CreateGravityDirectionAttr(),Gf.Vec3f(0.,0.,-1.))
    set_attr(scene.CreateGravityMagnitudeAttr(),-candidate.gravity_z)
    ph=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    set_attr(ph.CreateEnableGPUDynamicsAttr(),True)
    set_attr(ph.CreateBroadphaseTypeAttr(),'GPU')
    set_attr(ph.CreateTimeStepsPerSecondAttr(),candidate.hz)

    def collider(prim,rest,offset):
        need(bool(UsdPhysics.CollisionAPI.Apply(prim)),'cannot apply collision')
        col=PhysxSchema.PhysxCollisionAPI.Apply(prim)
        set_attr(col.CreateRestOffsetAttr(),rest);set_attr(col.CreateContactOffsetAttr(),offset)

    table=UsdGeom.Cube.Define(stage,TABLE)
    set_attr(table.CreateSizeAttr(),1.)
    set_attr(table.AddTranslateOp(),Gf.Vec3d(.33,0.,candidate.table_top_m-.027))
    set_attr(table.AddScaleOp(),Gf.Vec3f(.66,.65,.054))
    set_attr(table.CreateDisplayColorAttr(),Vt.Vec3fArray([(.22,.24,.28)]))
    collider(table.GetPrim(),candidate.table_rest_m,candidate.table_contact_m)

    mesh=UsdGeom.Mesh.Define(stage,CLOTH); prim=mesh.GetPrim()
    points=Vt.Vec3fArray(p)
    set_attr(mesh.CreatePointsAttr(),points)
    set_attr(mesh.CreateFaceVertexCountsAttr(),Vt.IntArray([3]*len(f)))
    set_attr(mesh.CreateFaceVertexIndicesAttr(),Vt.IntArray([v for tri in f for v in tri]))
    set_attr(mesh.CreateSubdivisionSchemeAttr(),UsdGeom.Tokens.none)
    set_attr(mesh.CreateDoubleSidedAttr(),True)
    set_attr(mesh.AddTranslateOp(),Gf.Vec3d(*plan['cloth_initial_xform_translation_m']))
    set_attr(mesh.CreateDisplayColorAttr(),Vt.Vec3fArray([(.89,.89,.84)]))
    api(prim,'OmniPhysicsDeformableBodyAPI')
    api(prim,'OmniPhysicsSurfaceDeformableSimAPI')
    collider(prim,candidate.cloth_rest_m,candidate.cloth_contact_m)
    api(prim,'PhysxSurfaceDeformableBodyAPI')
    attrs(prim,{'omniphysics:mass':candidate.mass_kg,
                'omniphysics:restShapePoints':points,'omniphysics:restTriVtxIndices':Vt.Vec3iArray(f),
                'physxDeformableBody:disableGravity':False,'physxDeformableBody:selfCollision':True,
                'physxDeformableBody:solverPositionIterationCount':candidate.solver_positions})

    mat=UsdShade.Material.Define(stage,'/World/D4Coupon/Material')
    mp=mat.GetPrim()
    for name in ('OmniPhysicsBaseMaterialAPI','OmniPhysicsDeformableMaterialAPI','OmniPhysicsSurfaceDeformableMaterialAPI'):api(mp,name)
    attrs(mp,{'omniphysics:dynamicFriction':candidate.mu_dynamic,
              'omniphysics:density':candidate.mass_kg/(plan['surface_area_m2']*candidate.thickness_m),
              'omniphysics:youngsModulus':candidate.young_pa,'omniphysics:poissonsRatio':candidate.poisson,
              'omniphysics:surfaceThickness':candidate.thickness_m,
              'omniphysics:surfaceBendStiffness':candidate.bend_pa})
    need(UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,materialPurpose='physics'),'material not bound')
    return plan

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--obj',type=Path,default=OBJ)
    cli.add_argument('--out',type=Path)
    args=cli.parse_args()
    js=json.dumps(recipe(path=args.obj)[2],indent=2,allow_nan=False)+'\n'
    if args.out:
        with args.out.open('x') as stream: stream.write(js)
    else: print(js,end='')
