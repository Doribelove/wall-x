"""Native Isaac Sim 6.0.1 RTX authoring for the vendor's nominal 5756 wrist, no simulation launch."""
import math
from hashlib import sha1
from pathlib import Path

SCHEMA='OmniLensDistortionOpenCvFisheyeAPI'
EXTENSION='omni.usd.schema.omni_lens_distortion'
START_EXTS=(EXTENSION,'isaacsim.sensors.experimental.rtx')
VENDOR_BLOB='12dba417ae4e73f3b90b32b0ea50e89789246dc9'
W,H=640,480
H_SPEC,V_SPEC,D_SPEC=137.2,101.,179.2
FX=FY=math.hypot(W/2,H/2)/math.radians(D_SPEC/2)
CX,CY=W/2,H/2
K=(0.,0.,0.,0.)
CLIP=(.01,10.)
EFL_CM=.25
PREFIX='omni:lensdistortion:opencvFisheye:'

def require(ok,msg):
    if not ok:raise RuntimeError(msg)

def add_startup_extension_args(argv):
    """Call after application argparse but BEFORE constructing SimulationApp."""
    out=list(argv)
    for name in START_EXTS:
        if not any(a=='--enable' and b==name for a,b in zip(out,out[1:])):
            out.extend(('--enable',name))
    return out

def blob(path):
    b=Path(path).read_bytes()
    return sha1(b'blob '+str(len(b)).encode()+b'\x00'+b).hexdigest()

def recipe():
    return {'model':'opencvFisheye','schema':SCHEMA,'width':W,'height':H,
      'fx':FX,'fy':FY,'cx':CX,'cy':CY,'k1_k4':K,'clip_m':CLIP,
      'raster_edge_fov_deg':{'horizontal':math.degrees(W/FX),'vertical':math.degrees(H/FY),'diagonal':D_SPEC},
      'datasheet_nominal_fov_deg':{'horizontal':H_SPEC,'vertical':V_SPEC,'diagonal':D_SPEC},
      'vendor_git_blob_sha1':VENDOR_BLOB,
      'scope':'x2real 5756 nominal simulation intrinsics, not physical per-camera calibration'}

def k_record():
    return {'width':W,'height':H,'fx':FX,'fy':FY,'cx':CX,'cy':CY,
      'status':'nominal vendor x2real 5756 ideal equidistant in native RTX; not a calibrated physical camera'}

def _lens_attrs(gf):
    return {'omni:lensdistortion:model':'opencvFisheye',
      **{PREFIX+n:x for n,x in [('fx',FX),('fy',FY),('cx',CX),('cy',CY)]},
      PREFIX+'imageSize':gf.Vec2i(W,H),**{PREFIX+f'k{i}':x for i,x in enumerate(K,1)}}

def author(camera):
    """Applies the documented native model to a newly defined session Camera, before render product."""
    from pxr import Gf, UsdGeom
    prim=camera.GetPrim()
    require(prim.IsA(UsdGeom.Camera),'only a native USD Camera can receive the 5756 lens')
    require(prim.GetStage().GetEditTarget().GetLayer()==prim.GetStage().GetSessionLayer(),
            'do not change root asset or vendor optical mount')
    other={str(s) for s in prim.GetAppliedSchemas() if str(s).startswith('OmniLensDistortion')}
    require(other in (set(),{SCHEMA}),'refusing conflicting lens schemas')
    require(prim.ApplyAPI(SCHEMA) and prim.HasAPI(SCHEMA),
            'native lens schema unavailable; enable '+EXTENSION+' at Kit boot, no pinhole fallback')
    for name,value in _lens_attrs(Gf).items():
        attr=prim.GetAttribute(name)
        require(bool(attr) and attr.Set(value),'schema attribute missing or not set: '+name)
    # Nominal 2.5mm film gate as in vendor convert_camera_intrinsics_to_usd(.25).
    # Native lens schema, not these film-gate ratios, defines fisheye ray projection.
    for attr,x in ((camera.GetFocalLengthAttr(),EFL_CM),
                   (camera.GetHorizontalApertureAttr(),EFL_CM*W/FX),
                   (camera.GetVerticalApertureAttr(),EFL_CM*H/FY),
                   (camera.GetClippingRangeAttr(),Gf.Vec2f(*CLIP))):
        require(attr.Set(x),'native camera optics could not be authored: '+str(attr))
    validate_authored(camera)
    return recipe()

def validate_authored(camera):
    """A valid schema is necessary, not sufficient to claim actual rays. Use the rendered chart."""
    from pxr import Gf
    prim=camera.GetPrim()
    require(prim.HasAPI(SCHEMA),'native fisheye schema not present on camera')
    for name,want in _lens_attrs(Gf).items():
        attr=prim.GetAttribute(name)
        require(attr and attr.HasAuthoredValueOpinion(),'native lens attr not authored: '+name)
        actual=attr.Get()
        good=(math.isclose(actual,want,rel_tol=1e-7,abs_tol=2e-5) if isinstance(want,float) else actual==want)
        require(good,'native lens disagrees with vendor parameters: '+name)
    clips=camera.GetClippingRangeAttr().Get()
    require(len(clips)==2 and all(math.isclose(x,y,abs_tol=1e-7) for x,y in zip(clips,CLIP)),
            'unexpected wrist clipping after passing through RTX camera wrapper')
    return True

def project_optical(ray):
    """Analytic no-render calibration preview; optical +X right, +Y down, +Z forward."""
    x,y,z=map(float,ray)
    if not all(map(math.isfinite,(x,y,z))) or z<=0:raise ValueError('forward finite optical ray required')
    v=math.hypot(x,y)
    if not v:return CX,CY
    th=math.atan2(v,z);return CX+FX*th*x/v,CY+FY*th*y/v

if __name__=='__main__':
    import json
    print(json.dumps(recipe(),indent=2))
