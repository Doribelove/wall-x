import json
import os
import sys
from pathlib import Path

import numpy as np


def scene_file(task, assets, out):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    if task == "D1":
        return assets/"D1_B_clutter_fixtures.usda"
    if task in ("D2", "D3"):
        return assets/(task+".usda")
    path=out/"fixture.usda"
    if task == "D4":
        sys.path.insert(0,str(assets))
        from d4_surface import author_coupon, Candidate, CLOTH, TABLE
        st=Usd.Stage.CreateNew(str(path))
        setup=os.getenv("FOLD_CASE","reference")
        candidate=Candidate(table_top_m=.66) if setup=="reference" else Candidate(table_top_m=.66,bend_pa=550.,mu_dynamic=1.05,solver_positions=24)
        author_coupon(st,candidate=candidate,obj=assets/"towel.obj")
        if setup!="reference":
            from pxr import UsdShade
            mat=UsdShade.Material.Define(st,"/World/TableRubber")
            api=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            api.CreateStaticFrictionAttr(1.40);api.CreateDynamicFrictionAttr(1.05);api.CreateRestitutionAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(st.GetPrimAtPath(TABLE)).Bind(mat,materialPurpose="physics")
        cloth=UsdGeom.Mesh(st.GetPrimAtPath(CLOTH))
        cloth.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(.465,-.025,.678))
        cloth.CreateDisplayColorAttr().Set([Gf.Vec3f(.24,.49,.80)])
        colors=[Gf.Vec3f(.24,.49,.80) for _ in range(len(cloth.GetPointsAttr().Get()))]
        for i,v in enumerate(cloth.GetPointsAttr().Get()):
            if v[0]>.11 and v[1]<-.11:
                colors[i]=Gf.Vec3f(.96,.45,.09)
        cloth.GetDisplayColorPrimvar().Set(colors)
        cloth.GetDisplayColorPrimvar().SetInterpolation("vertex")
        UsdGeom.Cube(st.GetPrimAtPath(TABLE)).GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(.47,0,.633))
        box=UsdGeom.Cube.Define(st,"/World/StorageMarker")
        box.CreateSizeAttr(1.);box.AddTranslateOp().Set((.42,-.01,.6606) if setup in ("supported","crease") else (.419,.095,.6606) if setup.startswith("carry_corners") else (.43,.116,.6606) if setup.startswith("carry") else (.435,.14,.6606));box.AddScaleOp().Set((.13,.13,.0005))
        box.CreateDisplayColorAttr().Set([(.25,.65,.55)])
    else:
        st=Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(st,UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(st,1.)
        root=UsdGeom.Xform.Define(st,"/World")
        st.SetDefaultPrim(root.GetPrim())
        # Compose each native assembly including all cable links and joint relationships.
        g=UsdGeom.Xform.Define(st,"/World/BenchAssembly")
        g.GetPrim().GetReferences().AddReference(str(assets/"D5.usda"),Sdf.Path("/World"))
        offset=Gf.Vec3d(.46,-.055,.65)
        for body in g.GetPrim().GetChildren():
            attr=body.GetAttribute("xformOp:translate")
            if attr and attr.Get() is not None:
                attr.Set(Gf.Vec3d(*attr.Get())+offset)
        table=st.GetPrimAtPath("/World/BenchAssembly/Table")
        table.GetAttribute("xformOp:scale").Set(Gf.Vec3d(.74,.74,.024))
        st.GetPrimAtPath("/World/BenchAssembly/Physics").SetActive(False)
        male=st.OverridePrim("/World/BenchAssembly/ServiceDrive")
        male.GetReferences().AddReference(str(assets/"D5_male.usda"),Sdf.Path("/World/Male"))
        male.GetAttribute("xformOp:translate").Set(Gf.Vec3d(.411,-.199,.660))
        xf=UsdGeom.Xformable(male)
        from scipy.spatial.transform import Rotation
        # Male and socket are free bodies. Start the drive in its own supply area with a small yaw offset.
        socket_transform=st.GetPrimAtPath("/World/BenchAssembly/FemaleShell").GetAttribute("xformOp:orient").Get()
        q0=list(socket_transform.GetImaginary())+[socket_transform.GetReal()]
        rr=Rotation.from_euler("z",.16)*Rotation.from_quat(q0)
        q=rr.as_quat()
        orient=xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble,opSuffix="tabletopStart")
        orient.Set(Gf.Quatd(float(q[3]),Gf.Vec3d(*q[:3])))
        xf.SetXformOpOrder([next(op for op in xf.GetOrderedXformOps() if op.GetOpType()==UsdGeom.XformOp.TypeTranslate),orient])

        scene=UsdPhysics.Scene.Define(st,"/World/Physics")
        scene.CreateGravityDirectionAttr((0.,0.,-1.));scene.CreateGravityMagnitudeAttr(9.81)
    st.GetRootLayer().Save()
    return path


def configure_stage(st,task):
    from pxr import UsdGeom,UsdLux,UsdPhysics,Gf
    floor=UsdGeom.Cube.Define(st,"/World/StudioFloor")
    floor.CreateSizeAttr(1.);floor.AddTranslateOp().Set((.4,0,-.11));floor.AddScaleOp().Set((6.,6.,.025))
    floor.CreateDisplayColorAttr().Set([(.19,.23,.27)])
    UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
    if task!="D1":
        dome=UsdLux.DomeLight.Define(st,"/World/StudioDome")
        dome.CreateIntensityAttr(700.)
        light=UsdLux.DistantLight.Define(st,"/World/StudioKey")
        light.CreateIntensityAttr(2000.)
        light.AddRotateXYZOp().Set((24.,-37.,-32.))
        if task!="D5":
            z={"D2":.710,"D3":.615,"D4":.633}[task]
            for i,(x,y) in enumerate(((.37,-.27),(.60,-.27),(.37,.21),(.60,.21))):
                leg=UsdGeom.Cube.Define(st,f"/World/StudioTableLeg{i}")
                leg.CreateSizeAttr(1.);leg.AddTranslateOp().Set((x,y,z/2-.05));leg.AddScaleOp().Set((.022,.022,z+.04))
                leg.CreateDisplayColorAttr().Set([(.23,.29,.31)])
    # Capsule/shape observations are reserved to the independent evaluator.
    return {"D1":["red0","red1","red2","blue3","blue4","green5","mint","yellow","yellow_neighbor"],
            "D2":["Board","EraserProxy"],"D3":["Cap","Bottle"],"D4":["D4Coupon/Cloth"],
            "D5":["BenchAssembly/ServiceDrive","BenchAssembly/FemaleShell","BenchAssembly/DistalMale"]}[task]


def film_camera(st,task):
    from pxr import UsdGeom,Gf
    from isaacsim.sensors.experimental.rtx import RtxCamera,CameraSensor
    pos=np.array([1.34,-.85,1.34]); center=np.array([.34,-.045,.77])
    if task=="D1":
        center=np.array([.37,-.19,.73])
    if task=="D5":
        pos=np.array([1.03,-.62,1.17]);center=np.array([.39,-.095,.74])
    prim=UsdGeom.Camera.Define(st,"/World/RunFilm")
    prim.CreateFocalLengthAttr(2.4);prim.CreateHorizontalApertureAttr(3.10);prim.CreateVerticalApertureAttr(1.744)
    prim.CreateClippingRangeAttr((.02,8.))
    cp=prim.GetPrim();cp.AddAppliedSchema("OmniSensorAPI")
    sensor=CameraSensor(RtxCamera(path="/World/RunFilm",tick_rate=0.,reset_xform_op_properties=False),
                        resolution=(480,852),annotators=["rgb"])
    st.SetEditTarget(st.GetSessionLayer())
    xf=UsdGeom.Xformable(cp)
    matrix=Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pos),Gf.Vec3d(*center),Gf.Vec3d(0,0,1)).GetInverse()
    op=xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble,"observer")
    op.Set(matrix);xf.SetXformOpOrder([op],resetXformStack=False)
    return sensor


def camera_rgb(sensor):
    data, _ = sensor.get_data("rgb")
    if data is None:
        raise RuntimeError("empty external camera")
    arr=np.asarray(data.numpy() if hasattr(data,"numpy") else data)
    if arr.shape!=(480,852,4) and arr.shape!=(480,852,3):
        raise RuntimeError("observer dimensions " + str(arr.shape))
    return np.ascontiguousarray(arr[:,:,:3])
