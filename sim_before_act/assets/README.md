# assets

`scene_demo/` holds example OUTPUTS of `python -m sim_before_act scene-rollout --fixture small_motion --seed 3 --perturb --render --video`
(summary JSON, trace npz, cv2 plot, MuJoCo contact sheet, MP4). They come from an INVENTED synthetic fixture in the SYNTHETIC DEMO SCENE
(invented values, CAD geometry; not a calibrated twin); they are not policy output and not recorded data.

The scene itself is composed in memory by `twin/scene/builder.py` from the read-only `so101_mujoco/so101_new_calib.xml`; no model XML is copied or modified here.
