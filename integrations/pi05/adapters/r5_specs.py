"""Wire metadata distinguishes normalized OSC controls from physical metres/radians."""
def action_spec():
    from vla_eval.specs import DimSpec,GRIPPER_CLOSE_POS
    return {
        'position':DimSpec('position',3,'osc_delta_xyz_control',(-1,1),description='20Hz controller input; each unit maps to 0.05m before native controller clipping'),
        'rotation':DimSpec('rotation',3,'osc_delta_rotation_control',(-1,1),description='20Hz controller input; each unit maps to 0.5rad; not an absolute axis-angle pose'),
        'gripper':GRIPPER_CLOSE_POS,
    }
