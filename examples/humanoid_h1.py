from pathlib import Path
from typing import Optional, Sequence

import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

import numpy as np
from simpub.sim.mj_publisher import MujocoPublisher
from simpub.xr_device.meta_quest3 import MetaQuest3

from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_h1" / "scene.xml"

def apply_z_rotation(quat, z_angle=np.pi / 2):
    """
    Apply a rotation around the Z-axis to a given quaternion.

    Args:
        quat: The original quaternion (x, y, z, w).
        z_angle: The rotation angle around the Z-axis in radians.

    Returns:
        A new quaternion after applying the Z-axis rotation.
    """
    # Convert the input quaternion to a rotation object
    rotation = R.from_quat(quat)

    # Create a rotation around the Z-axis
    z_rotation = R.from_euler("z", z_angle)

    # Combine the rotations
    new_rotation = rotation * z_rotation  # Order matters: z_rotation is applied first

    # Convert back to quaternion
    return new_rotation.as_quat()

def compensate_gravity(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    subtree_ids: Sequence[int],
    qfrc_applied: Optional[np.ndarray] = None,
) -> None:
    """Compute forces to counteract gravity for the given subtrees.

    Args:
        model: Mujoco model.
        data: Mujoco data.
        subtree_ids: List of subtree ids. A subtree is defined as the kinematic tree
            starting at the body and including all its descendants. Gravity
            compensation forces will be applied to all bodies in the subtree.
        qfrc_applied: Optional array to store the computed forces. If not provided,
            the applied forces in `data` are used.
    """
    qfrc_applied = data.qfrc_applied if qfrc_applied is None else qfrc_applied
    qfrc_applied[:] = 0.0  # Don't accumulate from previous calls.
    jac = np.empty((3, model.nv))
    for subtree_id in subtree_ids:
        total_mass = model.body_subtreemass[subtree_id]
        mujoco.mj_jacSubtreeCom(model, data, jac, subtree_id)
        qfrc_applied[:] -= model.opt.gravity * total_mass @ jac

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)

    feet = ["right_foot", "left_foot"]
    hands = ["right_wrist", "left_wrist"]
   
    subtree_id = model.body("pelvis").id

    tasks = [
        pelvis_orientation_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=10.0,
        ),
        posture_task := mink.PostureTask(model, cost=1.0),
        com_task := mink.ComTask(cost=200.0),
    ]

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=10.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    hand_tasks = []
    for hand in hands:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=200.0,
            orientation_cost=0.0,
            lm_damping=1.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]

    model = configuration.model
    data = configuration.data
    publisher = MujocoPublisher(model, data, host="192.168.0.134")
    player1 = MetaQuest3("ALRMQ3-2")
    
    solver = "quadprog"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("stand")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)

        # Initialize mocap bodies at their respective sites.
        for hand, foot in zip(hands, feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
        data.mocap_pos[com_mid] = data.subtree_com[1]

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            
            input_data = player1.get_input_data()
            if input_data is not None:
                left_hand = input_data["left"]
                right_hand = input_data["right"]
                if left_hand["hand_trigger"]:
                    pos = np.array(input_data["left"]["pos"])
                    pos[1] = pos[1] - 1.0
                    data.mocap_pos[model.body("left_wrist_target").mocapid[0]] = pos
                    rot = input_data["left"]["rot"]
                    # rot = apply_z_rotation(rot, z_angle = - np.pi / 2)
                    data.mocap_quat[model.body("left_wrist_target").mocapid[0]] = np.array([rot[3], rot[0], rot[1], rot[2]])
                if right_hand["hand_trigger"]:
                    pos = np.array(input_data["right"]["pos"])
                    pos[1] = pos[1] - 1.0
                    data.mocap_pos[model.body("right_wrist_target").mocapid[0]] = pos
                    rot = input_data["right"]["rot"]
                    # rot = apply_z_rotation(rot, z_angle = np.pi / 2)
                    data.mocap_quat[model.body("right_wrist_target").mocapid[0]] = np.array([rot[3], rot[0], rot[1], rot[2]])

            # Update task targets.
            com_task.set_target(data.mocap_pos[com_mid])
            for i, (hand_task, foot_task) in enumerate(zip(hand_tasks, feet_tasks)):
                foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
                hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))

            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1)
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)
            
            compensate_gravity(
                model,
                data,
                [subtree_id],
            )
            mujoco.mj_step(model, data)
            # Visualize at fixed FPS.
            viewer.sync()
            # rate.sleep()
