from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf
from loop_rate_limiters import RateLimiter

import mink

from simpub.sim.mj_publisher import MujocoPublisher
from simpub.xr_device.meta_quest3 import MetaQuest3

from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).parent
_ARM_XML = _HERE / "ufactory_xarm7" / "scene.xml"
_HAND_XML = _HERE / "leap_hand" / "right_hand.xml"

fingers = ["tip_1", "tip_2", "tip_3", "th_tip"]

# fmt: off
HOME_QPOS = [
    # xarm.
    0, -.247, 0, .909, 0, 1.15644, 0,
    # leap.
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
]
# fmt: on


def apply_z_rotation(quat, z_angle = np.pi / 2):
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
    z_rotation = R.from_euler('z', z_angle)

    # Combine the rotations
    new_rotation = rotation * z_rotation  # Order matters: z_rotation is applied first

    # Convert back to quaternion
    return new_rotation.as_quat()

def construct_model():
    arm_mjcf = mjcf.from_path(_ARM_XML.as_posix())
    arm_mjcf.find("key", "home").remove()

    hand_mjcf = mjcf.from_path(_HAND_XML.as_posix())
    palm = hand_mjcf.worldbody.find("body", "palm_lower")
    palm.euler = (np.pi, 0, 0)
    palm.pos = (0.065, -0.04, 0)
    attach_site = arm_mjcf.worldbody.find("site", "attachment_site")
    attach_site.attach(hand_mjcf)

    arm_mjcf.keyframe.add("key", name="home", qpos=HOME_QPOS)

    for finger in fingers:
        body = arm_mjcf.worldbody.add("body", name=f"{finger}_target", mocap=True)
        body.add(
            "geom",
            type="sphere",
            size=".02",
            contype="0",
            conaffinity="0",
            rgba=".6 .3 .3 .5",
        )

    return mujoco.MjModel.from_xml_string(
        arm_mjcf.to_xml_string(), arm_mjcf.get_assets()
    )


if __name__ == "__main__":
    model = construct_model()

    configuration = mink.Configuration(model)

    end_effector_task = mink.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        lm_damping=1.0,
    )

    posture_task = mink.PostureTask(model=model, cost=5e-2)

    finger_tasks = []
    for finger in fingers:
        task = mink.RelativeFrameTask(
            frame_name=f"leap_right/{finger}",
            frame_type="site",
            root_name="leap_right/palm_lower",
            root_type="body",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=1e-3,
        )
        finger_tasks.append(task)

    tasks = [end_effector_task, posture_task, *finger_tasks]

    limits = [
        mink.ConfigurationLimit(model=model),
    ]

    # IK settings.
    solver = "quadprog"
    model = configuration.model
    data = configuration.data

    publisher = MujocoPublisher(model, data, host="192.168.0.134", visible_geoms_groups=[0, 1, 2])
    mq3 = MetaQuest3("IRLMQ3-1")


    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")
        for finger in fingers:
            mink.move_mocap_to_frame(
                model, data, f"{finger}_target", f"leap_right/{finger}", "site"
            )

        T_eef_prev = configuration.get_transform_frame_to_world(
            "attachment_site", "site"
        )

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            # Update kuka end-effector task.
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            input_data = mq3.get_input_data()
            if input_data is not None:
                # left_hand = input_data["left"]
                right_hand = input_data["right"]
                if right_hand["hand_trigger"]:
                    pos = np.array(input_data["right"]["pos"])
                    pos[0] = pos[0] - 0.3
                    data.mocap_pos[model.body("target").mocapid[0]] = pos
                    rot = input_data["right"]["rot"]
                    rot = apply_z_rotation(rot, z_angle = - np.pi)
                    data.mocap_quat[model.body("target").mocapid[0]] = np.array([rot[3], rot[0], rot[1], rot[2]])
                    # if right_hand["index_trigger"]:
                    #     data.ctrl[left_gripper_actuator] = 0.002
                    # else:
                    #     data.ctrl[left_gripper_actuator] = 0.037


            # Update finger tasks.
            for finger, task in zip(fingers, finger_tasks):
                T_pm = configuration.get_transform(
                    f"{finger}_target", "body", "leap_right/palm_lower", "body"
                )
                task.set_target(T_pm)

            for finger in fingers:
                T_eef = configuration.get_transform_frame_to_world(
                    "attachment_site", "site"
                )
                T = T_eef @ T_eef_prev.inverse()
                T_w_mocap = mink.SE3.from_mocap_name(model, data, f"{finger}_target")
                T_w_mocap_new = T @ T_w_mocap
                data.mocap_pos[model.body(f"{finger}_target").mocapid[0]] = (
                    T_w_mocap_new.translation()
                )
                data.mocap_quat[model.body(f"{finger}_target").mocapid[0]] = (
                    T_w_mocap_new.rotation().wxyz
                )

            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-3, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            T_eef_prev = T_eef.copy()

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
