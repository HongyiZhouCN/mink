from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

from simpub.sim.mj_publisher import MujocoPublisher
from simpub.xr_device.meta_quest3 import MetaQuest3
from scipy.spatial.transform import Rotation as R

_HERE = Path(__file__).parent
_XML = _HERE / "boston_dynamics_spot" / "scene.xml"


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


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())
    data = mujoco.MjData(model)

    publisher = MujocoPublisher(model, data, host="192.168.0.134", visible_geoms_groups=[1, 2])
    mq3 = MetaQuest3("IRLMQ3-1")

    ## =================== ##
    ## Setup IK.
    ## =================== ##

    configuration = mink.Configuration(model)

    feet = ["FL", "FR", "HR", "HL"]

    base_task = mink.FrameTask(
        frame_name="body",
        frame_type="body",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    posture_task = mink.PostureTask(model, cost=1e-5)

    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="geom",
            position_cost=1.0,
            orientation_cost=0.0,
        )
        feet_tasks.append(task)

    eef_task = mink.FrameTask(
        frame_name="EE",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
    )

    tasks = [base_task, posture_task, *feet_tasks, eef_task]

    ## =================== ##

    base_mid = model.body("body_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    eef_mid = model.body("EE_target").mocapid[0]

    # IK settings.
    solver = "quadprog"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        posture_task.set_target_from_configuration(configuration)
        for foot in feet:
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "geom")
        mink.move_mocap_to_frame(model, data, "body_target", "body", "body")
        mink.move_mocap_to_frame(model, data, "EE_target", "EE", "site")

        rate = RateLimiter(frequency=500.0, warn=False)
        while viewer.is_running():
            base_task.set_target(mink.SE3.from_mocap_id(data, base_mid))
            for i, task in enumerate(feet_tasks):
                task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
            eef_task.set_target(mink.SE3.from_mocap_id(data, eef_mid))


            # Update posture task target.
            input_data = mq3.get_input_data()
            if input_data is not None:
                # left_hand = input_data["left"]
                right_hand = input_data["right"]
                if right_hand["hand_trigger"]:
                    pos = np.array(input_data["right"]["pos"])
                    # pos[0] = pos[0] + 0.1
                    data.mocap_pos[model.body("EE_target").mocapid[0]] = pos
                    rot = input_data["right"]["rot"]
                    rot = apply_z_rotation(rot, z_angle = - np.pi)
                    data.mocap_quat[model.body("EE_target").mocapid[0]] = np.array([rot[3], rot[0], rot[1], rot[2]])
                    # if right_hand["index_trigger"]:
                    #     data.ctrl[left_gripper_actuator] = 0.002
                    # else:
                    #     data.ctrl[left_gripper_actuator] = 0.037


            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-3)
                configuration.integrate_inplace(vel, rate.dt)

                pos_achieved = True
                ori_achieved = True
                for task in [
                    eef_task,
                    base_task,
                    *feet_tasks,
                ]:
                    err = eef_task.compute_error(configuration)
                    pos_achieved &= bool(np.linalg.norm(err[:3]) <= pos_threshold)
                    ori_achieved &= bool(np.linalg.norm(err[3:]) <= ori_threshold)
                if pos_achieved and ori_achieved:
                    break

            data.ctrl = configuration.q[7:]
            mujoco.mj_step(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
