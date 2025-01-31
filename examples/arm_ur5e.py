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
_XML = _HERE / "universal_robots_ur5e" / "scene.xml"

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

    configuration = mink.Configuration(model)

    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="attachment_site",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
    ]

    # Enable collision avoidance between the following geoms:
    collision_pairs = [
        (["wrist_3_link"], ["floor", "wall"]),
    ]

    limits = [
        mink.ConfigurationLimit(model=model),
        mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
    ]

    max_velocities = {
        "shoulder_pan": np.pi,
        "shoulder_lift": np.pi,
        "elbow": np.pi,
        "wrist_1": np.pi,
        "wrist_2": np.pi,
        "wrist_3": np.pi,
    }
    velocity_limit = mink.VelocityLimit(model, max_velocities)
    limits.append(velocity_limit)

    mid = model.body("target").mocapid[0]
    model = configuration.model
    data = configuration.data
    solver = "quadprog"

    publisher = MujocoPublisher(model, data, host="192.168.0.134", visible_geoms_groups=[1, 2])
    mq3 = MetaQuest3("IRLMQ3-1")

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("home")

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=500.0, warn=False)
        while viewer.is_running():
            # Update task target.
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
                    # data.mocap_quat[model.body("target").mocapid[0]] = np.array([rot[3], rot[0], rot[1], rot[2]])
                    # if right_hand["index_trigger"]:
                    #     data.ctrl[left_gripper_actuator] = 0.002
                    # else:
                    #     data.ctrl[left_gripper_actuator] = 0.037



            # Compute velocity and integrate into the next configuration.
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-3, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            # Note the below are optional: they are used to visualize the output of the
            # fromto sensor which is used by the collision avoidance constraint.
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)

            # Visualize at fixed FPS.
            viewer.sync()
            rate.sleep()
