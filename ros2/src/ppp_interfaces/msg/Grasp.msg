# A parallel-jaw grasp (stage 4). TCP frame: +z = approach direction, fingers close along +/-y, origin at the
# fingertips (Franka: panda_hand + 0.1034 m along z).
geometry_msgs/PoseStamped tcp       # TCP pose in the robot base frame
geometry_msgs/Pose tcp_in_object    # the same grasp in the object frame (kept while the object is held)
float32 width                       # object width between the fingers (m)
float32 tilt                        # approach angle from straight down (rad)
string face                         # e.g. "+z/+x": approach through the +z face, closing along x
