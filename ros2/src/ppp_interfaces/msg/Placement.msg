# Where and how an object can be set down (stage 5): a point on a horizontal support with free space around it, and
# the rest pose the object has there (one of the stable ways it can stand, e.g. upright or on its side).
geometry_msgs/PointStamped point    # on the support surface, robot base frame
geometry_msgs/Quaternion orientation   # object orientation in that rest pose (base frame; its yaw is free)
float32 object_bottom               # height of the object's origin above the support in that pose (m)
string rest_pose                    # "upright", "on its side", ...
uint8 rank                          # 0 = most preferred rest pose (upright first)
float32 clearance                   # free radius around the point (m)
float32 headroom                    # free height above the support (m; inf = open)
float32 reach                       # distance from the Franka shoulder (m)
