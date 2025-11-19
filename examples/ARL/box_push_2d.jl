using Dojo
using DojoEnvironments: Z_AXIS
using LinearAlgebra

# -----------------------------
# 1. Define bodies and joints
# -----------------------------
origin = Origin{Float64}()

# Box: width, height, depth, mass
box = Box(1.0, 1.0, 1.0, 1.0)

# Pusher: radius, mass
pusher = Dojo.Sphere(0.15, 0.2; color=RGBA(1,0,0,1))

# The box uses a planar joint that allows translation in the x–y plane
# and rotation about the z axis. In two_finger_box_control.jl they
# used [1;0;0] (yz-plane); here we interpret Z_AXIS as the plane
# normal, so motion is in the x–y plane.
joint_box   = JointConstraint(PlanarAxis(origin, box, Z_AXIS))

# The pusher is also attached with a planar joint (x–y plane motion).
joint_push  = JointConstraint(PlanarAxis(origin, pusher, Z_AXIS))

bodies = [box, pusher]
joints = [joint_box, joint_push]

# -----------------------------
# 2. Define contacts
#   (pattern copied from two_finger_box_control.jl)
# -----------------------------
side = 1.0

# Define contact points at the 8 corners of the box
# (assume a cube with side length = 1.0)
contact_origins = [
            [[ side / 2.0;  side / 2.0; -side / 2.0]]
            [[ side / 2.0; -side / 2.0; -side / 2.0]]
            [[-side / 2.0;  side / 2.0; -side / 2.0]]
            [[-side / 2.0; -side / 2.0; -side / 2.0]]
            [[ side / 2.0;  side / 2.0;  side / 2.0]]
            [[ side / 2.0; -side / 2.0;  side / 2.0]]
            [[-side / 2.0;  side / 2.0;  side / 2.0]]
            [[-side / 2.0; -side / 2.0;  side / 2.0]]
        ]

# Normals are all in the z direction (up/down)
normals = fill(Z_AXIS, 8)
friction_coefficients = fill(0.5, 8)

# Sphere–Box collision model (same parameter format as
# two_finger_box_control.jl)
collision = SphereBoxCollision{Float64,2,3,6}(
    zeros(3), 1.0, 1.0, 2 * 1.0, 0.5
)


friction_parameterization = [
    1.0  0.0
    0.0  1.0
]

body_body_contact = NonlinearContact{Float64,8}(
    1.9,                   # Friction coefficient
    friction_parameterization,
    collision
)

contacts = [
    ContactConstraint((body_body_contact, pusher.id, box.id), name=:pusher_box)
]

# -----------------------------
# 3. Create the mechanism
# -----------------------------
mech = Mechanism(
    origin,
    bodies,
    joints,
    contacts;
    gravity = 0.0,      # gravity is turned off to emulate pure 2D planar motion
    timestep = 0.05
)

# -----------------------------
# 4. Initial state setup
# -----------------------------
# x2: position (translation) vector
# We assume everything is on the z=0 plane (3D simulation used as 2D)
box.state.x2    = [0.0, 0.0, 0.0]     # box is near the origin
pusher.state.x2 = [-3.0, 0.0, 0.0]    # pusher is placed to the left to push in +x direction

# Initial velocities are zero
box.state.v15    = [0.0, 0.0, 0.0]
pusher.state.v15 = [0.0, 0.0, 0.0]

# -----------------------------
# 5. Controller: move only the pusher in +x direction
# -----------------------------
function controller!(mechanism, k)
    # Implement a very simple controller that forces the pusher's x-velocity
    # → this can be replaced later with set_input!-based force/torque control.
    v_push = 2.0           # m/s, world-frame +x velocity

    # We want the pusher body to have a fixed world-frame velocity [v_push, 0, 0].
    # Compute the minimal translational velocity coordinates (Δv) that produce
    # the desired world velocity and apply them via set_minimal_velocities!.
    joint = joint_push
    pbody = get_body(mechanism, joint.parent_id)
    cbody = get_body(mechanism, joint.child_id)

    # desired world velocity (x, y, z) as plain Array (no StaticArrays required)
    desired_v = [v_push, 0.0, 0.0]

    # Atra maps minimal translational coordinates -> body-local translation
    # nullspace_mask(...) typically returns a small StaticArray; convert to Matrix and transpose
    Atra = transpose(Matrix(nullspace_mask(joint.translational)))  # 3 x N

    # rotation of parent body (maps body frame -> world frame)
    R = Matrix(Dojo.rotation_matrix(pbody.state.q2))  # 3 x 3

    # map minimal translational velocities to world-frame velocities
    M = R * Atra  # 3 x N

    # solve least-squares for minimal Δv: M * Δv ≈ desired_v
    # (M is 3xN, typically 3x2 for a planar joint)
    Δv = M \ desired_v

    # rotational minimal dimension for this joint
    rot_dim = input_dimension(joint.rotational)

    # build minimal velocity vector [Δv; Δω] and set it
    vω = vcat(Δv, zeros(rot_dim))
    set_minimal_velocities!(mechanism, joint, vω)
end

# -----------------------------
# 6. Simulation & visualization
# -----------------------------
sim_time = 5.0  # simulate for 5 seconds
storage = simulate!(mech, sim_time, controller!; record=true)

for i in 1:10
    println(storage.v[2][i])  # check stored v15 values (linear velocity)
end

visualize(mech, storage)
