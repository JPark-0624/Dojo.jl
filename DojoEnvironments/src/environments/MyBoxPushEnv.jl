module MyBoxPushEnv

using Dojo
using DojoEnvironments: Z_AXIS
using LinearAlgebra

"Simple env type holding the box+pusher mechanism, solver options,
and the pusher joint index."
struct BoxPushEnv
    mech::Mechanism
    joint_push_index::Int
    opts::SolverOptions
end

"Builder function that structures the box+pusher example into an environment."
function build_env(; timestep::Float64 = 0.05)
    origin = Origin{Float64}()

    # 1) bodies
    box    = Box(1.0, 1.0, 1.0, 1.0)
    pusher = Dojo.Sphere(0.15, 0.2; color=RGBA(1, 0, 0, 1))

    # 2) planar joints (x–y translation + z rotation)
    joint_box  = JointConstraint(PlanarAxis(origin, box, Z_AXIS))
    joint_push = JointConstraint(PlanarAxis(origin, pusher, Z_AXIS))

    bodies = [box, pusher]
    joints = [joint_box, joint_push]

    # 3) sphere–box contact model
    side = 1.0

    collision = SphereBoxCollision{Float64,2,3,6}(
        zeros(3),  # origin_sphere
        1.0, 1.0, 2 * 1.0,
        0.5         # sphere radius
    )

    friction_parameterization = [
        1.0  0.0
        0.0  1.0
    ]

    body_body_contact = NonlinearContact{Float64,8}(
        1.9,
        friction_parameterization,
        collision
    )

    contacts = [
        ContactConstraint((body_body_contact, pusher.id, box.id),
                          name = :pusher_box)
    ]

    mech = Mechanism(
        origin,
        bodies,
        joints,
        contacts;
        gravity  = 0.0,
        timestep = timestep
    )

    # 4) default initial conditions
    box.state.x2    = [0.0, 0.0, 0.0]
    pusher.state.x2 = [-3.0, 0.0, 0.0]

    box.state.v15    = [0.0, 0.0, 0.0]
    pusher.state.v15 = [0.0, 0.0, 0.0]

    opts = SolverOptions()

    # assume `joint_push` is the second joint (bodies[2])
    joint_push_index = 2

    return BoxPushEnv(mech, joint_push_index, opts)
end

"Reset the mechanism inside the env to default pose/velocity and return state."
function reset!(env::BoxPushEnv)
    mech = env.mech
    box    = mech.bodies[1]
    pusher = mech.bodies[2]

    box.state.x2    = [0.0, 0.0, 0.0]
    pusher.state.x2 = [-3.0, 0.0, 0.0]

    box.state.v15    = [0.0, 0.0, 0.0]
    pusher.state.v15 = [0.0, 0.0, 0.0]

    return get_state(env)
end

"Return the mechanism's minimal-coordinate state (flattened)."
get_state(env::BoxPushEnv) = Dojo.get_minimal_state(env.mech)

"Helper that applies a world-frame +x velocity `v_push` to the pusher."
function set_pusher_world_velocity!(env::BoxPushEnv, v_push::Float64)
    mech  = env.mech
    joint = mech.joints[env.joint_push_index]
    pbody = Dojo.get_body(mech, joint.parent_id)

    desired_v = [v_push, 0.0, 0.0]  # world frame (vx, vy, vz)

    # Map minimal translational coordinates -> body-local translation
    Atra = transpose(Matrix(nullspace_mask(joint.translational)))  # 3 x N

    # Parent body rotation (body -> world)
    R = Matrix(Dojo.rotation_matrix(pbody.state.q2))               # 3 x 3

    # Map minimal-coordinate velocities to world-frame velocities
    M  = R * Atra             # 3 x N
    Δv = M \ desired_v        # least-squares solution

    rot_dim = input_dimension(joint.rotational)
    vω = vcat(Δv, zeros(rot_dim))

    set_minimal_velocities!(mech, joint, vω)
    return
end

"One-step dynamics: take `x` and `v_push`, return next state vector."
function step(env::BoxPushEnv,
              x::AbstractVector{<:Real},
              v_push::Real)
    # 1) set mechanism state to `x`
    Dojo.set_minimal_state!(env.mech, x)
    # 2) set pusher world-frame velocity
    set_pusher_world_velocity!(env, float(v_push))
    # 3) assume control input is zero vector for now
    T = eltype(x)
    u = zeros(T, input_dimension(env.mech))
    # 4) integrate one step
    Dojo.step_minimal_coordinates!(env.mech, x, u; opts = env.opts)
    # 5) return new state
    return Dojo.get_minimal_state(env.mech)
end


end # module
