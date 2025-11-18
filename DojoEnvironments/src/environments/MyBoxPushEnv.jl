module MyBoxPushEnv

using Dojo
using DojoEnvironments: Z_AXIS
using LinearAlgebra

"박스+푸셔 메커니즘과 옵션, 푸셔 조인트 인덱스를 담는 간단한 env 타입"
struct BoxPushEnv
    mech::Mechanism
    joint_push_index::Int
    opts::SolverOptions
end

"네가 만든 box+pusher 예제를 그대로 구조화한 빌더 함수"
function build_env(; timestep::Float64 = 0.05)
    origin = Origin{Float64}()

    # 1) 몸체
    box    = Box(1.0, 1.0, 1.0, 1.0)
    pusher = Dojo.Sphere(0.15, 0.2; color=RGBA(1, 0, 0, 1))

    # 2) 플래너 조인트 (x–y 평면 + z축 회전)
    joint_box  = JointConstraint(PlanarAxis(origin, box, Z_AXIS))
    joint_push = JointConstraint(PlanarAxis(origin, pusher, Z_AXIS))

    bodies = [box, pusher]
    joints = [joint_box, joint_push]

    # 3) sphere–box 접촉 모델
    side = 1.0

    collision = SphereBoxCollision{Float64,2,3,6}(
        szeros(3),  # origin_sphere
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

    # 4) 기본 초기조건
    box.state.x2    = [0.0, 0.0, 0.0]
    pusher.state.x2 = [-3.0, 0.0, 0.0]

    box.state.v15    = [0.0, 0.0, 0.0]
    pusher.state.v15 = [0.0, 0.0, 0.0]

    opts = SolverOptions()

    # joint_push는 bodies[2]에 해당하는 두 번째 조인트라고 가정
    joint_push_index = 2

    return BoxPushEnv(mech, joint_push_index, opts)
end

"env 안 메커니즘을 기본 pose/velocity로 리셋하고 state를 리턴"
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

"현재 메커니즘의 최소좌표 state를 flatten 해서 리턴"
get_state(env::BoxPushEnv) = Dojo.get_minimal_state(env.mech)

"푸셔에 world-frame +x 방향 속도 v_push를 걸어주는 헬퍼"
function set_pusher_world_velocity!(env::BoxPushEnv, v_push::Float64)
    mech  = env.mech
    joint = mech.joints[env.joint_push_index]
    pbody = Dojo.get_body(mech, joint.parent_id)

    desired_v = [v_push, 0.0, 0.0]  # world frame (vx, vy, vz)

    # planar joint 내부 최소좌표 → body-local translation
    Atra = transpose(Matrix(nullspace_mask(joint.translational)))  # 3 x N

    # parent body 회전 (body → world)
    R = Matrix(Dojo.rotation_matrix(pbody.state.q2))               # 3 x 3

    # 최소좌표 속도 → world-frame 속도
    M  = R * Atra             # 3 x N
    Δv = M \ desired_v        # 최소자승 해

    rot_dim = input_dimension(joint.rotational)
    vω = vcat(Δv, zeros(rot_dim))

    set_minimal_velocities!(mech, joint, vω)
    return
end

"한 스텝 dynamics: x, v_push를 받아 x_next를 벡터로 리턴"
function step(env::BoxPushEnv,
              x::AbstractVector{<:Real},
              v_push::Real)
    # 1) 메커니즘 상태를 x로 세팅
    Dojo.set_minimal_state!(env.mech, x)

    # 2) 푸셔 world-frame 속도 지정
    set_pusher_world_velocity!(env, float(v_push))

    # 3) control input은 일단 0 벡터로 가정
    T = eltype(x)
    u = zeros(T, input_dimension(env.mech))

    # 4) 한 스텝 적분
    Dojo.step_minimal_coordinates!(env.mech, x, u; opts = env.opts)

    # 5) 새 상태 반환
    return Dojo.get_minimal_state(env.mech)
end


end # module
