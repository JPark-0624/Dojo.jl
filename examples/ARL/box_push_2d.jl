using Dojo
using DojoEnvironments: Z_AXIS
using LinearAlgebra

# -----------------------------
# 1. 몸체(bodies)와 조인트(joints) 정의
# -----------------------------
origin = Origin{Float64}()

# 박스: 가로, 세로, 높이, 질량
box = Box(1.0, 1.0, 1.0, 1.0)

# 푸셔: 반지름, 질량
pusher = Dojo.Sphere(0.15, 0.2; color=RGBA(1,0,0,1))

# 박스는 x–y 평면에서만 움직이고 z축 기준으로만 회전하는 Planar joint
# two_finger_box_control.jl에서는 [1;0;0] (yz-평면) 썼는데,
# 여기서는 z축이 평면의 법선 → x–y 평면 운동으로 해석.
joint_box   = JointConstraint(PlanarAxis(origin, box, Z_AXIS))

# 푸셔도 x–y 평면에서 움직이는 planar joint로 둔다.
joint_push  = JointConstraint(PlanarAxis(origin, pusher, Z_AXIS))

bodies = [box, pusher]
joints = [joint_box, joint_push]

# -----------------------------
# 2. 접촉(contact) 정의
#   (two_finger_box_control.jl에서 그대로 가져온 패턴)
# -----------------------------
side = 1.0

# 박스의 8개 코너에 접촉점 정의 (박스가 한 변 길이 1.0인 큐브라고 가정)
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

# 노멀은 모두 z축 방향 (위/아래)
normals = fill(Z_AXIS, 8)
friction_coefficients = fill(0.5, 8)

# Sphere–Box 충돌 모델 (two_finger_box_control.jl와 동일한 파라미터 형태)
collision = SphereBoxCollision{Float64,2,3,6}(
    szeros(3), 1.0, 1.0, 2 * 1.0, 0.5
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
# 3. 메커니즘 생성
# -----------------------------
mech = Mechanism(
    origin,
    bodies,
    joints,
    contacts;
    gravity = 0.0,      # 중력은 꺼두고, 순수 2D 평면 운동처럼 사용
    timestep = 0.05
)

# -----------------------------
# 4. 초기 상태 설정
# -----------------------------
# x2: 위치 (translation) 벡터
# 여기서는 z=0 평면 위에 놓여 있는 걸로 가정 (3D지만 2D처럼)
box.state.x2    = [0.0, 0.0, 0.0]     # 박스는 원점 근처
pusher.state.x2 = [-3.0, 0.0, 0.0]    # 박스를 왼쪽에서 +x 방향으로 밀어올 위치

# 초기 속도는 0
box.state.v15    = [0.0, 0.0, 0.0]
pusher.state.v15 = [0.0, 0.0, 0.0]

# -----------------------------
# 5. 컨트롤러: 푸셔만 +x 방향으로 움직이게
# -----------------------------
function controller!(mechanism, k)
    # 여기서는 아주 단순하게 "푸셔의 x-속도"를 강제로 지정하는 방식으로 구현
    # → 나중에 이 부분을 set_input! 기반 force/torque 제어로 교체 가능.
    v_push = 2.0           # m/s, world-frame +x 방향 속도

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
# 6. 시뮬레이션 & 시각화
# -----------------------------
sim_time = 5.0  # 3초간 시뮬레이션
storage = simulate!(mech, sim_time, controller!; record=true)

for i in 1:10
    println(storage.v[2][i])  # v15 저장값 확인 (linear vel)
end

visualize(mech, storage)
