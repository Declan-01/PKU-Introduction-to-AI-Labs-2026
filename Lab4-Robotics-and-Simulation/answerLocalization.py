from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 0.25
MAX_ERROR = 50000

K = 0.425
NOISE_POSITION_STD = 0.1
NOISE_THETA_STD = 0.1
MAX_TRIES=1000

def is_valid_particle(x, y, walls):
    for wall in walls:
        wall_x, wall_y = wall
        closest_x = min(max(x, wall_x - 0.5), wall_x + 0.5) 
        closest_y = min(max(y, wall_y - 0.5), wall_y + 0.5) 
        dx = x - closest_x 
        dy = y - closest_y 
        if dx**2+dy**2 < COLLISION_DISTANCE**2: 
            return False
    return True

def random_particle_with_weight(walls, weight):
    x_min, y_min = np.min(walls, axis=0)
    x_max, y_max = np.max(walls, axis=0)

    for _ in range(10000):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)

        if is_valid_particle(x, y, walls):
            theta = np.random.uniform(0, 2 * np.pi)
            return Particle(x, y, theta, weight)

    # 极端兜底，正常不会走到这里
    return Particle((x_min + x_max) / 2, (y_min + y_max) / 2, 0.0, weight)


def add_noise(particle, walls):

    for _ in range(MAX_TRIES):
        new_position = particle.position + np.random.normal(
            0,
            NOISE_POSITION_STD,
            size=2
        )

        new_theta = (
            particle.theta + np.random.normal(0, NOISE_THETA_STD)
        ) % (2 * np.pi)

        if is_valid_particle(new_position[0], new_position[1], walls):
            return Particle(
                new_position[0],
                new_position[1],
                new_theta,
                particle.weight
            )

    # 如果局部扰动一直非法，就全局随机采一个，防止死循环
    return random_particle_with_weight(walls, particle.weight)


### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    ### 你的代码 ###
    x_min, y_min = np.min(walls, axis=0)
    x_max, y_max = np.max(walls, axis=0)
    count = 0
    while count < N:
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        if is_valid_particle(x, y, walls):
            theta = np.random.uniform(0, 2 * np.pi)
            all_particles.append(Particle(x, y, theta, 1.0 / N))
            count += 1
    ### 你的代码 ###
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight = 1.0
    ### 你的代码 ###
    weight = np.exp(-K * np.linalg.norm(estimated - gt))  
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    ### 你的代码 ###
     ### 你的代码 ###
    N = len(particles)
    if N == 0:
        return []

    weights = np.array([particle.weight for particle in particles], dtype=float)
    total_weight = np.sum(weights)

    if total_weight <= 0 or not np.isfinite(total_weight):
        return generate_uniform_particles(walls, N)

    weights = weights / total_weight

    sample_num = np.floor(weights * N).astype(int)

    new_weight = 1.0 / N

    for index, particle in enumerate(particles):
        base_particle = Particle(
            particle.position[0],
            particle.position[1],
            particle.theta,
            new_weight
        )

        for _ in range(sample_num[index]):
            resampled_particles.append(add_noise(base_particle, walls))

    rest_num = N - len(resampled_particles)

    for _ in range(rest_num):
        resampled_particles.append(
            random_particle_with_weight(walls, new_weight)
        )

    if len(resampled_particles) > N:
        resampled_particles = resampled_particles[:N]

    ### 你的代码 ###
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###
    p.theta = (p.theta + dtheta) % (2 * np.pi)

    p.position[0] += (traveled_distance * np.cos(p.theta))
    p.position[1] += (traveled_distance * np.sin(p.theta))
    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    final_result = Particle()
    ### 你的代码 ###
    weights = np.array([particle.weight for particle in particles])
    final_result = particles[np.argmax(weights)]
    ### 你的代码 ###
    return final_result