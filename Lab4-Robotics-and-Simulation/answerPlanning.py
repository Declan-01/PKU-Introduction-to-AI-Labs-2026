import numpy as np
from typing import List
from utils import TreeNode
from simuScene import PlanningMap


### 定义一些你需要的变量和函数 ###
STEP_DISTANCE = 1.2
TARGET_THREHOLD = 0.6
MAX_ITER = 2500
GOAL_SAMPLE_RATE = 0.10
GOAL_NEAR_SAMPLE_RATE = 0.30
GOAL_NEAR_STD = 2.5
WAYPOINT_REACHED_DISTANCE = 0.55
MAX_LOOKAHEAD = 2
STUCK_LIMIT = 90
### 定义一些你需要的变量和函数 ###


class RRT:
    def __init__(self, walls) -> None:
        """
        输入包括地图信息，你需要按顺序吃掉的一列事物位置 
        注意：只有按顺序吃掉上一个食物之后才能吃下一个食物，在吃掉上一个食物之前Pacman经过之后的食物也不会被吃掉
        """
        self.goal = None
        self.map = PlanningMap(walls)
        self.walls = walls
        
        # 其他需要的变量
        ### 你的代码 ###      
        wall_min = np.min(self.walls, axis=0)
        wall_max = np.max(self.walls, axis=0)
        self.bounds = (wall_min[0] + 1.0, wall_max[0] - 1.0,
                       wall_min[1] + 1.0, wall_max[1] - 1.0)
        self.free_points = self.collect_free_points()
        self.current_target_index = 0
        self.stuck_counter = 0
        self.replan_cooldown = 0
        self.last_target_dist = None
        ### 你的代码 ###
        
        # 如有必要，此行可删除
        self.path = None
        
        
    def find_path(self, current_position, next_food):
        """
        在程序初始化时，以及每当 pacman 吃到一个食物时，主程序会调用此函数
        current_position: pacman 当前的仿真位置
        next_food: 下一个食物的位置
        
        本函数的默认实现是调用 build_tree，并记录生成的 path 信息。你可以在此函数增加其他需要的功能
        """
        
        ### 你的代码 ###      
        real_start = np.array(current_position, dtype=float)
        start = self.snap_to_free_point(real_start)
        goal = self.snap_to_free_point(np.array(next_food, dtype=float))
        self.goal = goal
        self.current_target_index = 0
        self.stuck_counter = 0
        self.replan_cooldown = 0
        self.last_target_dist = None

        if self.is_segment_free(start, goal) or np.linalg.norm(goal - start) < TARGET_THREHOLD:
            self.path = [goal]
            return

        raw_path = self.build_tree(start, goal)

        # RRT 如果意外失败，不接受半截路径；继续以 goal 为最后目标，后续 get_target 会重规划。
        if raw_path is None or len(raw_path) == 0 or np.linalg.norm(raw_path[-1] - goal) > TARGET_THREHOLD:
            raw_path = [goal]

        path = self.smooth_path_limited(raw_path, max_skip_dist=2.2)

        if len(path) > 1 and np.linalg.norm(path[0] - start) < WAYPOINT_REACHED_DISTANCE:
            path = path[1:]

        # 如果规划起点是从贴墙位置修正出来的，先把该安全点作为逃离墙边的目标。
        if np.linalg.norm(start - real_start) > 1e-6:
            path = [start] + path

        self.path = path if len(path) > 0 else [goal]
        ### 你的代码 ###
        
        
    def get_target(self, current_position, current_velocity):
        """
        主程序将在每个仿真步内调用此函数，并用返回的位置计算 PD 控制力来驱动 pacman 移动 
        current_position: pacman 当前的仿真位置 
        current_velocity: pacman 当前的仿真速度 
        一种可能的实现策略是，仅作为参考： 
        （1）记录该函数的调用次数 
        （2）假设当前 path 中每个节点需要作为目标 n 次 
        （3）记录目前已经执行到当前 path 的第几个节点，以及它的执行次数，如果超过 n，则将目标改为下一节点 
        你也可以考虑根据当前位置与 path 中节点位置的距离来决定如何选择 target 
        同时需要注意，仿真中的 pacman 并不能准确到达 path 中的节点。
        你可能需要考虑在什么情况下重新规划 path
        """
        target_pose = np.zeros_like(current_position)
        ### 你的代码 ###
        current = np.array(current_position, dtype=float)

        # 物理仿真可能让 Pacman 贴墙到 checkoccupy=True；这时不要立刻 RRT，
        # 先给一个附近的安全点，把它从墙边拉出来。
        if self.map.checkoccupy(current):
            escape = self.nearest_free_point(current)
            if np.linalg.norm(escape - current) > 1e-6:
                return escape

        if self.path is None or len(self.path) == 0:
            return current

        if self.replan_cooldown > 0:
            self.replan_cooldown -= 1

        if self.goal is None:
            self.goal = np.array(self.path[-1], dtype=float)

        if self.is_segment_free(current, self.goal):
            return self.goal

        # 到达当前节点附近后再换下一个节点，不按固定调用次数换。
        while self.current_target_index < len(self.path) - 1:
            target = self.path[self.current_target_index]
            if np.linalg.norm(current - target) > WAYPOINT_REACHED_DISTANCE:
                break
            self.current_target_index += 1

        # 当前 target 被墙隔开时，先尝试在原路径上找一个仍可直达的点；
        # 若整段剩余路径都不可见，再低频重规划。
        if not self.is_segment_free(current, self.path[self.current_target_index]):
            visible_idx = None
            for idx in range(self.current_target_index + 1, len(self.path)):
                if self.is_segment_free(current, self.path[idx]):
                    visible_idx = idx
                    break
            if visible_idx is not None:
                self.current_target_index = visible_idx
            elif self.replan_cooldown == 0:
                old_goal = self.goal.copy()
                self.find_path(current, old_goal)
                self.replan_cooldown = 80
                if self.path is not None and len(self.path) > 0:
                    return self.path[0]
                return old_goal

        # 限制 look-ahead：只跳有限个可直达点，避免远距离 PD 把 Pacman 拉到墙上。
        farthest = min(len(self.path) - 1, self.current_target_index + MAX_LOOKAHEAD)
        for idx in range(farthest, self.current_target_index, -1):
            if self.is_segment_free(current, self.path[idx]):
                self.current_target_index = idx
                break

        target_pose = self.path[self.current_target_index]

        # 简单卡住检测：长时间没有接近当前目标，则重规划。
        dist = np.linalg.norm(current - target_pose)
        if self.last_target_dist is not None and dist > self.last_target_dist - 0.01:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_target_dist = dist

        if self.stuck_counter > STUCK_LIMIT and np.linalg.norm(current_velocity) < 0.25:
            if self.replan_cooldown == 0:
                old_goal = self.goal.copy()
                self.find_path(current, old_goal)
                self.replan_cooldown = 100
                self.stuck_counter = 0
                if self.path is not None and len(self.path) > 0:
                    return self.path[0]
                return old_goal
            escape = self.nearest_free_point(current)
            if np.linalg.norm(escape - current) > 1e-6:
                self.stuck_counter = 0
                return escape

        ### 你的代码 ###
        return target_pose
        
    ### 以下是RRT中一些可能用到的函数框架，全部可以修改，当然你也可以自己实现 ###
    def build_tree(self, start, goal):
        """ 实现你的快速探索搜索树，输入为当前目标食物的编号，
        规划从 start 位置食物到 goal 位置的路径 
        返回一个包含坐标的列表，为这条路径上的pd targets 
        你可以调用find_nearest_point和connect_a_to_b两个函数 
        另外self.map的checkoccupy和checkline也可能会需要，
        可以参考simuScene.py中的PlanningMap类查看它们的用法 
        """
        
        
        """
        纯 RRT-Connect 双向随机树。
        从 start 和 goal 同时生长，每轮扩展一棵树，再让另一棵树贪心连接新节点。
        """
        if self.is_segment_free(start, goal):
            return [start, goal]

        start_tree: List[TreeNode] = [TreeNode(-1, start[0], start[1])]
        goal_tree: List[TreeNode] = [TreeNode(-1, goal[0], goal[1])]

        for iteration in range(MAX_ITER):
            if iteration % 2 == 0:
                active_tree = start_tree
                passive_tree = goal_tree
                sample_target = goal
                grow_from_start = True
            else:
                active_tree = goal_tree
                passive_tree = start_tree
                sample_target = start
                grow_from_start = False

            random_point = self.sample_point(sample_target, self.bounds)
            if self.map.checkoccupy(random_point):
                continue

            new_idx = self.extend_tree_once(active_tree, random_point)
            if new_idx is None:
                continue

            new_point = active_tree[new_idx].pos
            connect_idx = self.connect_tree_greedily(passive_tree, new_point)
            if connect_idx is None:
                continue

            connect_point = passive_tree[connect_idx].pos
            if np.linalg.norm(connect_point - new_point) <= TARGET_THREHOLD and self.is_segment_free(connect_point, new_point):
                # 保证两棵树之间最后一小段也作为树边存在，便于拼路径。
                passive_tree.append(TreeNode(connect_idx, new_point[0], new_point[1]))
                connect_idx = len(passive_tree) - 1

                if grow_from_start:
                    return self.merge_two_trees(start_tree, new_idx, goal_tree, connect_idx)
                else:
                    return self.merge_two_trees(start_tree, connect_idx, goal_tree, new_idx)

        return None

    @staticmethod
    def find_nearest_point(point, graph):
        """
        找到图中离目标位置最近的节点，返回该节点的编号和到目标位置距离、
        """
        nearest_idx = -1
        nearest_distance = 10000000.
        ### 你的代码 ###
        for idx, node in enumerate(graph):
            distance = np.linalg.norm(point - node.pos)
            if distance < nearest_distance:
                nearest_idx = idx
                nearest_distance = distance
        ### 你的代码 ###
        return nearest_idx, nearest_distance
    
    def connect_a_to_b(self, point_a, point_b):
        """ 以A点为起点，沿着A到B的方向前进STEP_DISTANCE的距离，
        并用self.map.checkline函数检查这段路程是否可以通过 
        输入： point_a, point_b: 维度为(2,)的np.array，A点和B点位置，注意是从A向B方向前进 
        输出： is_empty: bool，True表示从A出发前进STEP_DISTANCE这段距离上没有障碍物 
        newpoint: 从A点出发向B点方向前进STEP_DISTANCE距离后的新位置，
        如果is_empty为真，之后的代码需要把这个新位置添加到图中 
        """
        is_empty = False
        newpoint = np.zeros(2)
        ### 你的代码 ###
        point_a = np.array(point_a, dtype=float)
        point_b = np.array(point_b, dtype=float)

        direction = point_b - point_a
        distance = np.linalg.norm(direction)
        if distance < 1e-8:
            return False, point_a.copy()

        step = min(STEP_DISTANCE, distance)
        newpoint = point_a + step * direction / distance

        # 将扩展点吸附到附近自由格点中心，减少贴墙目标点。
        snapped = self.snap_to_free_point(newpoint)
        if np.linalg.norm(snapped - point_a) > 1e-6 and np.linalg.norm(snapped - newpoint) < 1.0:
            newpoint = snapped

        if np.linalg.norm(newpoint - point_a) < 1e-8:
            return False, point_a.copy()
        is_empty = self.is_segment_free(point_a, newpoint)
        ### 你的代码 ###
        return is_empty, newpoint

    def is_segment_free(self, point_a, point_b):
        point_a = np.array(point_a, dtype=float)
        point_b = np.array(point_b, dtype=float)

        if self.map.checkoccupy(point_a) or self.map.checkoccupy(point_b):
            return False
        if np.linalg.norm(point_b - point_a) < 1e-8:
            return True
        return not self.map.checkline(point_a.tolist(), point_b.tolist())[0]

    def extract_path(self, graph, node_idx):
        path = []
        while node_idx != -1:
            path.append(graph[node_idx].pos.copy())
            node_idx = graph[node_idx].parent_idx
        path.reverse()
        return path

    def sample_point(self, goal, bounds):
        """
        目标偏置采样：
        10% 直接采样目标；
        20% 从目标附近的自由格点采样；
        70% 从全图自由格点采样。
        仍然是随机采样，但避免大量样本落在墙体内。
        """
        x_min, x_max, y_min, y_max = bounds
        r = np.random.rand()

        if r < GOAL_SAMPLE_RATE:
            sample = goal.copy()
        elif r < GOAL_NEAR_SAMPLE_RATE and len(self.free_points) > 0:
            dist = np.linalg.norm(self.free_points - goal, axis=1)
            near = self.free_points[dist < 6.0]
            if len(near) > 0:
                sample = near[np.random.randint(len(near))].copy()
            else:
                sample = goal + np.random.normal(0, GOAL_NEAR_STD, size=2)
        elif len(self.free_points) > 0:
            sample = self.free_points[np.random.randint(len(self.free_points))].copy()
        else:
            sample = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max)
            ])

        sample[0] = np.clip(sample[0], x_min, x_max)
        sample[1] = np.clip(sample[1], y_min, y_max)
        return sample

    def collect_free_points(self):
        x_min, x_max, y_min, y_max = self.bounds
        points = []
        for x in range(int(np.ceil(x_min)), int(np.floor(x_max)) + 1):
            for y in range(int(np.ceil(y_min)), int(np.floor(y_max)) + 1):
                p = np.array([x, y], dtype=float)
                if not self.map.checkoccupy(p):
                    points.append(p)
        if len(points) == 0:
            return np.zeros((0, 2))
        return np.array(points, dtype=float)

    def extend_tree_once(self, tree, target_point):
        nearest_idx, _ = self.find_nearest_point(target_point, tree)
        nearest_node = tree[nearest_idx]
        is_empty, newpoint = self.connect_a_to_b(nearest_node.pos, target_point)
        if not is_empty:
            return None
        if self.map.checkoccupy(newpoint):
            return None
        tree.append(TreeNode(nearest_idx, newpoint[0], newpoint[1]))
        return len(tree) - 1

    def connect_tree_greedily(self, tree, target_point):
        """
        RRT-Connect：让另一棵树连续朝 target_point 走，而不是只尝试直连一次。
        """
        last_idx = None
        for _ in range(35):
            nearest_idx, nearest_dist = self.find_nearest_point(target_point, tree)
            nearest_node = tree[nearest_idx]

            if nearest_dist <= TARGET_THREHOLD:
                if self.is_segment_free(nearest_node.pos, target_point):
                    return nearest_idx
                return last_idx

            is_empty, newpoint = self.connect_a_to_b(nearest_node.pos, target_point)
            if not is_empty:
                return last_idx

            tree.append(TreeNode(nearest_idx, newpoint[0], newpoint[1]))
            last_idx = len(tree) - 1

            if np.linalg.norm(newpoint - target_point) <= TARGET_THREHOLD:
                return last_idx

        return last_idx

    def merge_two_trees(self, start_tree, start_meet_idx, goal_tree, goal_meet_idx):
        path_from_start = self.extract_path(start_tree, start_meet_idx)
        path_from_goal = self.extract_path(goal_tree, goal_meet_idx)  # goal -> ... -> meet
        path_to_goal = list(reversed(path_from_goal[:-1]))
        return path_from_start + path_to_goal

    def snap_to_free_point(self, point):
        """把点吸附到最近的自由格点中心；RRT 节点放在格点中心会更稳。"""
        point = np.array(point, dtype=float)
        if hasattr(self, "free_points") and len(self.free_points) > 0:
            dist = np.linalg.norm(self.free_points - point, axis=1)
            idx = int(np.argmin(dist))
            if dist[idx] < 2.0:
                return self.free_points[idx].copy()
        return self.nearest_free_point(point)

    def nearest_free_point(self, point):
        """
        找当前连续位置附近最近的非占用点。
        RRT 的根节点如果处在 checkoccupy=True 的位置，整棵树会完全长不出来，
        所以在建树和执行时都需要做这个局部修正。
        """
        point = np.array(point, dtype=float)
        x_min, x_max, y_min, y_max = self.bounds
        p0 = point.copy()
        p0[0] = np.clip(p0[0], x_min, x_max)
        p0[1] = np.clip(p0[1], y_min, y_max)
        if not self.map.checkoccupy(p0):
            return p0

        if hasattr(self, "free_points") and len(self.free_points) > 0:
            dist = np.linalg.norm(self.free_points - p0, axis=1)
            idx = int(np.argmin(dist))
            if dist[idx] < 3.0:
                return self.free_points[idx].copy()

        best = None
        best_dist = 1e18
        for radius in np.linspace(0.15, 2.5, 16):
            for angle in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                cand = p0 + radius * np.array([np.cos(angle), np.sin(angle)])
                cand[0] = np.clip(cand[0], x_min, x_max)
                cand[1] = np.clip(cand[1], y_min, y_max)
                if self.map.checkoccupy(cand):
                    continue
                dist = np.linalg.norm(cand - point)
                if dist < best_dist:
                    best = cand.copy()
                    best_dist = dist
            if best is not None:
                return best

        return p0

    def smooth_path_limited(self, path, max_skip_dist=5.0):
        """
        保守 shortcut：只删除距离不太远的中间点。
        这样保留 RRT 路径的安全性，同时减少节点数量。
        """
        if path is None or len(path) <= 2:
            return path

        smooth = [path[0]]
        i = 0
        while i < len(path) - 1:
            best_j = i + 1
            upper = len(path) - 1
            for j in range(i + 1, upper + 1):
                if np.linalg.norm(path[j] - path[i]) > max_skip_dist:
                    break
                if self.is_segment_free(path[i], path[j]):
                    best_j = j
            smooth.append(path[best_j])
            i = best_j
        return smooth
