import logging
import os
import libero
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain
from libero.libero.envs.regions import TargetZone
from libero.libero.envs.objects import get_object_fn
from libero.libero.envs.predicates import eval_predicate_fn
from robosuite.utils.mjcf_utils import new_site

# LeRobot 补丁用
import lerobot.envs.libero as le_libero
from libero.libero.envs.problems import PROBLEM_INFO

# ==============================================================================
# 1. 定义 SOMA 专属任务类 (基于 BDDLBaseDomain)
# ==============================================================================
class SomaDistractorTask(BDDLBaseDomain):
    def __init__(self, *args, **kwargs):
        # --- A. 硬编码任务定义 (替代 BDDL 文件解析) ---
        # 这里定义了所有的物体、位置区域和目标
        self.parsed_problem = {
            "fixtures": {
                "kitchen_table": ["kitchen_table_1"]
            },
            "objects": {
                "tomato_soup_can": ["tomato_soup_can_1"], # 目标
                "spam": ["spam_1", "spam_2"],             # 干扰物
                "plate": ["plate_1"]                      # 放置容器
            },
            "regions": {
                # 1. 目标初始区 (正前方)
                "soup_init_region": {
                    "ranges": [[-0.02, -0.12, 0.02, -0.08]], # x_min, y_min, x_max, y_max
                    "rgba": [0, 1, 0, 0.1] # 绿色调试框
                },
                # 2. 干扰物1 (挡路)
                "distractor_front_region": {
                    "ranges": [[-0.02, -0.27, 0.02, -0.23]],
                    "rgba": [1, 0, 0, 0.1]
                },
                # 3. 干扰物2 (侧面)
                "distractor_side_region": {
                    "ranges": [[0.08, -0.12, 0.12, -0.08]],
                    "rgba": [1, 0, 0, 0.1]
                },
                # 4. 盘子区 (左后方)
                "plate_init_region": {
                    "ranges": [[-0.25, 0.15, -0.15, 0.25]],
                    "rgba": [0, 0, 1, 0.1]
                },
                # 5. 目标判定区 (盘子上方) - 用于判定成功
                "plate_target_region": {
                    "ranges": [[-0.25, 0.15, -0.15, 0.25]],
                    "rgba": [0, 1, 0, 0]
                }
            },
            # 初始状态配置 (将物体放入对应区域)
            "init_state": [
                ("On", "tomato_soup_can_1", "soup_init_region"),
                ("On", "spam_1", "distractor_front_region"),
                ("On", "spam_2", "distractor_side_region"),
                ("On", "plate_1", "plate_init_region")
            ],
            # 胜利条件: 汤罐头 在 盘子 上
            "goal_state": [
                ("On", "tomato_soup_can_1", "plate_1") 
            ]
        }

        # --- B. 环境配置 (照抄您的模板) ---
        self.workspace_name = "kitchen_table"
        self.visualization_sites_list = []
        self.kitchen_table_full_size = (1.0, 1.2, 0.05)
        self.kitchen_table_offset = (0.0, 0, 0.90)
        self.z_offset = 0.01 - self.kitchen_table_full_size[2]

        # 强制配置
        kwargs.update({"robots": [f"Mounted{robot_name}" for robot_name in kwargs.get("robots", ["Panda"])]})
        kwargs.update({"workspace_offset": self.kitchen_table_offset})
        kwargs.update({"arena_type": "kitchen"})
        if "scene_xml" not in kwargs:
            kwargs["scene_xml"] = "scenes/libero_kitchen_tabletop_base_style.xml"
        
        # 场景纹理属性
        if "scene_properties" not in kwargs:
            kwargs["scene_properties"] = {
                "floor_style": "gray-ceramic",
                "wall_style": "yellow-linen",
            }

        # 这一步非常关键：BDDLBaseDomain.__init__ 会尝试解析文件
        # 我们传一个空文件名，但在它解析前，我们已经填充了 self.parsed_problem
        # 只要我们覆盖了 _load_from_bddl 方法让它什么都不做，就不会报错
        super().__init__(bddl_file_name=None, *args, **kwargs)

    # 覆盖父类方法：禁用 BDDL 文件加载
    def _load_from_bddl(self, *args):
        pass 

    # 覆盖父类方法：加载初始状态 (这是 procedural generation 的核心)
    def _add_placement_initializer(self):
        # 遍历我们在 parsed_problem["init_state"] 里定义的规则
        for state in self.parsed_problem["init_state"]:
            predicate, obj_name, region_name = state
            if predicate == "On":
                # 将物体放入对应的 region
                self.placement_initializer.add_objects_to_region(
                    obj_name, region_name, 
                    ensure_object_boundary_in_range=False, 
                    ensure_valid_placement=True
                )
        super()._add_placement_initializer()

    # --- 以下直接沿用您的模板 ---
    
    def _load_fixtures_in_arena(self, mujoco_arena):
        for fixture_category in self.parsed_problem["fixtures"]:
            if fixture_category == "kitchen_table": continue
            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(name=fixture_instance)

    def _load_objects_in_arena(self, mujoco_arena):
        for category in self.parsed_problem["objects"]:
            for obj_name in self.parsed_problem["objects"][category]:
                self.objects_dict[obj_name] = get_object_fn(category)(name=obj_name)

    def _load_sites_in_arena(self, mujoco_arena):
        # 将我们定义的 ranges 转换为 Mujoco 的 Site (用于放置和可视化)
        for region_name, region_data in self.parsed_problem["regions"].items():
            ranges = region_data["ranges"][0]
            # 计算中心点和尺寸
            size = ((ranges[2]-ranges[0])/2, (ranges[3]-ranges[1])/2)
            centroid = ((ranges[2]+ranges[0])/2 + self.kitchen_table_offset[0], 
                        (ranges[3]+ranges[1])/2 + self.kitchen_table_offset[1])
            
            target_zone = TargetZone(
                name=region_name, rgba=region_data["rgba"], zone_size=size,
                z_offset=self.kitchen_table_offset[2], zone_centroid_xy=centroid
            )
            
            # 添加到 Arena
            mujoco_arena.table_body.append(new_site(
                name=target_zone.name, pos=target_zone.pos + np.array([0,0,-0.9]),
                quat=target_zone.quat, rgba=target_zone.rgba, size=target_zone.size, type="box"
            ))

    def _setup_camera(self, mujoco_arena):
        # 您的相机视角
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.6586, 0.0, 1.6103],
            quat=[0.6380, 0.3048, 0.3048, 0.6380],
        )
        mujoco_arena.set_camera(
            camera_name="frontview", pos=[1.0, 0.0, 1.48], quat=[0.56, 0.43, 0.43, 0.56]
        )

    def _check_success(self):
        # 检查 parsed_problem["goal_state"]
        result = True
        for state in self.parsed_problem["goal_state"]:
            result = result and self._eval_predicate(state)
        return result

# ==============================================================================
# 2. Mock Benchmark (骗过 LeRobot)
# ==============================================================================
class SomaPythonBenchmark:
    def __init__(self):
        self.n_tasks = 1
        self.tasks = ["SOMA_Distractor_Challenge"]
        
    def get_num_tasks(self): return self.n_tasks
    def get_task_names(self): return self.tasks
    def get_task_file(self, index): return None 

    class TaskWrapper:
        def __init__(self, name):
            self.name = name
            self.language = "pick up the soup can behind the distractors"
            self.problem_folder = name 
            self.bddl_file = None      

    def get_task(self, index):
        return self.TaskWrapper(self.tasks[index])

# ==============================================================================
# 3. 安装补丁
# ==============================================================================
def install_soma_task():
    task_name = "SOMA_Distractor_Challenge"
    
    # A. 注册到底层 PROBLEM_INFO
    # 注意：这里我们直接用我们定义的 class，而不是 scene_class
    # Libero 底层 logic: if type(problem) == type: problem = problem()
    if task_name not in PROBLEM_INFO:
        PROBLEM_INFO[task_name] = {
            "language": "pick up the soup can behind the distractors",
            "domain": "robosuite"
        }

    # B. 注入 Benchmark
    le_libero.BENCHMARK_MAPPING["libero_soma"] = lambda: SomaPythonBenchmark()

    # C. 劫持 LiberoEnv 初始化
    original_init = le_libero.LiberoEnv.__init__

    def patched_init(self, task_name_arg, *args, **kwargs):
        if task_name_arg == task_name:
            logging.info(f"✨ [SOMA_ENV] Booting custom BDDLBaseDomain task: {task_name}")
            
            self.task_id = 0
            self.task_name = task_name_arg
            self.task_description = "pick up the soup can behind the distractors"
            self.bddl_file_path = None
            
            # 直接实例化我们的类！
            # 这里的 robots 参数通常由 LiberoEnv 传入或默认
            # 我们从 kwargs 里提取，或者给默认值
            robots = kwargs.get("robots", ["Panda"])
            
            self._env = SomaDistractorTask(
                bddl_file_name=None, 
                robots=robots,
                render_mode=None, # 由 LeRobot 控制渲染
                control_freq=20
            )
            
            self.observation_space = self._env.observation_space
            self.action_space = self._env.action_space
            self._init_states = None
            return 

        return original_init(self, task_name_arg, *args, **kwargs)

    le_libero.LiberoEnv.__init__ = patched_init
    logging.info("✅ [SOMA_ENV] Installation complete. Using BDDLBaseDomain definition.")