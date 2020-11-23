import time
import random
import os
import subprocess

from env.env_manager import EnvManager
from env.env_client import EnvClient


class EnvRunner(object):
    def __init__(self, env_id, server_port, agents, config):
        """对战环境初始化"""
        # env_id为仿真环境编号; server_port为基准端口号, 由此生成映射到容器内部各端口的宿主机端口号
        # agents为描述对战双方智能体的字典; config为仿真环境的基本配置信息(想定、镜像名等)
        super().__init__()

        random.seed(os.getpid() + env_id)

        self.env_id = env_id
        self.server_port = server_port + env_id
        self.config = config
        self.volume_list = self.config['volume_list']
        self.max_game_len = self.config['max_game_len']

        scene_name = self.config['scene_name']
        prefix = self.config['prefix']
        image_name = self.config['image_name']
        # 构建管理本地容器化仿真环境的实例对象
        self.env_manager = EnvManager(self.env_id, server_port, scene_name, prefix, image_name=image_name)

        self.env_client = None
        self.agents_conf = agents
        self.agents = self._init_agents()

    def __del__(self):
        # self.env_manager.stop_docker()  # 停止并删除旧环境
        pass

    def _start_env(self):
        """启动仿真环境"""
        # 查找是否存在同名的旧容器, 有的话先删除再启动新环境
        docker_name = 'env_{}'.format(self.env_id)
        docker_search = 'docker ps -a --filter name=^/{}$'.format(docker_name)
        # 捕获终端输出结果
        p = subprocess.Popen(docker_search, stdout=subprocess.PIPE, shell=True)
        out, err = p.communicate()
        # decode()将bytes对象转成str对象, strip()删除头尾字符空白符
        # split()默认以分隔符, 包括空格, 换行符\n, 制表符\t来对字符串进行分割
        out_str = out.decode()
        str_split = out_str.strip().split()
        if docker_name in str_split:
            print('存在同名旧容器,先删除之\n', out_str)
            self.env_manager.stop_docker()
        self.env_manager.start_docker(self.volume_list)  # 启动新环境
        time.sleep(10)

    def _init_agents(self):
        """根据配置信息构建红蓝双方智能体"""
        agents = []
        for name, agent_conf in self.agents_conf.items():
            cls = agent_conf['class']
            agent = cls(name, agent_conf)
            agents.append(agent)
        return agents

    def _reset(self):
        """智能体重置"""
        for agent in self.agents:
            agent.reset()

    def _run_env(self):
        pass

    def _get_observation(self):
        """态势获取"""
        return self.env_client.get_observation()

    def _run_actions(self, actions):
        """客户端向服务端发送指令"""
        self.env_client.take_action(actions)

    def _run_agents(self, observation):
        """调用智能体的step方法生成指令, 然后发送指令"""
        for agent in self.agents:
            side = agent.side
            sim_time = observation['sim_time']
            obs_red = observation['red']
            obs_blue = observation['blue']
            if side == 'red':
                actions = agent.step(sim_time, obs_red)
                self._action_validate(actions, obs_red, side, sim_time)
                self._run_actions(actions)
            else:
                actions = agent.step(sim_time, obs_blue)
                self._action_validate(actions, obs_blue, side, sim_time)
                self._run_actions(actions)

    @staticmethod
    def _get_done(observation):
        """推演是否结束"""
        sim_time = observation['sim_time']
        obs_blue = observation['blue']
        cmd_posts = [u for u in obs_blue['units'] if u['LX'] == 41]
        done = [False, 0, 0]    # [对战是否结束, 红方得分, 蓝方得分]
        # 若蓝方态势平台信息里没有指挥所, 说明两个指挥所已经被打掉
        if len(cmd_posts) == 0 or sim_time >= 9000:
            done[0] = True
        if len(cmd_posts) == 0:     # red win
            done[1:] = [1, -1]
        elif len(cmd_posts) == 1:   # tie
            done[1:] = [0, 0]
        elif len(cmd_posts) == 2:   # blue win
            done[1:] = [-1, 1]
        return done

    def _action_validate(self, actions, obs_own, side, sim_time):
        """指令有效性基本检查"""
        # 检查项: 执行主体, 目标, 护航对象, 机场相关指令是否有效
        # 检查项: 地防/护卫舰的初始部署位置只能在己方半场, 且只允许在开局2分钟内进行调整
        for action in actions:
            maintype = action['maintype']
            if maintype in ['Ship_Move_Deploy', 'Ground_Move_Deploy']:
                pos2d = (action['point_x'], action['point_y'])
                self._validate_deploy(side, pos2d, sim_time)
            if 'self_id' in action:
                self_id = int(action['self_id'])
                self._validate_self_id(maintype, self_id, obs_own)
            if 'target_id' in action:
                target_id = int(action['target_id'])
                self._validate_target_id(target_id, obs_own)
            if 'cov_id' in action:
                cov_id = int(action['cov_id'])
                self._validate_cov_id(cov_id, obs_own)
            if 'airport_id' in action:
                airport_id = int(action['airport_id'])
                self._validate_airport(airport_id, action, obs_own)

    @staticmethod
    def _validate_deploy(side, pos2d, sim_time):
        assert sim_time <= 120
        if side == 'red':
            assert pos2d[0] >= 0
        else:
            assert pos2d[0] <= 0

    @staticmethod
    def _validate_self_id(maintype, self_id, obs_own):
        """判断执行主体是否有效"""
        # 空中拦截指令(make_airattack)执行主体只能是单平台(需要选手考虑目标分配问题)
        # 返航指令(make_returntobase)执行主体可以是单平台也可以是编队(考虑编队内单机油量不足提前返航)
        # 其他指令执行主体原则上必须是编队, 为防止出错目前服务端也支持给单平台下指令.
        unit = [u for u in obs_own['units'] if u['ID'] == self_id]
        team = [u for u in obs_own['teams'] if u['TMID'] == self_id]
        if maintype == 'airattack':
            if len(unit) == 0:
                raise Exception("无效平台编号%s" % self_id)
        else:
            if len(unit) == 0 and len(team) == 0:
                raise Exception("无效执行主体编号%s" % self_id)
            else:
                obj = unit[0] if len(unit) > 0 else team[0]
                if obj['LX'] not in type4cmd[maintype]:
                    raise Exception("类型为%s的编队%s无法执行%s指令" % (obj[0]['LX'], self_id, maintype))

    @staticmethod
    def _validate_target_id(target_id, obs_own):
        """判断目标编号是否合法"""
        # 所有指令的目标编号, 必须是敌方单平台号
        unit = [u for u in obs_own['qb'] if u['ID'] == target_id]
        if len(unit) == 0:
            raise Exception("无效目标平台编号%s" % target_id)

    @staticmethod
    def _validate_cov_id(cov_id, obs_own):
        """护航对象是否有效"""
        # 护航对象编号, 必须为己方编队号
        team = [u for u in obs_own['teams'] if u['TMID'] == cov_id]
        if len(team) == 0:
            raise Exception("无效护航对象编号%s" % cov_id)
        else:
            type4cov = [12, 13, 15]
            if int(team[0]['LX']) not in type4cov:
                raise Exception("非法护航目标类型%s" % team[0]['LX'])

    @staticmethod
    def _validate_airport(airport_id, action, obs_own):
        """判断机场相关指令的有效性"""
        airports = obs_own['airports']
        airport = [u for u in airports if u['ID'] == airport_id]
        # 根据机场情况判断指令合法性
        maintype = action['maintype']
        if len(airport) == 0:
            raise Exception("无效机场编号%s" % airport_id)
        elif maintype == 'returntobase':
            pass
        else:
            obj = airport[0]
            if not obj['WH']:
                raise Exception("机场%s修复中无法执行起飞指令" % obj['ID'])
            if maintype == 'takeoffprotect':
                fly_type = 11   # 起飞护航指令默认起飞歼击机
            elif maintype in ['takeoffareahunt', 'takeofftargethunt']:
                fly_type = 15   # 起飞突击类指令默认起飞轰炸机
            else:
                fly_type = action['fly_type']
            fly_num = action['fly_num']
            if int(fly_type) not in type4cmd[maintype]:
                raise Exception("机场无法起降类型为%s的单位" % fly_type)
            type_map = {11: 'AIR', 12: 'AWCS', 13: 'JAM', 14: 'UAV', 15: 'BOM'}
            attr = type_map[fly_type]
            if fly_num > obj[attr]:
                print('指令>>>', action)
                raise Exception("起飞数量%d大于机场可起飞数量%d" % (fly_num, obj[attr]))


# 不同指令可执行主体的类型列表
type4cmd = {
    # 作战飞机
    "areapatrol": [11, 12, 13, 14, 15],
    "takeoffareapatrol": [11, 12, 13, 14, 15],
    "linepatrol": [11, 12, 13, 14, 15],
    "takeofflinepatrol": [11, 12, 13, 14, 15],
    "areahunt": [15],
    "takeoffareahunt": [15],
    "targethunt": [15],
    "takeofftargethunt": [15],
    "protect": [11],
    "takeoffprotect": [11],
    "airattack": [11],
    "returntobase": [11, 12, 13, 14, 15],
    # 地防
    "Ground_Add_Target": [31],
    "Ground_Remove_Target": [31],
    "GroundRadar_Control": [31],
    "Ground_Set_Direction": [31],
    "Ground_Move_Deploy": [31],
    # 护卫舰
    "Ship_Move_Deploy": [21],
    "Ship_areapatrol": [21],
    "Ship_Add_Target": [21],
    "Ship_Remove_Target": [21],
    "Ship_Radar_Control": [21],
    # 预警机
    "awcs_areapatrol": [12],
    "awcs_linepatrol": [12],
    "awcs_mode": [12],
    "awcs_radarcontrol": [12],
    "awcs_cancledetect": [12],
    # 干扰机
    "area_disturb_patrol": [13],
    "line_disturb_patrol": [13],
    "set_disturb": [13],
    "close_disturb": [13],
    "stop_disturb": [13],
    # 无人侦察机
    "uav_areapatrol": [14],
    "uav_linepatrol": [14],
    "uav_cancledetect": [14],
    # 地面雷达
    "base_radarcontrol": [32]
}
