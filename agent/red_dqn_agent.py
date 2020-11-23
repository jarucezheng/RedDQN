from enum import Enum
import math, json
import numpy as np
import agent.config as config
from agent.dqn import DQN
from agent.agent import Agent
from env.env_cmd import EnvCmd
from env.env_def import UnitType, UnitStatus, RED_AIRPORT_ID

AIR_PATROL_HEIGHT = 8000
A2G_PATROL_HEIGHT = 7000
AWACS_PATROL_HEIGHT = 7500
DISTURB_PATROL_HEIGHT = 8500

# 护卫舰
# 北部巡逻阵位
SHIP_POINT1 = [5000, 65000, 0]
# 中部巡逻阵位
SHIP_POINT2 = [15000, 0, 0]
# 南部巡逻阵位
SHIP_POINT3 = [25000, -65000, 0]
SHIP_PATROL_PARAMS_0 = [90, 20000, 20000, 190, 7200, 0]

# 无人机
# 北部侦察阵位(-105000, 65000)--距离北岛约35km, 突击北岛后可快速确认战果.
UAV_POINT1 = [-105000, 65000, 5000]
# 中部待命阵位(-125000, 0)--距离敌南/北岛距离均约88km, 距离己方北部干扰阵位约97km
UAV_POINT2 = [-125000, 0, 5000]
# 中北侦察阵位(-125000, 45000)--北岛南侧约47km, 突击北岛后快速确认战果(备选阵位)
UAV_POINT3 = [-125000, 45000, 5000]
# 中南侦察阵位(-125000, -45000)--南岛北侧约47km, 突击南岛后快速确认战果
UAV_POINT4 = [-125000, -45000, 5000]
# 南部待命阵位(-35000, -85000)--距离南岛约95km(敌地防视线内打击范围外), 距己北部干扰阵位约140km, 起吸引注意力作用
UAV_POINT5 = [-35000, -85000, 5000]
# 南部规避阵位(-35000, -55000)--向北部干扰阵位方向撤退30km
UAV_POINT6 = [-35000, -55000, 5000]
# 南部侦察阵位(-95000, -85000)--南岛东侧约37km, 突击南岛后快速确认战果
UAV_POINT7 = [-95000, -85000, 5000]
UAV_PATROL_PARAMS = [270, 20000, 20000, 190, 7200]
UAV_PATROL_PARAMS_0 = [270, 20000, 20000, 190, 7200, 0]
UAV_PATROL_PARAMS_1 = [270, 20000, 20000, 190, 7200, 1]

# 预警机
# 预警机待命阵位
AWACS_PATROL_POINT = [25000, 0, AWACS_PATROL_HEIGHT]
# 预警机南部规避阵位
AWACS_SOUTH_POINT = [55000, -30000, AWACS_PATROL_HEIGHT]
# 预警机北部规避阵位
AWACS_NORTH_POINT = [55000, 30000, AWACS_PATROL_HEIGHT]
# dir, len, wid, speed, time, mode:0:air/1:surface/2:both
AWACS_PATROL_PARAMS = [270, 20000, 20000, 160, 7200, 2]

# 空中干扰
# 空中干扰待命阵位
AIR_DISTURB_POINT1 = [45000, 0, DISTURB_PATROL_HEIGHT]
# 空中干扰北部干扰阵位
AIR_DISTURB_POINT2 = [-45000, 55000, DISTURB_PATROL_HEIGHT]
# 空中干扰北部干扰规避阵位
AIR_DISTURB_POINT3 = [-25000, 55000, DISTURB_PATROL_HEIGHT]
# 空中干扰南部干扰阵位
AIR_DISTURB_POINT4 = [-45000, -55000, DISTURB_PATROL_HEIGHT]
# 空中干扰南部干扰规避阵位
AIR_DISTURB_POINT5 = [-25000, -55000, DISTURB_PATROL_HEIGHT]


# 北部航线干扰
class Point():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


# NORTH_LINE = [{'x':45000,'y':0,'z':8500},{'x':-45000,'y':55000,'z':8500}]
NORTH_LINE = [Point(45000, 0, 8500), Point(25000,50000,8500),Point(-45000, 55000, 8500)]
# 南部航线干扰
# SOUTH_LINE = [{'x':-45000,'y':55000,'z':8500},{'x':-45000,'y':-55000,'z':8500}]
SOUTH_LINE = [Point(-45000, 55000, 8500), Point(-45000, -55000, 8500)]

DISTURB_PATROL_PARAMS = [270, 10000, 10000, 160, 7200]  # direction,length,width,speed,disturb_time,

# 对空作战
# 北部警戒阵位1(-65000, 25000)--距离北部突击阵位约40km
AIR_PATROL_POINT1 = [-65000, 25000, AIR_PATROL_HEIGHT]
# 北部警戒阵位2(-95000, 55000)--距离北部突击阵位约40km
AIR_PATROL_POINT2 = [-95000, 55000, AIR_PATROL_HEIGHT]
# 中部阻援/集结阵位(-105000, 0)--距离南/北部干扰阵位均约78km, 可伏击敌增援北岛战机, 距离南岛约90km, 也是南下的集结阵位
AIR_PATROL_POINT3 = [-105000, 0, AIR_PATROL_HEIGHT]
# 南部伏击阵位(-45000, -55000)--位于北部干扰阵位约110km(处于保护范围内), 可伏击敌南部出击的飞机
AIR_PATROL_POINT4 = [-45000, -55000, AIR_PATROL_HEIGHT]
# 南部警戒阵位1(-125000, -45000)--南岛北侧约47km, 距离己南部干扰阵位80km
AIR_PATROL_POINT5 = [-125000, -45000, AIR_PATROL_HEIGHT]
# 南部警戒阵位2(-105000, -65000)--南岛东北方向约37km
AIR_PATROL_POINT6 = [-105000, -65000, AIR_PATROL_HEIGHT]
# 南部警戒阵位3(-95000, -85000)--南岛东侧约37km, 距离己南部干扰阵位60km
AIR_PATROL_POINT7 = [-95000, -85000, AIR_PATROL_HEIGHT]

AIR_PATROL_PARAMS = [270, 10000, 10000, 160, 7200]
AIR_PATROL_PARAMS_0 = [270, 10000, 10000, 160, 7200, 0]
AIR_PATROL_PARAMS_1 = [270, 10000, 10000, 160, 7200, 1]

# 对地海打击
# 北部突击阵位(-55000, 65000)--北部干扰阵位的左上区域, 距离北岛约78km
AREA_HUNT_POINT1 = [-55000, 65000, A2G_PATROL_HEIGHT]
# AREA_HUNT_POINT0 = [-125000, 85000, A2G_PATROL_HEIGHT]
AREA_HUNT_POINT0 = [-129533.05624, 87664.0398, A2G_PATROL_HEIGHT]
# 北部规避阵位(-35000, 65000)--向己方半场后撤20km
AREA_HUNT_POINT2 = [-35000, 65000, A2G_PATROL_HEIGHT]
# 中部待命阵位(-115000, 0)--距离敌南/北岛距离均约88km, 距离己方北部干扰阵位约90km(处于保护范围内)
AREA_HUNT_POINT3 = [-115000, 0, A2G_PATROL_HEIGHT]
# 南部突击阵位(-55000, -65000)--南部干扰阵位的左下区域, 距离南岛约78km
AREA_HUNT_POINT4 = [-55000, -65000, A2G_PATROL_HEIGHT]
# AREA_HUNT_POINT4_0 = [-125000, -85000, A2G_PATROL_HEIGHT]
AREA_HUNT_POINT4_0 = [-131156.63859, -87887.86736, A2G_PATROL_HEIGHT]
# 南部规避阵位(-35000, -65000)--向己方半场后撤20km
AREA_HUNT_POINT5 = [-35000, -65000, A2G_PATROL_HEIGHT]

AREA_HUNT_PARAMS = [270, 1000, 1000]
AREA_PATROL_PARAMS = [270, 1000, 1000, 160, 7200]
AREA_HUNT_PARAMS_0 = [270, 1000, 1000, 160, 7200, 0]
AREA_HUNT_PARAMS_1 = [270, 1000, 1000, 160, 7200, 1]

AWACS_PATROL_TIME = 0
AWACS_ESCORT_TIME = 60

DISTURB_PATROL_TIME = 120
DISTURB_ESCORT_TIME = 280

AIR_PATROL_TIME11 = 300
AIR_PATROL_TIME12 = 420
AIR_DISTURB_TIME1 = 480

AIR_PATROL_TIME21 = 900
AIR_PATROL_TIME22 = 1020
AIR_DISTURB_TIME2 = 1080

AIR_PATROL_TIME31 = 1500
AIR_PATROL_TIME32 = 1620
AIR_DISTURB_TIME3 = 1680

AREA_HUNT_TIME11 = 2100
AREA_HUNT_TIME12 = 2400
HUNT_DISTURB_TIME1 = 3000

AREA_HUNT_TIME21 = 2700
AREA_HUNT_TIME22 = 3000
AREA_HUNT_TIME23 = 3300
HUNT_DISTURB_TIME2 = 3600

AIR_ATTACK_PERIOD = 10


class RedAgentState(Enum):
    first_time = 1
    second_time = 2
    AWACS_PATROL = 1
    AWACS_ESCORT = 2

    DISTURB_TAKEOFF = 3
    DISTURB_ESCORT = 4

    AIR_PATROL11 = 11
    AIR_PATROL12 = 12
    AIR_PATROL21 = 13
    AIR_PATROL22 = 14
    AIR_PATROL31 = 15
    AIR_PATROL32 = 16

    AREA_HUNT11 = 21
    AREA_HUNT12 = 22
    AREA_HUNT21 = 23
    AREA_HUNT22 = 24
    AREA_HUNT23 = 25
    AREA_HUNT24 = 26

    AIR_DISTURB1 = 51
    AIR_DISTURB2 = 52
    AIR_DISTURB3 = 53
    AIR_DISTURB4 = 54
    AIR_DISTURB5 = 55

    HUNT_DISTURB1 = 61
    HUNT_DISTURB2 = 62

    # 起飞全部结束
    END_TAKEOFF = 100
    END_DISTURB = 200


class RedDQNAgent(Agent):
    def __init__(self, name, config, **kwargs):
        super().__init__(name, config['side'])

        self._init()

    def _init(self):
        self.aircraft_dict = {}

        self.a2a_list = []
        self.target_list = []
        self.blue_list = []
        self.blue_dic = {}

        self.attacking_targets = {}

        self.awacs_team_id = -1
        self.disturb_team_id = -1

        self.agent_state = 0
        self.disturb_state = RedAgentState.AIR_DISTURB1
        self.area_hurt_a = RedAgentState.AREA_HUNT11
        self.area_hurt_b = RedAgentState.AREA_HUNT11
        self.area_hurt_c = RedAgentState.AREA_HUNT11
        self.area_hurt_d = RedAgentState.AREA_HUNT11
        self.air_attack_time = 0
        self.a2g_ha = 0
        self.a2g_hb = 0
        self.team_id_dic = {}
        self.Task = None
        self.dqn = DQN()
        self.reward = 0
        self.done = False
        ### 先随意初始化一个状态向量
        self.last_state = [10000]*config.a2a_LX11 + [6]*config.a2a_LX11 + \
                          [10000] * config.a2a_LX11 + [1] * config.a2a_LX11

    def reset(self):
        self._init()

    def step(self, sim_time,obs_red,**kwargs):

        self._parse_teams(obs_red)

        cmd_list = []
        self.done = False
        self._parse_observation(obs_red)
        # print('红方情报：',obs_red['qb'])
        '''第一波次'''
        # 护卫舰初始化
        if self.agent_state == 0:
            index = 1
            for ship in obs_red['units']:
                if ship['LX'] == 21:
                    if index == 1:
                        cmd_list.extend(self._ship_movedeploy(ship['ID'], SHIP_POINT1))
                        print('1号护卫舰就位')
                        index += 1
                        continue
                    if index == 2:
                        cmd_list.extend(self._ship_movedeploy(ship['ID'], SHIP_POINT2))
                        print('2号护卫舰就位')
                        index += 1
                        continue
                    if index == 3:
                        cmd_list.extend(self._ship_movedeploy(ship['ID'], SHIP_POINT3))
                        print('3号护卫舰就位')
                        index += 1
                        continue
            self.agent_state = 1

        # 预警机1架--YA + 护航歼击机2机编队--JA
        if self.agent_state == 1:
            for awas in obs_red['units']:
                if awas['LX'] == 12:
                    cmd_list.extend(self._awacs_patrol(awas['ID'], AWACS_PATROL_POINT, AWACS_PATROL_PARAMS))
                    print('预警机巡逻')
                    self.agent_state = 5
        if self.agent_state == 5:
            if 'YA' in list(self.team_id_dic.keys()):
                cmd_list.extend(self._awacs_escort(self.team_id_dic['YA']))
                print('给预警机护航')
                self.agent_state = 6

        # 干扰机3架(一起使用作360°干扰)--RA + 护航歼击机2机编队--JB
        # 正式发布版本会对模型进行聚合, 即只提供一架干扰机(所以这里作了修改)
        if self.agent_state == 6:
            cmd_list.extend(self._takeoff_areapatrol(1, 13, AIR_DISTURB_POINT1, DISTURB_PATROL_PARAMS))
            print('干扰机起飞')
            self.agent_state = 7
        if self.agent_state == 7:
            if 'RA' in list(self.team_id_dic.keys()):
                cmd_list.extend(self._disturb_escort(self.team_id_dic['RA']))
                print('给干扰机护航')
                self.agent_state = 8

        # 干扰机到达待定区域后下达区域干扰指令
        # 在待定区开启航线干扰
        if self.disturb_state == RedAgentState.AIR_DISTURB1:
            for disturb in obs_red['units']:
                if disturb['LX'] == 13 and 15000 < disturb['X'] < 75000 and -20000 < disturb['Y'] < 20000:
                    cmd_list.extend(self._disturb_linepatrol(self.team_id_dic['RA'], NORTH_LINE))
                    self.disturb_state = RedAgentState.AIR_DISTURB2
                    print('在待定区开启航线干扰')
        # 向北部指定区域行进干扰
        if self.disturb_state == RedAgentState.AIR_DISTURB2:
            for disturb in obs_red['units']:
                if disturb['LX'] == 13 and -50000 < disturb['X'] < -40000 and 50000 < disturb['Y'] < 60000:
                    cmd_list.extend(
                        self._disturb_patrol(self.team_id_dic['RA'], AIR_DISTURB_POINT2, DISTURB_PATROL_PARAMS))
                    self.disturb_state = RedAgentState.AIR_DISTURB3
                    print('在待定区开启区域干扰')
        # 干扰机南下干扰
        ship_flag = True
        if self.disturb_state == RedAgentState.AIR_DISTURB3:
            for blue_unit in obs_red['qb']:
                if blue_unit['LX'] == 21:
                    ship_flag = False
            # if ship_flag:
            if sim_time > 3000:
                cmd_list.extend(self._disturb_patrol(self.team_id_dic['RA'], AIR_DISTURB_POINT4, DISTURB_PATROL_PARAMS))
                print('干扰机南下干扰')
                self.disturb_state = RedAgentState.AIR_DISTURB4

        # 向南部指定区域行进干扰
        if self.disturb_state == RedAgentState.AIR_DISTURB4:
            for disturb in obs_red['units']:
                if disturb['LX'] == 13 and -50000 < disturb['X'] < -40000 and -60000 < disturb['Y'] < -50000:
                    cmd_list.extend(
                        self._disturb_patrol(self.team_id_dic['RA'], AIR_DISTURB_POINT4, DISTURB_PATROL_PARAMS))
                    self.disturb_state = RedAgentState.AIR_DISTURB5

        # 轰炸机2机编队--HA + 护航歼击机2机编队--JC
        if self.agent_state == 15:
            cmd_list.extend(self._takeoff_areapatrol(2, 15, AREA_HUNT_POINT1, AREA_PATROL_PARAMS))
            print('轰炸机HA起飞')
            self.agent_state = 9
        for ship in obs_red['qb']:
            if ship['LX'] == 21 and 'HA' in list(self.team_id_dic.keys()) and self.area_hurt_a == RedAgentState.AREA_HUNT11:
                cmd_list.extend(self._targethunt(self.team_id_dic['HA'], ship['ID']))
                self.area_hurt_a = RedAgentState.AREA_HUNT12
                print('HA进行目标突击，目标为蓝方舰船')
        if self.agent_state == 9:
            if 'HA' in list(self.team_id_dic.keys()):
                cmd_list.extend(self._A2G_escort(self.team_id_dic['HA']))
                print('给轰炸机HA护航')
                self.agent_state = 10
        # JC进入北部警戒阵位1
        if self.a2g_ha == 0 and 'HA' in list(self.team_id_dic.keys()):
            for a2g in obs_red['units']:
                if a2g['TMID'] == self.team_id_dic['HA'] and -75000 < a2g['X'] < -35000 and 50000 < a2g['Y'] < 80000:
                    for a2a in obs_red['units']:
                        if a2a['TMID'] == self.team_id_dic['JC']:
                            cmd_list.extend(self._areapatrol(a2a['ID'], AIR_PATROL_POINT1, AIR_PATROL_PARAMS_0))
                            print('JC进入北部警戒阵位1')
                    self.a2g_ha = 1

        # 轰炸机6机编队--HB + 护航歼击机2机编队--JD
        if self.agent_state == 10:
            cmd_list.extend(self._takeoff_areahunt(2, AREA_HUNT_POINT0))
            cmd_list.extend(self._takeoff_areahunt(4, AREA_HUNT_POINT0))
            print('轰炸机HB起飞')
            self.agent_state = 11

        if self.agent_state == 11:
            if 'HB' in list(self.team_id_dic.keys()):
                # cmd_list.extend(self._targethunt(self.team_id_dic['HB'], 5011))

                cmd_list.extend(self._A2G_escort(self.team_id_dic['HB']))
                print('给轰炸机HB护航')
                self.agent_state = 12
        # JD进入北部警戒阵位2
        if self.a2g_hb == 0 and 'HB' in list(self.team_id_dic.keys()):
            for a2g in obs_red['units']:
                if a2g['TMID'] == self.team_id_dic['HB'] and -75000 < a2g['X'] < -35000 and 50000 < a2g['Y'] < 80000:
                    for a2a in obs_red['units']:
                        if a2a['TMID'] == self.team_id_dic['JD']:
                            cmd_list.extend(self._areapatrol(a2a['ID'], AIR_PATROL_POINT2, AIR_PATROL_PARAMS_0))
                    self.a2g_hb = 1

        # 阻援歼击机2机编队--JE
        if self.agent_state == 8:
            cmd_list.extend(self._takeoff_areapatrol(2, 11, AIR_PATROL_POINT3, AIR_PATROL_PARAMS))
            print('阻援歼击机JE起飞')
            self.agent_state = 13
        # 歼击机JG, JH-->中部阻援阵位(第二阶段从中部南下警戒)
        if self.agent_state == 13:
            cmd_list.extend(self._takeoff_areapatrol(2, 11, [-105001, 0, 8000], AIR_PATROL_PARAMS))
            print('阻援歼击机JG起飞')
            self.agent_state = 14
        if self.agent_state == 14:
            cmd_list.extend(self._takeoff_areapatrol(2, 11, [-105002, 0, 8000], AIR_PATROL_PARAMS))
            print('阻援歼击机JH起飞')
            # print('阻援歼击机JI起飞')
            self.agent_state = 15
        '''第二波次'''

        # 如果干扰机存活则突击南部，如果干扰机不在，则突击北部
        hc = 0
        if self.agent_state == 12 and sim_time > 2000:
            for disturb in obs_red['units']:
                if disturb['LX'] == 13:
                    hc = 1
                    break
            if hc == 1:
                cmd_list.extend(self._takeoff_areahunt(2, AREA_HUNT_POINT4_0))
                print('轰炸机HC起飞突击南部')
                self.agent_state = 16
            else:
                cmd_list.extend(self._takeoff_areahunt(2, AREA_HUNT_POINT0))
                print('轰炸机HC起飞突击北部')
                self.agent_state = 16
        if self.agent_state == 16:
            if 'HC' in list(self.team_id_dic.keys()):
                cmd_list.extend(self._A2G_escort(self.team_id_dic['HC']))
                print('给轰炸机HC护航')
                self.agent_state = 17

        '''第三波次兵力列表(轰炸机6架, 歼击机4架)'''
        # 空战歼击机2机编队--JK
        if self.agent_state == 17:
            cmd_list.extend(self._takeoff_areapatrol(2, 11, AIR_PATROL_POINT7, AIR_PATROL_PARAMS))
            print('阻援歼击机JK起飞')
            self.agent_state = 18
        # 轰炸机6机编队--HD + 护航歼击机2机编队--JL
        if self.agent_state == 18:
            cmd_list.extend(self._takeoff_areahunt(2, AREA_HUNT_POINT4_0))
            cmd_list.extend(self._takeoff_areahunt(4, AREA_HUNT_POINT4_0))
            print('轰炸机HD起飞')
            self.agent_state = 19

        if self.agent_state == 19:
            if 'HD' in list(self.team_id_dic.keys()):
                cmd_list.extend(self._A2G_escort(self.team_id_dic['HD']))
                # cmd_list.extend(self._targethunt(self.team_id_dic['HD'], 5010))

                print('给轰炸机HD护航')
                self.agent_state = 20

        # 拦截
        blue_lx_list = []
        for blue_unit in obs_red['qb']:
            blue_lx_list.append(blue_unit['LX'])

            # 获取蓝方单位并且是存活状态

            if blue_unit['LX'] == 11 or blue_unit['LX'] == 15 or blue_unit['LX'] == 12:
                if blue_unit['WH'] == 1 and blue_unit['ID'] not in self.blue_list:

                    distances, fuel, weapons = [], [], []
                    a2a_units = []
                    for a2a in obs_red['units']:
                        if a2a['LX'] == 11 and a2a['Fuel'] > 3000 and '170' in list(a2a['WP'].keys()) and int(a2a['WP']['170']) > 0:

                            distance = math.sqrt(math.pow(a2a['X'] - blue_unit['X'], 2) + math.pow(a2a['Y'] - blue_unit['Y'], 2))
                            distances.append(distance)
                            weapons.append(int(a2a['WP']['170']))
                            fuel.append(a2a["Fuel"])
                            a2a_units.append(a2a)

                    if len(a2a_units) != 0:
                        live = len(distances)
                        dead = config.a2a_LX11 - live
                        distances = np.array(distances + [0] * dead)
                        fuel = np.array(fuel + [0] * dead)
                        weapons = np.array(weapons + [0] * dead)
                        dead_mask = np.array([1] * live + [0] * dead)
                        curr_state = np.hstack((distances, fuel, weapons, dead_mask))

                        who_index = self.dqn._choose_action(distances, fuel, weapons,dead_mask)
                        print("******************a2a_units length, index: ",len(a2a_units), who_index)
                        a2a = a2a_units[who_index]    
                        self.done = True

                        cmd_list.extend(self._airattack(a2a['ID'], blue_unit['ID']))
                        print("***************************执行战斗任务!!!!", a2a['ID'], "  ATTACK   ", blue_unit['ID'])
                        self.blue_list.append(blue_unit['ID'])
                        self.blue_dic[a2a['ID']] = blue_unit['ID']
        # 红方将蓝方单位击落或者红方拦截飞机被蓝方击落
        blue = 0
        del_blue = False
        del_blue2 = False
        for blue_target in self.blue_list:
            for blue_unit in obs_red['qb']:
                if blue_target == blue_unit['ID']:
                    blue = 1
                    break
            # 这里是判断我方位飞机是否存活
            for a2a_id in list(self.blue_dic.keys()):
                for a2a in obs_red['units']:
                    if a2a['ID'] == a2a_id:
                        del_blue = True
                        break
                if del_blue is False:
                    self.reward = -5
                    self.blue_dic.pop(a2a_id)
            for a2a_id in list(self.blue_dic.keys()):
                if blue_target == self.blue_dic[a2a_id]:
                    del_blue2 = True
                    break
            if blue == 0 or del_blue is False or del_blue2 is False:
                self.blue_list.remove(blue_target)
                # 需根据红方飞机当前状态重新下指令
                for a2a in obs_red['units']:
                    # 此时对状态为15 或 13 的蓝方飞机进行判断
                    if a2a['LX'] == 11 and a2a['ID'] in list(self.blue_dic.keys()) and self.blue_dic[a2a['ID']] == blue_target:
                        self.reward = 5
                        if a2a['ST'] == 15 or a2a['ST'] == 13:
                            # 如果油量小于4000或者子弹数量为0则返航，否者去预定区域进行区域巡逻
                            if a2a['Fuel'] < 4000 or int(a2a['WP']['170']) == 0:
                                cmd_list.extend(self._returntobase(a2a['ID']))
                            else:
                                for Tid in list(self.team_id_dic.keys()):
                                    if a2a['TMID'] == self.team_id_dic[Tid]:
                                        if Tid == 'JA':
                                            cmd_list.extend(
                                                self._areapatrol(a2a['ID'], AWACS_PATROL_POINT, AIR_PATROL_PARAMS_0))
                                        if Tid == 'JB':
                                            cmd_list.extend(
                                                self._areapatrol(a2a['ID'], AIR_DISTURB_POINT2, AIR_PATROL_PARAMS_0))
                                        if Tid == 'JC':
                                            cmd_list.extend(
                                                self._areapatrol(a2a['ID'], AIR_PATROL_POINT1, AIR_PATROL_PARAMS_0))
                                        if Tid == 'JD':
                                            cmd_list.extend(
                                                self._areapatrol(a2a['ID'], AIR_PATROL_POINT2, AIR_PATROL_PARAMS_0))
                                        if Tid == 'JE':
                                            cmd_list.extend(
                                                self._areapatrol(a2a['ID'], AIR_PATROL_POINT3, AIR_PATROL_PARAMS_0))
                                        if Tid == 'JF':
                                            cmd_list.extend(
                                                self._areapatrol(a2a['ID'], AIR_PATROL_POINT3, AIR_PATROL_PARAMS_0))
                                        if Tid == 'JG':
                                            cmd_list.extend(
                                                self._areapatrol(a2a['ID'], AIR_PATROL_POINT5, AIR_PATROL_PARAMS_0))
                                        if Tid == 'JH':
                                            cmd_list.extend(
                                                self._areapatrol(a2a['ID'], AIR_PATROL_POINT6, AIR_PATROL_PARAMS_0))
                                        if Tid == 'JI':
                                            cmd_list.extend(
                                                self._areapatrol(a2a['ID'], AIR_PATROL_POINT7, AIR_PATROL_PARAMS_0))
                                        if Tid == 'JJ':
                                            cmd_list.extend(
                                                self._areapatrol(a2a['ID'], AIR_PATROL_POINT7, AIR_PATROL_PARAMS_0))
                                        if Tid == 'JK':
                                            cmd_list.extend(
                                                self._areapatrol(a2a['ID'], AIR_PATROL_POINT7, AIR_PATROL_PARAMS_0))
                                        if Tid == 'JL':
                                            cmd_list.extend(
                                                self._areapatrol(a2a['ID'], AIR_PATROL_POINT4, AIR_PATROL_PARAMS_0))
        if self.done:
            self.dqn._store_transition(self.last_state, who_index, self.reward, curr_state)
            self.dqn._learning()
            self.last_state = curr_state
        self.done = False
        return cmd_list

    def _parse_observation(self, obs_red):
        self._parse_teams(obs_red)

    # 获取编队ID
    def _parse_teams(self, red_dict):
        for team in red_dict['teams']:
            if team['Task']:
                self.Task = json.loads(team['Task'])
                # print('self.Task:',self.Task)

            # 预警机编队
            if team['LX'] == UnitType.AWACS:
                self.team_id_dic['YA'] = team['TMID']

            # 干扰机编队
            elif team['LX'] == UnitType.DISTURB:
                if 'fly_num' in list(self.Task.keys()):
                    if self.Task['point_x'] == 45000 and self.Task['point_y'] == 0:
                        self.team_id_dic['RA'] = team['TMID']

            # 轰炸机编队
            elif team['LX'] == UnitType.A2G:
                if 'fly_num' in list(self.Task.keys()):
                    # HA
                    if self.Task['fly_num'] == 2 and self.Task['point_x'] == -55000 and self.Task['point_y'] == 65000:
                        self.team_id_dic['HA'] = team['TMID']
                    # HB
                    elif self.Task['fly_num'] == 4 and self.Task['point_x'] == -129533.05624 and self.Task[
                        'point_y'] == 87664.0398:
                        self.team_id_dic['HB'] = team['TMID']
                    # HC
                    elif self.Task['fly_num'] == 2 and self.Task['point_x'] == -131156.63859 and self.Task[
                        'point_y'] == -87887.86736:
                        self.team_id_dic['HC'] = team['TMID']
                    # HD
                    elif self.Task['fly_num'] == 4 and self.Task['point_x'] == -131156.63859 and self.Task[
                        'point_y'] == -87887.86736:
                        self.team_id_dic['HD'] = team['TMID']

            # 歼击机编队
            elif team['LX'] == UnitType.A2A:
                if self.Task['maintype'] == 'takeoffprotect':
                    # JA
                    if self.Task['cov_id'] == self.team_id_dic['YA']:
                        self.team_id_dic['JA'] = team['TMID']
                    # JB
                    elif self.Task['cov_id'] == self.team_id_dic['RA']:
                        self.team_id_dic['JB'] = team['TMID']
                    # JC
                    elif self.Task['cov_id'] == self.team_id_dic['HA']:
                        self.team_id_dic['JC'] = team['TMID']
                    # JD
                    elif self.Task['cov_id'] == self.team_id_dic['HB']:
                        self.team_id_dic['JD'] = team['TMID']
                    # JF
                    elif self.Task['cov_id'] == self.team_id_dic['HC']:
                        self.team_id_dic['JF'] = team['TMID']
                    # JL
                    elif self.Task['cov_id'] == self.team_id_dic['HD']:
                        self.team_id_dic['JL'] = team['TMID']
                else:
                    # JE
                    if 'fly_num' in list(self.Task.keys()):
                        if self.Task['fly_num'] == 2 and self.Task['point_x'] == -105000 and self.Task['point_y'] == 0:
                            self.team_id_dic['JE'] = team['TMID']
                    # JG
                    if 'fly_num' in list(self.Task.keys()):
                        if self.Task['fly_num'] == 2 and self.Task['point_x'] == -105001 and self.Task['point_y'] == 0:
                            self.team_id_dic['JG'] = team['TMID']
                    # JH
                    if 'fly_num' in list(self.Task.keys()):
                        if self.Task['fly_num'] == 2 and self.Task['point_x'] == -105002 and self.Task['point_y'] == 0:
                            self.team_id_dic['JH'] = team['TMID']
                    # JI
                    if 'fly_num' in list(self.Task.keys()):
                        if self.Task['fly_num'] == 2 and self.Task['point_x'] == -45000 and self.Task[
                            'point_y'] == -55000:
                            self.team_id_dic['JI'] = team['TMID']
                    # JJ
                    if 'fly_num' in list(self.Task.keys()):
                        if self.Task['fly_num'] == 2 and self.Task['point_x'] == -45001 and self.Task[
                            'point_y'] == -55000:
                            self.team_id_dic['JJ'] = team['TMID']
                    # JK
                    if 'fly_num' in list(self.Task.keys()):
                        if self.Task['fly_num'] == 2 and self.Task['point_x'] == -95000 and self.Task[
                            'point_y'] == -85000:
                            self.team_id_dic['JK'] = team['TMID']

    # 无人机出击
    @staticmethod
    def _uav_areapatrol(uav_id, uav_point, uav_params):
        return [EnvCmd.make_uav_areapatrol(uav_id, *uav_point, *uav_params)]

    # 预警机出击
    @staticmethod
    def _awacs_patrol(self_id, AWACS_PATROL_POINT, AWACS_PATROL_PARAMS):
        return [EnvCmd.make_awcs_areapatrol(self_id, *AWACS_PATROL_POINT, *AWACS_PATROL_PARAMS)]

    # 预警机护航
    def _awacs_escort(self, awacs_team_id):
        return [EnvCmd.make_takeoff_protect(RED_AIRPORT_ID, 2, awacs_team_id, 0, 100, 250)]

    # 干扰机进行区域干扰
    def _disturb_patrol(self, disturb_team_id, patrol_point, patrol_params):
        return [EnvCmd.make_disturb_areapatrol(disturb_team_id, *patrol_point, *patrol_params)]

    # 干扰机进行航线干扰
    def _disturb_linepatrol(self, self_id, point_list):
        return [EnvCmd.make_disturb_linepatrol(self_id, 160, 0, 'line', point_list)]

    # 轰炸机起飞突击
    @staticmethod
    def _takeoff_areahunt(num, area_hunt_point):
        return [EnvCmd.make_takeoff_areahunt(RED_AIRPORT_ID, num, 270, 80, *area_hunt_point, *[270, 1000, 1000, 160])]

    # 干扰机护航
    def _disturb_escort(self, disturb_team_id):
        return [EnvCmd.make_takeoff_protect(RED_AIRPORT_ID, 2, disturb_team_id, 1, 100, 250)]

    # 轰炸机护航
    def _A2G_escort(self, a2g_team_id):
        return [EnvCmd.make_takeoff_protect(RED_AIRPORT_ID, 2, a2g_team_id, 1, 100, 250)]

    # 起飞区域巡逻
    @staticmethod
    def _takeoff_areapatrol(num, lx, patrol_point, patrol_params):
        # patrol_params为5个参数
        return [EnvCmd.make_takeoff_areapatrol(RED_AIRPORT_ID, num, lx, *patrol_point, *patrol_params)]

    @staticmethod
    def _airattack(unit_id, target_id):
        return [EnvCmd.make_airattack(unit_id, target_id, 0)]

    # 区域巡逻
    @staticmethod
    def _areapatrol(unit_id, patrol_point, patrol_params):
        # patrol_params为6个参数
        return [EnvCmd.make_areapatrol(unit_id, *patrol_point, *patrol_params)]

    # 返航
    @staticmethod
    def _returntobase(unit_id):
        return [EnvCmd.make_returntobase(unit_id, 30001)]

    # 轰炸机目标突击
    @staticmethod
    def _targethunt(self_id, target_id):
        return [EnvCmd.make_targethunt(self_id, target_id, 270, 80)]

    # 轰炸机区域突击
    @staticmethod
    def _areahunt(self_id, point):
        return [EnvCmd.make_areahunt(self_id, 270, 80, *point, *AREA_HUNT_PARAMS)]

    # 护卫舰区域巡逻
    def _ship_areapatrol(self, self_id, point):
        return [EnvCmd.make_ship_areapatrol(self_id, *point, *SHIP_PATROL_PARAMS_0)]

    # 护卫舰初始化部署
    def _ship_movedeploy(self, self_id, point):
        return [EnvCmd.make_ship_movedeploy(self_id, *point, 90, 1)]
