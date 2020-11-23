from enum import Enum
import math,json

from agent.agent import Agent
from env.env_cmd import EnvCmd
from env.env_def import UnitType, UnitStatus, BLUE_AIRPORT_ID



# 预警机
# 预警机待命阵位
AWACS_PATROL_POINT = [45000, 0, 7500]

# dir, len, wid, speed, time, mode:0:air/1:surface/2:both
AWACS_PATROL_PARAMS = [270, 20000, 20000, 160, 7200, 2]

AREA_PATROL_HEIGHT = 7000

PATROL_POINT1 = [-5000, 65000, AREA_PATROL_HEIGHT]
PATROL_POINT2 = [-5000, -65000, AREA_PATROL_HEIGHT]
PATROL_POINT3 = [-55000, 35000, AREA_PATROL_HEIGHT]
PATROL_POINT4 = [-55000, -35000, AREA_PATROL_HEIGHT]
PATROL_POINT5 = [-65000, 5000, AREA_PATROL_HEIGHT]
PATROL_POINT6 = [-65000, -5000, AREA_PATROL_HEIGHT]

PATROL_POINT11 = [-80000, 75000, AREA_PATROL_HEIGHT]
PATROL_POINT12 = [-80000, 45000, AREA_PATROL_HEIGHT]
PATROL_POINT13 = [-80000, 15000, AREA_PATROL_HEIGHT]
PATROL_POINT14 = [-80000, -15000, AREA_PATROL_HEIGHT]
PATROL_POINT15 = [-80000, -45000, AREA_PATROL_HEIGHT]
PATROL_POINT16 = [-80000, -75000, AREA_PATROL_HEIGHT]

PATROL_AREA_LEN = 30000
PATROL_AREA_WID = 30000
PATROL_DIRECTION = 90
PATROL_SPEED = 250
PATROL_TIME = 7200
PATROL_MODE_0 = 0
PATROL_MODE_1 = 1
PATROL_PARAMS = [PATROL_DIRECTION, PATROL_AREA_LEN, PATROL_AREA_WID, PATROL_SPEED, PATROL_TIME]
PATROL_PARAMS_0 = [PATROL_DIRECTION, PATROL_AREA_LEN, PATROL_AREA_WID, PATROL_SPEED, PATROL_TIME, PATROL_MODE_0]
PATROL_PARAMS_1 = [PATROL_DIRECTION, PATROL_AREA_LEN, PATROL_AREA_WID, PATROL_SPEED, PATROL_TIME, PATROL_MODE_1]

PATROL_TIME1 = 0
PATROL_TIME2 = 60
PATROL_TIME3 = 120
PATROL_TIME4 = 180
PATROL_TIME5 = 240
PATROL_TIME6 = 300

PATROL_TIME11 = 600
PATROL_TIME12 = 900
PATROL_TIME13 = 1200
PATROL_TIME14 = 1500
PATROL_TIME15 = 1800
PATROL_TIME16 = 2100

AIR_ATTACK_PERIOD = 10


class BlueAgentState(Enum):

    PATROL0 = 0
    PATROL1 = 1
    PATROL2 = 2
    PATROL3 = 3
    PATROL4 = 4
    PATROL5 = 5
    PATROL6 = 6

    PATROL11 = 11
    PATROL12 = 12
    PATROL13 = 13
    PATROL14 = 14
    PATROL15 = 15
    PATROL16 = 16

    END_TAKEOFF = 100


class BlueRuleAgent(Agent):
    def __init__(self, name, config, **kwargs):
        super().__init__(name, config['side'])

        self._init()

    def _init(self):
        self.command_list = []
        self.ship_list = []
        self.s2a_list = []
        self.radar_list = []
        self.aircraft_dict = {}

        self.a2a_list = []
        self.a2a_attack_list = []  # 已经收到拦截指令的飞机
        self.target_list = []
        self.target_ship_list = []
        self.red_dic = {}
        self.attacking_targets = {}

        self.agent_state = BlueAgentState.PATROL0
        self.air_attack_time = 0
        self.airport_flag_time = 9000
        self.airport_flag = False
        self.tm_a2a = 0

    def reset(self):
        self._init()

    def step(self, sim_time,obs_blue, **kwargs):
        # with open('wwwwwww.json','w',encoding='utf-8') as f:
        #     json.dump(obs_blue,f)
        curr_time = sim_time
        cmd_list = []

        # 第一轮起飞巡逻
        if self.agent_state == BlueAgentState.PATROL0 and curr_time >= PATROL_TIME1:
            cmd_list.extend(self._takeoff_areapatrol(1, 11, PATROL_POINT1, PATROL_PARAMS))
            self.agent_state = BlueAgentState.PATROL1

        elif self.agent_state == BlueAgentState.PATROL1 and curr_time >= PATROL_TIME2:
            for awas in obs_blue['units']:
                if awas['LX'] == 12:
                    cmd_list.extend(self._awacs_patrol(awas['ID'], [-100000, 0, 7500], AWACS_PATROL_PARAMS))
                    self.agent_state = BlueAgentState.PATROL2

        elif self.agent_state == BlueAgentState.PATROL2 and curr_time >= PATROL_TIME2:
            cmd_list.extend(self._takeoff_areapatrol(1, 11, PATROL_POINT1, PATROL_PARAMS))
            self.agent_state = BlueAgentState.PATROL3
        elif self.agent_state == BlueAgentState.PATROL3 and curr_time >= PATROL_TIME3:
            cmd_list.extend(self._takeoff_areapatrol(1, 11, PATROL_POINT2, PATROL_PARAMS))
            self.agent_state = BlueAgentState.PATROL4
        elif self.agent_state == BlueAgentState.PATROL4 and curr_time >= PATROL_TIME4:
            cmd_list.extend(self._takeoff_areapatrol(1, 11, PATROL_POINT3, PATROL_PARAMS))
            self.agent_state = BlueAgentState.PATROL5
        elif self.agent_state == BlueAgentState.PATROL5 and curr_time >= PATROL_TIME5:
            cmd_list.extend(self._takeoff_areapatrol(1, 11, PATROL_POINT4, PATROL_PARAMS))
            self.agent_state = BlueAgentState.PATROL6
        elif self.agent_state == BlueAgentState.PATROL6 and curr_time >= PATROL_TIME6:
            cmd_list.extend(self._takeoff_areapatrol(1, 11, PATROL_POINT5, PATROL_PARAMS))
            self.agent_state = BlueAgentState.PATROL11

        # print('蓝方：', self.red_dic)
        # print('蓝方：', self.target_list)
        # 发现红方，立马攻击，并且派最近的编队去支援，同时机场再派出替补编队去替补支援编队空出的位置
        if obs_blue['qb']:
            for red_unit in obs_blue['qb']:

                # 获取红方单位并且是存活状态
                if red_unit['LX'] == 13 or red_unit['LX'] == 11 or red_unit['LX'] == 12 or \
                        red_unit['LX'] == 15:
                    if red_unit['WH'] == 1 and red_unit['ID'] not in self.target_list:
                        # print(self.target_list)
                        dic_distance = {}
                        for a2a in obs_blue['units']:
                            # 根据飞机当前的状态，如果油量大于3000并且弹药大于0，则执行以下逻辑
                            if a2a['LX'] == 11 and a2a['Fuel'] > 3000 and '170' in list(a2a['WP'].keys()) and int(a2a['WP']['170']) > 0:
                                # 计算蓝方飞机与情报中红方飞机的距离，取最近的1个过去拦截
                                distance = math.sqrt(
                                    math.pow(a2a['X'] - red_unit['X'], 2) + math.pow(a2a['Y'] - red_unit['Y'], 2))
                                dic_distance[distance] = a2a
                        list_distance = list(dic_distance.keys())
                        list_distance.sort()
                        # 派一架飞机去拦截
                        for dis in list_distance[:1]:
                            # 如果油量小于3000或者子弹数量为0则返航，否者原地进行区域巡逻
                            # if dic_distance[dis]['Fuel'] < 3000 or int(dic_distance[dis]['WP']['170']) == 0:
                            #     cmd_list.extend(self._returntobase(dic_distance[dis]['ID']))
                            # else:
                            # 拦截
                            cmd_list.extend(self._airattack(dic_distance[dis]['ID'], red_unit['ID']))
                            self.target_list.append(red_unit['ID'])
                            # print('添加打击目标id：', red_unit['ID'])
                            self.red_dic[dic_distance[dis]['ID']] = red_unit['ID']
                            # print(f"蓝方{dic_distance[dis]['ID']}打击红方{red_unit['ID']}")

                        # 派出替补编队进行替补(需考虑二次起飞的弹药补给时间1200s)
                        for dis in list_distance[:1]:
                            point_T = [dic_distance[dis]['X'], dic_distance[dis]['Y'], 7000]
                            for airports in obs_blue['airports']:
                                if airports['AIR'] == 0:
                                    self.airport_flag = True
                                if self.airport_flag and airports['AIR'] >= 1:
                                    self.airport_flag_time = sim_time
                                    self.airport_flag = False
                                # 补给时间过20min
                                if (sim_time - self.airport_flag_time) > 1200 and airports['AIR'] > 0:
                                    cmd_list.extend(self._takeoff_areapatrol(1, 11, point_T, PATROL_PARAMS))
                                    print('蓝方派出补给完成的替补编队进行替补')
                                # 初次起飞飞机
                                if self.airport_flag is False and self.airport_flag_time == 9000 and self.tm_a2a < 6:
                                    cmd_list.extend(self._takeoff_areapatrol(1, 11, PATROL_POINT1, PATROL_PARAMS))
                                    print('蓝方派出初次替补编队进行替补')
                                    self.tm_a2a += 1

                # 获取红方舰船
                if red_unit['LX'] == 21 and red_unit['ID'] not in self.target_ship_list:
                    cmd_list.extend(
                        self._takeoff_areahunt(2, [red_unit['X'], red_unit['Y'], 5000], [90, 1000, 1000, 160]))
                    self.target_ship_list.append(red_unit['ID'])
                    print('蓝方派出轰炸机攻击红方护卫舰')
        # 如果没有情报就按计划区域巡逻
        else:
            self.target_list = []

        # 蓝方将红方单位击落或者蓝方拦截飞机被红方击落
        red = 0
        del_red = False
        del_red2 = False
        # 敌方在打击列表中
        for red_target in self.target_list:
            # 判断是否在情报里，不在就从打击列表中删除
            for red_unit in obs_blue['qb']:
                if red_target == red_unit['ID']:
                    red = 1
                    break
            # 判断我方飞机是否存活，若死亡则把敌方从打击列表中删除
            for a2a_id in list(self.red_dic.keys()):
                for a2a in obs_blue['units']:
                    if a2a['ID'] == a2a_id:
                        del_red = True
                        break
                if del_red is False:
                    # print(f"蓝方{a2a_id}已死亡")
                    self.red_dic.pop(a2a_id)
            # 判断此打击目标是否有我方飞机去打击（中途目标可能会改变）
            for a2a_id in list(self.red_dic.keys()):
                if red_target == self.red_dic[a2a_id]:
                    del_red2 = True
                    break
            if red == 0 or del_red is False or del_red2 is False:
                self.target_list.remove(red_target)
                # print('删除目标id：', red_target)
                # 如果红方单位已被歼灭，需根据蓝方飞机当前状态重新下指令
                for a2a in obs_blue['units']:
                    # 此时对状态为15 或 13 的蓝方飞机进行判断
                    # if a2a['LX'] == 11:
                        # print(a2a['ID'],a2a['ST'],self.red_dic,red_target)
                    if a2a['LX'] == 11 and a2a['ID'] in list(self.red_dic.keys()) and self.red_dic[a2a['ID']] == red_target:
                        # print('打击目标死亡后，对蓝方飞机重新下指令~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                        if a2a['ST'] == 15 or a2a['ST'] == 13:
                            # 如果油量小于3000或者子弹数量为0则返航，否者原地进行区域巡逻
                            if a2a['Fuel'] < 3000 or int(a2a['WP']['170']) == 0:
                                cmd_list.extend(self._returntobase(a2a['ID']))
                            else:
                                if int(a2a['Y']) > 35000:
                                    point_now = PATROL_POINT1
                                    cmd_list.extend(self._areapatrol(a2a['ID'], point_now, PATROL_PARAMS_0))
                                elif 0 < int(a2a['Y']) <= 35000:
                                    point_now = PATROL_POINT3
                                    cmd_list.extend(self._areapatrol(a2a['ID'], point_now, PATROL_PARAMS_0))
                                elif -35000 <= int(a2a['Y']) <= 0:
                                    point_now = PATROL_POINT4
                                    cmd_list.extend(self._areapatrol(a2a['ID'], point_now, PATROL_PARAMS_0))
                                elif int(a2a['Y']) < -35000:
                                    point_now = PATROL_POINT2
                                    cmd_list.extend(self._areapatrol(a2a['ID'], point_now, PATROL_PARAMS_0))

        return cmd_list

    @staticmethod
    def _takeoff_areapatrol(num, lx, patrol_point, patrol_params):
        return [EnvCmd.make_takeoff_areapatrol(BLUE_AIRPORT_ID, num, lx, *patrol_point, *patrol_params)]

    # 区域巡逻
    @staticmethod
    def _areapatrol(unit_id, patrol_point, patrol_params):
        return [EnvCmd.make_areapatrol(unit_id, *patrol_point, *patrol_params)]

    @staticmethod
    def _airattack(unit_id, target_id):
        return [EnvCmd.make_airattack(unit_id, target_id, 1)]

    # 返航
    @staticmethod
    def _returntobase(unit_id):
        return [EnvCmd.make_returntobase(unit_id, 20001)]

    # 预警机出击
    @staticmethod
    def _awacs_patrol(self_id, AWACS_PATROL_POINT, AWACS_PATROL_PARAMS):
        return [EnvCmd.make_awcs_areapatrol(self_id, *AWACS_PATROL_POINT, *AWACS_PATROL_PARAMS)]

    # 轰炸机起飞
    @staticmethod
    def _takeoff_areahunt(num, area_hunt_point, area_hunt_area):
        return [EnvCmd.make_takeoff_areahunt(20001, num, 90, 100, *area_hunt_point, *area_hunt_area)]
