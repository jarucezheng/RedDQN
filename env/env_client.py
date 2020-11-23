import json
import rpyc


class EnvClient(object):
    def __init__(self, server, port):
        """建立rpyc连接"""
        # server为宿主机IP地址; port为容器内部5455端口映射到外部宿主机的端口号
        # self.conn = rpyc.connect(server, port)
        self.conn = rpyc.connect(server, port, config={'sync_request_timeout': 120})

    def get_observation(self):
        """获取态势"""
        response = self.conn.root.get_state()
        ob_data = json.loads(response['json_data'])
        return ob_data

    def take_action(self, cmd_list):
        """下发指令"""
        # cmd_list为智能体生成的指令列表
        data_json = json.dumps(cmd_list)
        self.conn.root.take_action(data_json)
        return True
