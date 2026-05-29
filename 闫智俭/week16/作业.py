#核心架构：Game Engine（裁判与规则）
class GameEngine:
    def __init__(self, agents):
        self.state = GameState()  # 真实世界状态
        self.agents = agents      # Agent列表
        self.round = 0

    def run(self):
        while not self.check_game_over():
            self.night_phase()
            self.day_phase()
            self.round += 1

    def night_phase(self):
        # 1. 狼人讨论 (仅狼人可见)
        # 2. 狼人杀人
        # 3. 预言家查验
        # 4. 女巫救人或毒
        pass

    def day_phase(self):
        # 1. 公布死亡信息
        # 2. 依次发言 (Speech Chain)
        # 3. 投票
        pass
      #灵魂设计：Agent Memory & Prompt（智能体大脑）
      #记忆系统 (Memory System)
      class AgentMemory:
    def __init__(self, role):
        self.role = role
        self.beliefs = {}  # 对其他人的怀疑度 {"Player1": 0.8}
        self.last_action = None
        self.dialogue_history = [] # 仅保留最近N轮
#动态 Prompt 工程 (The Secret Sauce)
WOLF_PROMPT = """
[身份设定]
你是一名狼人，你的队友是 {teammates}。你的目标是伪装成好人并屠杀村民。

[当前局势]
当前是第 {round} 天。
昨晚死亡的是：{dead_player}。
你的私有信息：{private_info}。

[历史记录]
{history}

[思考步骤] (强制要求一步步想)
1. 分析谁在带节奏？
2. 谁在查杀我？
3. 我应该选择悍跳还是倒钩？
4. 我的发言策略是什么？

[输出格式]
请输出你的发言（不超过50字）：
"""
#可视化与日志 (Logging & UI)
{
  "game_id": "uuid",
  "round": 1,
  "phase": "night",
  "action": "kill",
  "actor": "Wolf_1",
  "target": "Villager_3",
  "visible_to": ["Wolf_1", "Wolf_2"] // 信息隔离的关键
}
