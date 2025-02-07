class Settings():
    """存储《外星人入侵》的所有设置的类"""
    def __init__(self):
        """初始化游戏的设置"""
        # 屏幕设置
        self.screen_width = 1200
        self.screen_height = 800
        self.bg_color =(255, 255, 255)

        # 飞船的位置
        self.ship_limit = 3

        # 子弹设置
        self.bullet_width = 6
        self.bullet_height = 15
        self.bullet_color = 60, 60, 60
        self.bullet_allowed = 3

        # 外星人设置
        self.fleet_drop_speed = 40

        

        # 以什么样的速度加快游戏节奏
        self.speed_scale = 1.1

        # 外星人点数的提高速度
        self.score_scale = 1.5

        self.initialize_dynamic_settings()
    
    def initialize_dynamic_settings(self):
        """初始化随游戏进行而变化的设置"""
        self.ship_speed_factor = 1.5
        self.bullet_speed_factor = 3
        self.alien_speed_factor = 1

        # fleet_direction 1表示向右移动，-1表示向左移动
        self.fleet_direction = 1

        # 计分
        self.alien_points = 50
    
    def increase_speed(self):
        """提高速度设置"""
        self.ship_speed_factor *= self.speed_scale
        self.bullet_speed_factor *= self.speed_scale
        self.alien_speed_factor *= self.speed_scale
        self.alien_points = int(self.alien_points * self.score_scale)
        print(self.alien_points)

        