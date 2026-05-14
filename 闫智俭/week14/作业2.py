import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===================== 股票 Skill 类 =====================
class StockVisualSkill:
    def __init__(self, ticker: str, period: str = "3mo"):
        """
        初始化股票 Skill
        :param ticker: 股票代码 如 AAPL 000001.SS
        :param period: 爬取周期 默认3个月
        """
        self.ticker = ticker
        self.period = period
        self.df = None  # 日K数据
        self.weekly_df = None  # 周K数据

    def fetch_data(self):
        """获取股票数据"""
        print(f"📊 正在获取 {self.ticker} 股票数据...")
        stock = yf.Ticker(self.ticker)
        # 获取日K数据
        self.df = stock.history(period=self.period)
        
        # 计算日波动：当日最高价 - 最低价
        self.df["日波动"] = self.df["High"] - self.df["Low"]
        
        # 重采样为周K，计算周波动
        self.weekly_df = self.df.resample('W').agg({
            "High": "max",
            "Low": "min"
        })
        self.weekly_df["周波动"] = self.weekly_df["High"] - self.weekly_df["Low"]
        print("✅ 数据获取与计算完成")

    def visualize_both_volatility(self):
        """绘制 日波动 + 周波动 同图"""
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 正常显示中文
        plt.rcParams["axes.unicode_minus"] = False

        fig, ax1 = plt.subplots(figsize=(14, 6))

        # 绘制 日波动
        ax1.plot(self.df.index, self.df["日波动"], 
                 color="#1f77b4", label="日波动", linewidth=1.5)
        ax1.set_ylabel("日波动 (元)", color="#1f77b4", fontsize=12)
        ax1.tick_params(axis="y", labelcolor="#1f77b4")

        # 双轴：绘制 周波动
        ax2 = ax1.twinx()
        ax2.plot(self.weekly_df.index, self.weekly_df["周波动"], 
                 color="#ff4b5c", label="周波动", linewidth=3, marker="o")
        ax2.set_ylabel("周波动 (元)", color="#ff4b5c", fontsize=12)
        ax2.tick_params(axis="y", labelcolor="#ff4b5c")

        # 标题与布局
        plt.title(f"【{self.ticker}】日波动与周波动同图可视化", fontsize=14)
        fig.tight_layout()
        plt.grid(alpha=0.3)
        plt.show()

    def suggest_best_trade_time(self):
        """根据波动大小给出买卖建议"""
        # 计算最近N周平均波动 & 最近N日平均波动
        recent_week_vol = self.weekly_df["周波动"].mean()
        recent_day_vol = self.df["日波动"].mean()

        # 最近一周波动最大的时间点
        max_week_vol_date = self.weekly_df["周波动"].idxmax().date()
        max_week_vol_value = round(self.weekly_df["周波动"].max(), 2)

        print("\n==================== 股票买卖最佳时间建议 ====================")
        print(f"📅 股票代码：{self.ticker}")
        print(f"📈 平均日波动：{round(recent_day_vol, 2)} 元")
        print(f"📅 平均周波动：{round(recent_week_vol, 2)} 元")
        print(f"🔥 波动最大周时间：{max_week_vol_date}，当周波动：{max_week_vol_value} 元")

        # 核心策略：波动大 = 机会多
        print("\n💡 买卖建议（基于波动强度）：")
        if recent_week_vol > recent_day_vol * 3:
            print("→ 周波动极强，适合**波段操作**：周初低位买入，周末高位卖出")
        elif recent_week_vol > recent_day_vol * 1.5:
            print("→ 周波动适中，适合**短线操作**：周内大跌日买入，大涨日卖出")
        else:
            print("→ 波动较小，趋势平稳，建议**中长期持有**，低吸高抛")

        print("================================================================\n")

# ===================== 主程序运行 =====================
if __name__ == "__main__":
    # 股票代码：AAPL苹果 000001.SS平安银行 MSFT微软
    STOCK_CODE = "AAPL"
    
    # 1. 创建股票Skill
    stock_skill = StockVisualSkill(ticker=STOCK_CODE, period="3mo")
    
    # 2. 获取数据
    stock_skill.fetch_data()
    
    # 3. 可视化：日波动 + 周波动同图
    stock_skill.visualize_both_volatility()
    
    # 4. 给出买卖建议
    stock_skill.suggest_best_trade_time()
