from flask import Flask, render_template, request
import os

app = Flask(__name__)

# 10个金融应用主题
FINANCIAL_TOPICS = [
    {"id": 1, "name": "股票分析工具", "description": "股票数据获取、分析和可视化", "icon": "📈"},
    {"id": 2, "name": "量化交易策略", "description": "算法交易和策略回测", "icon": "🤖"},
    {"id": 3, "name": "风险管理系统", "description": "风险评估和投资组合优化", "icon": "🛡️"},
    {"id": 4, "name": "财务报表分析", "description": "财务数据处理和报表解读", "icon": "📊"},
    {"id": 5, "name": "加密货币分析", "description": "区块链数据和加密货币追踪", "icon": "⛓️"},
    {"id": 6, "name": "债券计算工具", "description": "债券定价和收益率计算", "icon": "💵"},
    {"id": 7, "name": "房地产投资分析", "description": "房产估值和投资回报计算", "icon": "🏠"},
    {"id": 8, "name": "金融风险管理", "description": "VaR计算和风险度量", "icon": "📉"},
    {"id": 9, "name": "外汇交易系统", "description": "汇率分析和交易信号", "icon": "💱"},
    {"id": 10, "name": "数据分析可视化", "description": "金融数据可视化和图表", "icon": "🎨"}
]

# 每个主题下的6个分类
TOPIC_CATEGORIES = [
    {"id": 1, "name": "数据获取", "description": "如何从各种API获取金融数据"},
    {"id": 2, "name": "数据处理", "description": "金融数据的清洗和预处理"},
    {"id": 3, "name": "数据分析", "description": "使用统计方法分析金融数据"},
    {"id": 4, "name": "可视化", "description": "金融数据的图表展示"},
    {"id": 5, "name": "机器学习", "description": "AI在金融分析中的应用"},
    {"id": 6, "name": "实战案例", "description": "完整的金融应用项目案例"}
]

# 示例代码数据
EXAMPLE_CODES = [
    # 主题5：加密货币分析
    {
        "topic_id": 5,
        "category_id": 1,
        "title": "加密货币市场数据获取",
        "code": """import requests
import pandas as pd
from datetime import datetime

def get_crypto_prices(symbol, start_date, end_date):
    \"\"\"使用CoinGecko API获取加密货币价格数据\"\"\"
    try:
        # 构建API请求URL
        url = f\"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart/range\"
        params = {
            "vs_currency": "usd",
            "from": int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()),
            "to": int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
        }

        response = requests.get(url, params=params)
        data = response.json()

        # 转换数据格式
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    except Exception as e:
        print(f\"获取{symbol}价格数据失败: {str(e)}\")
        return pd.DataFrame()

def get_crypto_ohlc(symbol, days=30):
    \"\"\"获取加密货币OHLC数据\"\"\"
    try:
        url = f\"https://api.coingecko.com/api/v3/coins/{symbol}/ohlc\"
        params = {
            "vs_currency": "usd",
            "days": days
        }

        response = requests.get(url, params=params)
        data = response.json()

        # 转换数据格式
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    except Exception as e:
        print(f\"获取{symbol}OHLC数据失败: {str(e)}\")
        return pd.DataFrame()

def get_crypto_listings():
    \"\"\"获取加密货币列表\"\"\"
    try:
        url = \"https://api.coingecko.com/api/v3/coins/markets\"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 50,
            "page": 1,
            "sparkline": False,
            "price_change_percentage": "24h"
        }

        response = requests.get(url, params=params)
        data = response.json()

        df = pd.DataFrame(data)
        return df[['id', 'symbol', 'name', 'current_price', 'market_cap',
                   'total_volume', 'price_change_percentage_24h']]

    except Exception as e:
        print(f\"获取加密货币列表失败: {str(e)}\")
        return pd.DataFrame()

# 使用示例
# 获取Bitcoin价格数据（2024年）
bitcoin_prices = get_crypto_prices('bitcoin', '2024-01-01', '2024-04-09')
print(\"Bitcoin价格数据:\\n\", bitcoin_prices.head())

# 获取Bitcoin OHLC数据（最近30天）
bitcoin_ohlc = get_crypto_ohlc('bitcoin', 30)
print(\"\\nBitcoin OHLC数据:\\n\", bitcoin_ohlc.head())

# 获取加密货币列表（Top 10）
crypto_listings = get_crypto_listings()
print(\"\\nTop 10加密货币列表:\\n\", crypto_listings.head(10))
""",
        "explanation": "此代码演示如何使用CoinGecko API获取加密货币的价格数据、OHLC数据和加密货币列表。获取高质量的加密货币数据是分析和策略开发的基础。"
    },
    {
        "topic_id": 5,
        "category_id": 2,
        "title": "加密货币数据预处理",
        "code": """import pandas as pd
import numpy as np

def preprocess_crypto_data(df):
    \"\"\"加密货币数据预处理\"\"\"

    # 检查和处理缺失值
    if df.isnull().any().any():
        print(\"存在缺失值，使用前向填充方法处理\")
        df = df.fillna(method='ffill')

    # 计算收益率
    df['return'] = df['price'].pct_change()

    # 计算对数收益率
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))

    # 计算波动率
    df['volatility'] = df['return'].rolling(window=24).std() * np.sqrt(24)  # 日波动率

    # 计算移动平均线
    df['ma_7'] = df['price'].rolling(window=7).mean()
    df['ma_30'] = df['price'].rolling(window=30).mean()
    df['ma_90'] = df['price'].rolling(window=90).mean()

    # 计算RSI指标
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 计算布林带
    df['bb_mid'] = df['ma_30']
    df['bb_upper'] = df['ma_30'] + 2 * df['price'].rolling(window=30).std()
    df['bb_lower'] = df['ma_30'] - 2 * df['price'].rolling(window=30).std()

    return df

def merge_multiple_crypto_data(data_list, symbols):
    \"\"\"合并多个加密货币的数据\"\"\"

    merged_data = pd.DataFrame()

    for df, symbol in zip(data_list, symbols):
        df_processed = preprocess_crypto_data(df)

        # 重命名列
        renamed_columns = {
            'price': f'{symbol}_price',
            'return': f'{symbol}_return',
            'log_return': f'{symbol}_log_return',
            'volatility': f'{symbol}_volatility',
            'ma_7': f'{symbol}_ma_7',
            'ma_30': f'{symbol}_ma_30',
            'ma_90': f'{symbol}_ma_90',
            'rsi': f'{symbol}_rsi',
            'bb_mid': f'{symbol}_bb_mid',
            'bb_upper': f'{symbol}_bb_upper',
            'bb_lower': f'{symbol}_bb_lower'
        }

        df_renamed = df_processed.rename(columns=renamed_columns)

        if merged_data.empty:
            merged_data = df_renamed
        else:
            merged_data = merged_data.join(df_renamed, how='outer')

    return merged_data

def clean_outliers(df):
    \"\"\"清理异常值\"\"\"

    # 使用3倍标准差方法检测异常值
    for column in df.select_dtypes(include=['float64']).columns:
        mean = df[column].mean()
        std = df[column].std()

        # 定义异常值阈值
        lower_threshold = mean - 3 * std
        upper_threshold = mean + 3 * std

        # 替换异常值
        df[column] = np.where(df[column] < lower_threshold, lower_threshold, df[column])
        df[column] = np.where(df[column] > upper_threshold, upper_threshold, df[column])

    return df

# 使用示例（假设已经获取数据）
# bitcoin_processed = preprocess_crypto_data(bitcoin_prices)
# print(\"预处理后的Bitcoin数据:\\n\", bitcoin_processed.head())
#
# # 合并多个加密货币数据
# ethereum_prices = get_crypto_prices('ethereum', '2024-01-01', '2024-04-09')
# merged_data = merge_multiple_crypto_data([bitcoin_prices, ethereum_prices], ['bitcoin', 'ethereum'])
# print(\"\\n合并后的加密货币数据:\\n\", merged_data.head())
""",
        "explanation": "此代码演示如何对加密货币数据进行预处理，包括数据清洗、缺失值处理、计算技术指标等。加密货币数据预处理是进行分析的必要步骤。"
    },
    {
        "topic_id": 5,
        "category_id": 3,
        "title": "加密货币数据分析",
        "code": """import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_crypto_returns(returns):
    \"\"\"分析加密货币收益率\"\"\"

    returns = returns.dropna()

    summary_stats = {
        "均值": returns.mean(),
        "标准差": returns.std(),
        "偏度": returns.skew(),
        "峰度": returns.kurt(),
        "最小值": returns.min(),
        "最大值": returns.max()
    }

    return pd.Series(summary_stats)

def calculate_var(returns, confidence_level=0.95):
    \"\"\"计算VaR值（风险价值）\"\"\"

    returns = returns.dropna()

    # 使用参数法计算VaR
    mean = returns.mean()
    std = returns.std()
    z_score = norm.ppf(1 - confidence_level)
    var = mean + z_score * std

    # 使用历史法计算VaR
    var_historical = np.percentile(returns, (1 - confidence_level) * 100)

    return var, var_historical

def calculate_cvar(returns, confidence_level=0.95):
    \"\"\"计算CVaR值（条件风险价值）\"\"\"

    returns = returns.dropna()

    # 使用参数法计算CVaR
    mean = returns.mean()
    std = returns.std()
    z_score = norm.ppf(1 - confidence_level)
    cvar = mean + (norm.pdf(z_score) / (1 - confidence_level)) * std

    # 使用历史法计算CVaR
    var_historical = np.percentile(returns, (1 - confidence_level) * 100)
    cvar_historical = returns[returns <= var_historical].mean()

    return cvar, cvar_historical

def analyze_correlation(data):
    \"\"\"分析加密货币之间的相关性\"\"\"

    # 选择价格列
    price_columns = [col for col in data.columns if 'price' in col]
    prices = data[price_columns]

    # 计算相关系数矩阵
    correlation_matrix = prices.corr()

    return correlation_matrix

def plot_correlation_matrix(correlation_matrix):
    \"\"\"绘制相关系数矩阵\"\"\"

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('加密货币价格相关系数矩阵')
    plt.tight_layout()
    plt.savefig('crypto_correlation.png')
    plt.show()

# 使用示例（假设已经有处理过的数据）
# # 计算收益率统计
# bitcoin_returns_stats = analyze_crypto_returns(bitcoin_processed['return'])
# print(\"Bitcoin收益率统计:\\n\", bitcoin_returns_stats)
#
# # 计算VaR和CVaR
# var_param, var_hist = calculate_var(bitcoin_processed['return'])
# cvar_param, cvar_hist = calculate_cvar(bitcoin_processed['return'])
# print(f\"\\nVaR (参数法): {var_param:.4f}\")
# print(f\"VaR (历史法): {var_hist:.4f}\")
# print(f\"CVaR (参数法): {cvar_param:.4f}\")
# print(f\"CVaR (历史法): {cvar_hist:.4f}\")
#
# # 分析相关性
# correlation_matrix = analyze_correlation(merged_data)
# print(\"\\n加密货币价格相关系数矩阵:\\n\", correlation_matrix)
#
# # 绘制相关系数矩阵
# plot_correlation_matrix(correlation_matrix)
""",
        "explanation": "此代码演示如何对加密货币数据进行分析，包括收益率统计、VaR和CVaR计算，以及加密货币之间的相关性分析。这些分析帮助我们理解加密货币市场的风险和特征。"
    },
    {
        "topic_id": 5,
        "category_id": 4,
        "title": "加密货币数据可视化",
        "code": """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_crypto_price(df, title='加密货币价格走势', filename='crypto_price_plot.png'):
    \"\"\"绘制加密货币价格走势\"\"\"

    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['price'])
    plt.title(title)
    plt.xlabel('时间')
    plt.ylabel('价格 (USD)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_crypto_returns(returns, title='加密货币收益率分布', filename='crypto_returns_plot.png'):
    \"\"\"绘制加密货币收益率分布\"\"\"

    plt.figure(figsize=(10, 6))
    sns.histplot(returns.dropna(), kde=True, bins=50)
    plt.title(title)
    plt.xlabel('收益率')
    plt.ylabel('频率')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_rolling_statistics(df, column='price', window=30, filename='rolling_stats_plot.png'):
    \"\"\"绘制滚动统计指标\"\"\"

    plt.figure(figsize=(12, 8))

    # 价格
    plt.subplot(3, 1, 1)
    plt.plot(df['timestamp'], df[column])
    plt.title('价格')
    plt.grid(True)

    # 滚动波动率
    plt.subplot(3, 1, 2)
    plt.plot(df['timestamp'], df['volatility'])
    plt.title(f'{window}日滚动波动率')
    plt.grid(True)

    # RSI
    plt.subplot(3, 1, 3)
    plt.plot(df['timestamp'], df['rsi'])
    plt.axhline(y=30, color='g', linestyle='--', label='超卖')
    plt.axhline(y=70, color='r', linestyle='--', label='超买')
    plt.title('RSI指标')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_candlestick_chart(ohlc_data, filename='candlestick_plot.png'):
    \"\"\"绘制蜡烛图\"\"\"

    # 创建蜡烛图
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制蜡烛图主体
    colors = ['g' if close >= open else 'r' for close, open in zip(ohlc_data['close'], ohlc_data['open'])]
    ax.bar(ohlc_data.index, ohlc_data['close'] - ohlc_data['open'], width=0.6, bottom=ohlc_data['open'], color=colors)

    # 绘制影线
    ax.vlines(ohlc_data.index, ohlc_data['low'], ohlc_data['high'], color=colors)

    plt.title('加密货币蜡烛图')
    plt.xlabel('时间')
    plt.ylabel('价格 (USD)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def plot_multiple_cryptos(data, symbols, filename='multiple_cryptos_plot.png'):
    \"\"\"绘制多个加密货币的价格走势\"\"

    plt.figure(figsize=(12, 6))

    for symbol in symbols:
        plt.plot(data['timestamp'], data[f'{symbol}_price'], label=symbol)

    plt.title('加密货币价格比较')
    plt.xlabel('时间')
    plt.ylabel('价格 (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# 使用示例（假设已经获取数据）
# # 绘制Bitcoin价格走势
# plot_crypto_price(bitcoin_prices, 'Bitcoin价格走势', 'bitcoin_price_plot.png')
#
# # 绘制收益率分布
# plot_crypto_returns(bitcoin_processed['return'], 'Bitcoin收益率分布', 'bitcoin_returns_plot.png')
#
# # 绘制滚动统计指标
# plot_rolling_statistics(bitcoin_processed, filename='bitcoin_rolling_stats.png')
#
# # 绘制蜡烛图
# plot_candlestick_chart(bitcoin_ohlc, 'bitcoin_candlestick_plot.png')
""",
        "explanation": "此代码演示如何对加密货币数据进行可视化，包括价格走势、收益率分布、滚动统计指标、蜡烛图和多个加密货币的价格比较。可视化帮助我们更好地理解加密货币市场的趋势和模式。"
    },
    {
        "topic_id": 5,
        "category_id": 5,
        "title": "加密货币机器学习",
        "code": """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def create_features(df):
    \"\"\"创建特征\"\"\"

    features = df[['volatility', 'ma_7', 'ma_30', 'ma_90', 'rsi', 'bb_mid', 'bb_upper', 'bb_lower']]

    return features

def create_target(df, days_ahead=1):
    \"\"\"创建目标变量（未来价格变化）\"\"

    target = df['price'].shift(-days_ahead) - df['price']
    return target

def prepare_data(df, days_ahead=1):
    \"\"\"准备数据\"\"

    features = create_features(df)
    target = create_target(df, days_ahead)

    # 删除包含NaN的行
    full_data = pd.concat([features, target], axis=1)
    full_data = full_data.dropna()

    X = full_data.iloc[:, :-1]
    y = full_data.iloc[:, -1]

    return X, y

def train_model(X_train, y_train, model_type='random_forest'):
    \"\"\"训练模型\"\"\"

    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'linear_regression':
        model = LinearRegression()
    else:
        raise ValueError(f\"不支持的模型类型: {model_type}\")

    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test):
    \"\"\"评估模型\"\"

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return mse, rmse, r2

def plot_predictions(y_test, y_pred, filename='predictions_plot.png'):
    \"\"\"绘制预测结果\"\"

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title('模型预测 vs 实际值')
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# 使用示例（假设已经有处理过的数据）
# # 准备数据
# X, y = prepare_data(bitcoin_processed)
#
# # 划分训练和测试数据
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 训练随机森林模型
# rf_model = train_model(X_train, y_train, 'random_forest')
#
# # 训练线性回归模型
# lr_model = train_model(X_train, y_train, 'linear_regression')
#
# # 评估模型
# rf_mse, rf_rmse, rf_r2 = evaluate_model(rf_model, X_test, y_test)
# lr_mse, lr_rmse, lr_r2 = evaluate_model(lr_model, X_test, y_test)
#
# print(f\"随机森林模型 - MSE: {rf_mse:.4f}, RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}\")
# print(f\"线性回归模型 - MSE: {lr_mse:.4f}, RMSE: {lr_rmse:.4f}, R²: {lr_r2:.4f}\")
#
# # 绘制随机森林模型的预测结果
# rf_pred = rf_model.predict(X_test)
# plot_predictions(y_test, rf_pred, 'rf_predictions_plot.png')
""",
        "explanation": "此代码演示如何使用机器学习方法预测加密货币价格变化，包括特征工程、模型训练和评估。机器学习可以帮助我们识别加密货币市场的模式和趋势。"
    },
    {
        "topic_id": 5,
        "category_id": 6,
        "title": "加密货币交易策略",
        "code": """import pandas as pd
import numpy as np

class TradingStrategy:
    def __init__(self, data, initial_capital=10000):
        self.data = data
        self.initial_capital = initial_capital
        self.positions = pd.Series(index=data.index, dtype=int)
        self.portfolio = pd.DataFrame(index=data.index)

    def generate_signals(self):
        \"\"\"生成交易信号\"\"\"
        raise NotImplementedError("子类必须实现generate_signals方法")

    def backtest(self):
        \"\"\"回测策略\"\"\"

        # 计算价格
        prices = self.data['price']

        # 计算每日收益
        self.portfolio['Price'] = prices

        # 初始化投资组合价值
        self.portfolio['Cash'] = self.initial_capital
        self.portfolio['Holdings'] = 0.0
        self.portfolio['Total'] = self.initial_capital

        for i in range(len(prices)):
            date = prices.index[i]

            # 计算持有的货币数量
            if self.positions[date] == 1:
                # 买入
                shares_to_buy = int(self.portfolio['Cash'][date] / prices[date])
                cost = shares_to_buy * prices[date]
                self.portfolio['Holdings'][date] = shares_to_buy
                self.portfolio['Cash'][date] -= cost
            elif self.positions[date] == -1:
                # 卖出
                shares_to_sell = int(self.portfolio['Holdings'][date])
                revenue = shares_to_sell * prices[date]
                self.portfolio['Cash'][date] += revenue
                self.portfolio['Holdings'][date] = 0

            # 计算投资组合总价值
            self.portfolio['Total'][date] = self.portfolio['Cash'][date] + self.portfolio['Holdings'][date] * prices[date]

        # 计算收益率
        self.portfolio['Return'] = self.portfolio['Total'].pct_change()

        return self.portfolio

    def calculate_performance_metrics(self):
        \"\"\"计算策略绩效指标\"\"

        # 计算总回报
        total_return = (self.portfolio['Total'][-1] - self.initial_capital) / self.initial_capital

        # 计算年化收益率
        num_years = len(self.portfolio) / 365
        annual_return = (1 + total_return) ** (1 / num_years) - 1

        # 计算最大回撤
        cumulative_returns = (1 + self.portfolio['Return']).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        # 计算夏普比率
        sharpe_ratio = self.portfolio['Return'].mean() / self.portfolio['Return'].std() * np.sqrt(365)

        performance_metrics = {
            '总回报': total_return,
            '年化收益率': annual_return,
            '最大回撤': max_drawdown,
            '夏普比率': sharpe_ratio
        }

        return performance_metrics

class MovingAverageStrategy(TradingStrategy):
    def generate_signals(self):
        \"\"\"移动平均交叉策略\"\"\"

        signals = pd.Series(0, index=self.data.index)

        # 当短期均线突破长期均线时买入
        signals[self.data['ma_7'] > self.data['ma_30']] = 1

        # 当短期均线跌破长期均线时卖出
        signals[self.data['ma_7'] < self.data['ma_30']] = -1

        self.positions = signals

        return signals

class RSIStrategy(TradingStrategy):
    def generate_signals(self):
        \"\"\"RSI策略\"\"\"

        signals = pd.Series(0, index=self.data.index)

        # 当RSI低于30时买入
        signals[self.data['rsi'] < 30] = 1

        # 当RSI高于70时卖出
        signals[self.data['rsi'] > 70] = -1

        self.positions = signals

        return signals

# 使用示例（假设已经有处理过的数据）
# # 使用移动平均策略
# strategy = MovingAverageStrategy(bitcoin_processed)
# strategy.generate_signals()
# portfolio = strategy.backtest()
#
# # 计算绩效指标
# performance_metrics = strategy.calculate_performance_metrics()
# print(\"移动平均策略绩效指标:\\n\", performance_metrics)
#
# # 使用RSI策略
# strategy = RSIStrategy(bitcoin_processed)
# strategy.generate_signals()
# portfolio = strategy.backtest()
#
# # 计算绩效指标
# performance_metrics = strategy.calculate_performance_metrics()
# print(\"\\nRSI策略绩效指标:\\n\", performance_metrics)
""",
        "explanation": "此代码演示如何实现加密货币交易策略，包括移动平均交叉策略和RSI策略。回测功能可以帮助我们评估策略的历史表现。"
    },
    # 主题4：财务报表分析
    {
        "topic_id": 4,
        "category_id": 1,
        "title": "财务报表数据获取",
        "code": """import pandas as pd
import numpy as np
import requests
from io import StringIO
import yfinance as yf

def get_income_statement(ticker):
    \"\"\"获取公司利润表数据\"\"\"
    try:
        stock = yf.Ticker(ticker)
        income_stmt = stock.income_stmt
        return income_stmt
    except Exception as e:
        print(f"Error fetching income statement for {ticker}: {e}")
        return pd.DataFrame()

def get_balance_sheet(ticker):
    \"\"\"获取公司资产负债表数据\"\"\"
    try:
        stock = yf.Ticker(ticker)
        balance_sheet = stock.balance_sheet
        return balance_sheet
    except Exception as e:
        print(f"Error fetching balance sheet for {ticker}: {e}")
        return pd.DataFrame()

def get_cash_flow(ticker):
    \"\"\"获取公司现金流量表数据\"\"\"
    try:
        stock = yf.Ticker(ticker)
        cash_flow = stock.cashflow
        return cash_flow
    except Exception as e:
        print(f"Error fetching cash flow for {ticker}: {e}")
        return pd.DataFrame()

def get_financial_ratios(ticker):
    \"\"\"获取公司财务比率数据\"\"\"
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info
    except Exception as e:
        print(f"Error fetching financial ratios for {ticker}: {e}")
        return {}

# 使用示例
ticker = "AAPL"

# 获取三大财务报表
income_statement = get_income_statement(ticker)
balance_sheet = get_balance_sheet(ticker)
cash_flow = get_cash_flow(ticker)

# 获取财务比率
financial_ratios = get_financial_ratios(ticker)

# 显示数据信息
print(f"\\n利润表形状: {income_statement.shape}")
print(f"资产负债表形状: {balance_sheet.shape}")
print(f"现金流量表形状: {cash_flow.shape}")
print(f"财务比率数量: {len(financial_ratios)}")
""",
        "explanation": "此代码演示如何使用yfinance库获取公司的三大财务报表（利润表、资产负债表、现金流量表）和财务比率数据。财务报表分析的基础是准确的数据获取。"
    },
    {
        "topic_id": 4,
        "category_id": 2,
        "title": "财务报表数据预处理",
        "code": """import pandas as pd
import numpy as np

def preprocess_financial_data(income_statement, balance_sheet, cash_flow):
    \"\"\"预处理财务报表数据\"\"\"
    # 统一数据格式
    for df in [income_statement, balance_sheet, cash_flow]:
        if not df.empty:
            df.index = df.index.map(lambda x: x.lower().replace(" ", "_"))
            df.columns = df.columns.map(lambda x: x.year)

    return income_statement, balance_sheet, cash_flow

def remove_outliers(data, threshold=3):
    \"\"\"移除异常值\"\"\"
    if data.empty:
        return data

    z_scores = np.abs((data - data.mean()) / data.std())
    return data[(z_scores < threshold).all(axis=1)]

def standardize_data(data):
    \"\"\"标准化财务数据\"\"\"
    if data.empty:
        return data

    # 对每一列进行标准化
    result = data.copy()
    for column in data.columns:
        if data[column].dtype in ['float64', 'int64']:
            mean = data[column].mean()
            std = data[column].std()
            if std > 0:
                result[column] = (data[column] - mean) / std

    return result

def calculate_growth_rates(data):
    \"\"\"计算增长率\"\"\"
    if data.empty:
        return data

    # 对每一行（指标）计算年度增长率
    growth_data = data.copy()
    for index in data.index:
        for i in range(1, len(data.columns)):
            if data.at[index, data.columns[i]] and data.at[index, data.columns[i-1]]:
                growth_rate = ((data.at[index, data.columns[i]] - data.at[index, data.columns[i-1]]) /
                              abs(data.at[index, data.columns[i-1]])) * 100
                growth_data.at[index, data.columns[i]] = growth_rate

    return growth_data

# 使用示例（假设之前已获取数据）
# income_statement, balance_sheet, cash_flow = preprocess_financial_data(income_statement, balance_sheet, cash_flow)
# income_statement_no_outliers = remove_outliers(income_statement)
# standard_income = standardize_data(income_statement_no_outliers)
# income_growth = calculate_growth_rates(income_statement_no_outliers)
""",
        "explanation": "此代码演示如何对财务报表数据进行预处理，包括数据格式化、异常值处理、标准化和增长率计算。数据预处理是财务报表分析的关键步骤。"
    },
    {
        "topic_id": 4,
        "category_id": 3,
        "title": "财务指标计算",
        "code": """import pandas as pd
import numpy as np

class FinancialRatioCalculator:
    def __init__(self, income_statement, balance_sheet, cash_flow):
        self.income = income_statement
        self.balance = balance_sheet
        self.cash_flow = cash_flow

    def calculate_profitability_ratios(self, year):
        \"\"\"计算盈利能力比率\"\"\"
        ratios = {}

        if not self.income.empty and year in self.income.columns:
            # 毛利率
            if "gross_profit" in self.income.index and "total_revenue" in self.income.index:
                ratios["gross_margin"] = (self.income.at["gross_profit", year] /
                                       self.income.at["total_revenue", year]) * 100

            # 净利润率
            if "net_income" in self.income.index and "total_revenue" in self.income.index:
                ratios["net_profit_margin"] = (self.income.at["net_income", year] /
                                             self.income.at["total_revenue", year]) * 100

            # 总资产收益率
            if "net_income" in self.income.index and "total_assets" in self.balance.index:
                avg_assets = (self.balance.at["total_assets", year] +
                            self.balance.at["total_assets", year - 1]) / 2
                ratios["roa"] = (self.income.at["net_income", year] / avg_assets) * 100

            # 股东权益收益率
            if "net_income" in self.income.index and "total_stockholder_equity" in self.balance.index:
                avg_equity = (self.balance.at["total_stockholder_equity", year] +
                            self.balance.at["total_stockholder_equity", year - 1]) / 2
                ratios["roe"] = (self.income.at["net_income", year] / avg_equity) * 100

        return ratios

    def calculate_liquidity_ratios(self, year):
        \"\"\"计算流动性比率\"\"\"
        ratios = {}

        if not self.balance.empty and year in self.balance.columns:
            # 流动比率
            if "total_current_assets" in self.balance.index and "total_current_liabilities" in self.balance.index:
                ratios["current_ratio"] = (self.balance.at["total_current_assets", year] /
                                          self.balance.at["total_current_liabilities", year])

            # 速动比率
            if ("total_current_assets" in self.balance.index and
                "inventory" in self.balance.index and
                "total_current_liabilities" in self.balance.index):
                ratios["quick_ratio"] = ((self.balance.at["total_current_assets", year] -
                                       self.balance.at["inventory", year]) /
                                       self.balance.at["total_current_liabilities", year])

        return ratios

    def calculate_solvency_ratios(self, year):
        \"\"\"计算偿付能力比率\"\"\"
        ratios = {}

        if (not self.income.empty and year in self.income.columns and
            not self.balance.empty and year in self.balance.columns):
            # 负债权益比
            if ("total_liabilities" in self.balance.index and
                "total_stockholder_equity" in self.balance.index):
                ratios["debt_to_equity"] = (self.balance.at["total_liabilities", year] /
                                           self.balance.at["total_stockholder_equity", year])

            # 利息保障倍数
            if ("ebit" in self.income.index and "interest_expense" in self.income.index):
                ratios["interest_coverage"] = (self.income.at["ebit", year] /
                                             self.income.at["interest_expense", year])

        return ratios

    def calculate_all_ratios(self, year):
        \"\"\"计算所有财务比率\"\"\"
        all_ratios = {}
        all_ratios.update(self.calculate_profitability_ratios(year))
        all_ratios.update(self.calculate_liquidity_ratios(year))
        all_ratios.update(self.calculate_solvency_ratios(year))
        return all_ratios

# 使用示例
# calculator = FinancialRatioCalculator(income_statement, balance_sheet, cash_flow)
# ratios = calculator.calculate_all_ratios(2024)
# print(ratios)
""",
        "explanation": "此代码演示如何计算各种财务比率，包括盈利能力、流动性和偿付能力比率。这些比率是财务报表分析的核心指标，帮助评估公司的财务健康状况。"
    },
    {
        "topic_id": 4,
        "category_id": 4,
        "title": "财务报表可视化",
        "code": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def plot_income_statement_trends(income_statement, key_metrics):
    \"\"\"绘制利润表趋势\"\"\"
    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(key_metrics, 1):
        if metric in income_statement.index:
            plt.subplot(2, 2, i)
            plt.plot(income_statement.columns, income_statement.loc[metric], marker="o")
            plt.title(f"{metric.replace('_', ' ').title()}")
            plt.xlabel("年份")
            plt.ylabel("金额")
            plt.grid(True)

    plt.tight_layout()
    plt.savefig("income_statement_trends.png")
    plt.show()

def plot_balance_sheet_composition(balance_sheet, year):
    \"\"\"绘制资产负债表结构\"\"\"
    if year in balance_sheet.columns:
        assets = [item for item in balance_sheet.index if "asset" in item and "total" not in item]
        liabilities = [item for item in balance_sheet.index if "liability" in item and "total" not in item]
        equity = [item for item in balance_sheet.index if "equity" in item]

        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

        # 饼图展示资产结构
        asset_values = [balance_sheet.at[item, year] for item in assets]
        ax1.pie(asset_values, labels=assets, autopct='%1.1f%%', startangle=90)
        ax1.set_title("资产结构")

        # 饼图展示负债和股东权益结构
        liability_equity_values = []
        liability_equity_labels = []

        if liabilities:
            liability_equity_values.append(balance_sheet.at["total_current_liabilities", year])
            liability_equity_labels.append("流动负债")
            liability_equity_values.append(balance_sheet.at["long_term_debt", year])
            liability_equity_labels.append("长期负债")

        if equity:
            liability_equity_values.append(balance_sheet.at["total_stockholder_equity", year])
            liability_equity_labels.append("股东权益")

        ax2.pie(liability_equity_values, labels=liability_equity_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title("负债和股东权益结构")

        plt.savefig("balance_sheet_composition.png")
        plt.show()

def plot_financial_ratios_comparison(ratios_data, ratio_types):
    \"\"\"绘制财务比率对比\"\"\"
    years = list(ratios_data.keys())
    all_ratios = {}

    # 收集所有要比较的比率
    for year, ratios in ratios_data.items():
        for ratio_type in ratio_types:
            if ratio_type not in all_ratios:
                all_ratios[ratio_type] = []
            all_ratios[ratio_type].append(ratios.get(ratio_type, 0))

    # 绘制对比图
    fig, axes = plt.subplots(len(all_ratios), 1, figsize=(12, 20))

    for i, (ratio, values) in enumerate(all_ratios.items()):
        axes[i].bar(years, values)
        axes[i].set_title(ratio.replace('_', ' ').title())
        axes[i].set_xlabel("年份")
        axes[i].set_ylabel("值")

        # 在每个条形上添加数值标签
        for j, value in enumerate(values):
            axes[i].text(j, value + max(values)*0.05, f"{value:.2f}",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("financial_ratios_comparison.png")
    plt.show()

# 使用示例（假设之前已计算数据）
# key_metrics = ["gross_profit", "operating_income", "net_income"]
# plot_income_statement_trends(income_statement, key_metrics)
# plot_balance_sheet_composition(balance_sheet, 2024)
# ratios_2024 = calculator.calculate_all_ratios(2024)
# plot_financial_ratios_comparison({"2024": ratios_2024, "2023": ratios_2023}, ["gross_margin", "current_ratio"])
""",
        "explanation": "此代码演示如何对财务报表数据进行可视化分析，包括趋势分析、结构分析和对比分析。可视化帮助我们更直观地理解财务数据的模式和关系。"
    },
    {
        "topic_id": 4,
        "category_id": 5,
        "title": "财务报表机器学习分析",
        "code": """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

class FinancialStatementAnalyst:
    def __init__(self, financial_data):
        self.data = financial_data
        self.model = None
        self.scaler = StandardScaler()

    def prepare_features(self, features):
        \"\"\"准备特征数据\"\"\"
        X = pd.DataFrame()

        for feature in features:
            # 计算财务比率
            if feature == "profitability":
                # 盈利能力指标
                X["gross_margin"] = (self.data["gross_profit"] / self.data["total_revenue"]) * 100
                X["net_margin"] = (self.data["net_income"] / self.data["total_revenue"]) * 100

            elif feature == "liquidity":
                # 流动性指标
                X["current_ratio"] = (self.data["total_current_assets"] /
                                     self.data["total_current_liabilities"])
                X["quick_ratio"] = ((self.data["total_current_assets"] -
                                    self.data["inventory"]) /
                                    self.data["total_current_liabilities"])

            elif feature == "solvency":
                # 偿付能力指标
                X["debt_to_equity"] = (self.data["total_liabilities"] /
                                      self.data["total_stockholder_equity"])

        # 计算趋势特征
        for column in X.columns:
            X[f"{column}_trend"] = X[column].rolling(window=3).mean().shift(1)

        return X.dropna()

    def prepare_labels(self, target_column="financial_health"):
        \"\"\"准备标签数据\"\"\"
        # 简单的健康标签（需要根据实际需求调整）
        labels = []

        for index, row in self.data.iterrows():
            # 基于财务比率阈值判断健康状况
            if (row["net_margin"] > 10 and
                row["current_ratio"] > 1.5 and
                row["debt_to_equity"] < 2):
                labels.append("健康")
            elif (row["net_margin"] > 0 and
                  row["current_ratio"] > 1 and
                  row["debt_to_equity"] < 3):
                labels.append("正常")
            else:
                labels.append("需要关注")

        return pd.Series(labels, index=self.data.index)

    def train_classifier(self, features, test_size=0.2):
        \"\"\"训练分类器\"\"\"
        X = self.prepare_features(features)
        y = self.prepare_labels()[X.index]

        # 划分训练和测试数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 训练随机森林分类器
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # 预测和评估
        y_pred = self.model.predict(X_test_scaled)

        # 打印评估报告
        print("分类报告:")
        print(classification_report(y_test, y_pred))
        print("\\n混淆矩阵:")
        print(confusion_matrix(y_test, y_pred))

        return self.model

    def predict_financial_health(self, new_data):
        \"\"\"预测财务健康状况\"\"\"
        if self.model:
            # 准备新数据的特征
            features = ["profitability", "liquidity", "solvency"]
            X_new = self.prepare_features(features)

            if not X_new.empty:
                X_scaled = self.scaler.transform(X_new)
                predictions = self.model.predict(X_scaled)
                return pd.Series(predictions, index=X_new.index)

        return None

# 使用示例（需要准备适当的数据集）
# df = pd.read_csv("financial_data.csv", index_col="Year")
# analyst = FinancialStatementAnalyst(df)
# model = analyst.train_classifier(["profitability", "liquidity", "solvency"])
# predictions = analyst.predict_financial_health(df.tail(1))
""",
        "explanation": "此代码演示如何使用机器学习方法（随机森林分类器）分析财务报表数据，包括特征工程、模型训练、预测和评估。机器学习可以帮助我们识别复杂的财务模式。"
    },
    {
        "topic_id": 4,
        "category_id": 6,
        "title": "完整财务报表分析报告",
        "code": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO

class FinancialReportGenerator:
    def __init__(self, company_name, financial_data):
        self.company_name = company_name
        self.financial_data = financial_data

    def generate_textual_report(self):
        \"\"\"生成文本报告\"\"\"
        report = StringIO()

        report.write(f"# 财务报表分析报告\\n")
        report.write(f"## {self.company_name}\\n")
        report.write(f"### 报告时间: {pd.Timestamp.now().strftime('%Y年%m月%d日')}\\n\\n")

        # 公司概览
        report.write("## 公司概览\\n")
        report.write("- 行业: 科技制造业\\n")
        report.write("- 主营业务: 电子产品和软件服务\\n")
        report.write("- 上市时间: 1980年\\n\\n")

        # 财务健康状况评估
        report.write("## 财务健康状况评估\\n")

        for year in self.financial_data["income"].columns[-3:]:
            report.write(f"### {year}年财务比率分析\\n")

            # 获取该年份的财务比率（假设之前已计算）
            # ratios = calculator.calculate_all_ratios(year)
            # for category, values in ratios.items():
            #     report.write(f"#### {category}\\n")
            #     for ratio, value in values.items():
            #         report.write(f"- {ratio}: {value:.2f}\\n")

            report.write("\\n")

        # 趋势分析
        report.write("## 财务趋势分析\\n")
        report.write("- 营业收入增长率: 过去三年平均增长15.2%\\n")
        report.write("- 净利润增长率: 过去三年平均增长20.5%\\n")
        report.write("- 资产负债率: 保持在30%左右，财务结构稳健\\n")
        report.write("- 净利率: 持续上升，从2022年的21.2%上升到2024年的25.8%\\n\\n")

        # 风险评估
        report.write("## 风险评估\\n")
        report.write("### 主要风险: 国际市场风险\\n")
        report.write("- 海外市场占比高，受汇率波动影响\\n")
        report.write("- 中美贸易摩擦可能影响供应链成本\\n")
        report.write("\\n")

        report.write("### 风险控制建议:\\n")
        report.write("- 加强外汇风险管理\\n")
        report.write("- 多元化供应链来源\\n")
        report.write("- 加强研发投入，保持产品竞争力\\n")

        return report.getvalue()

    def generate_visual_report(self):
        \"\"\"生成可视化报告\"\"\"
        # 创建报告标题页
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.8, f"{self.company_name}\\n财务报表分析报告", fontsize=20, ha="center")
        plt.text(0.5, 0.5, "报告时间: " + pd.Timestamp.now().strftime('%Y年%m月%d日'), fontsize=12, ha="center")
        plt.axis('off')
        plt.savefig("report_title.png", bbox_inches="tight")
        plt.close()

        # 其他图表（假设之前已生成）
        # income_statement_trends.png
        # balance_sheet_composition.png
        # financial_ratios_comparison.png
        return ["report_title.png", "income_statement_trends.png",
                "balance_sheet_composition.png", "financial_ratios_comparison.png"]

    def save_report(self, report_text, visual_report):
        \"\"\"保存报告\"\"\"
        with open("financial_report.md", "w", encoding="utf-8") as f:
            f.write(report_text)

        print("财务报表分析报告已生成")
        print(f"文本报告: financial_report.md")
        print(f"图表文件: {len(visual_report)}个")

# 使用示例（需要准备完整的财务数据）
# income_statement, balance_sheet, cash_flow = get_financial_data()
# data_dict = {"income": income_statement, "balance": balance_sheet, "cash_flow": cash_flow}
# report_generator = FinancialReportGenerator("苹果公司", data_dict)
# text_report = report_generator.generate_textual_report()
# visual_report = report_generator.generate_visual_report()
# report_generator.save_report(text_report, visual_report)
""",
        "explanation": "此代码演示如何生成完整的财务报表分析报告，包括文本报告和可视化图表。报告应综合财务比率分析、趋势分析和风险评估，为决策者提供全面的财务状况概览。"
    },
    # 主题3：风险管理系统
    {
        "topic_id": 3,
        "category_id": 1,
        "title": "风险数据获取",
        "code": """import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

def fetch_stock_data(symbols, start_date, end_date):
    \"\"\"获取股票数据\"\"\"
    data = {}
    for symbol in symbols:
        try:
            df = yf.download(symbol, start=start_date, end=end_date)
            data[symbol] = df
            print(f"成功获取{symbol}数据，共{len(df)}条")
        except Exception as e:
            print(f"获取{symbol}数据失败: {e}")
    return data

def fetch_index_data(symbol, start_date, end_date):
    \"\"\"获取指数数据\"\"\"
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        print(f"成功获取{symbol}指数数据，共{len(df)}条")
        return df
    except Exception as e:
        print(f"获取指数数据失败: {e}")
        return pd.DataFrame()

def fetch_macro_data():
    \"\"\"获取宏观经济数据（示例）\"\"\"
    dates = pd.date_range(start="2015-01-01", end=datetime.now(), freq="D")
    inflation = np.random.normal(0.02/365, 0.01/365, len(dates))
    interest_rate = np.random.normal(0.03/365, 0.005/365, len(dates))

    data = pd.DataFrame({
        "Date": dates,
        "Inflation": inflation,
        "InterestRate": interest_rate
    }).set_index("Date")

    print(f"生成宏观经济数据，共{len(data)}条")
    return data

# 使用示例
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
start_date = "2015-01-01"
end_date = "2025-01-01"

stock_data = fetch_stock_data(symbols, start_date, end_date)
index_data = fetch_index_data("^GSPC", start_date, end_date)
macro_data = fetch_macro_data()
""",
        "explanation": "此代码演示如何获取风险管理所需的各类数据，包括股票数据、指数数据和宏观经济数据。风险管理的基础是准确、全面的数据获取。"
    },
    {
        "topic_id": 3,
        "category_id": 2,
        "title": "风险数据预处理",
        "code": """import pandas as pd
import numpy as np

def preprocess_stock_data(stock_data):
    \"\"\"预处理股票数据\"\"\"
    processed_data = {}

    for symbol, df in stock_data.items():
        # 检查并填充缺失值
        df = df.fillna(method="ffill")

        # 计算收益率
        df["Return"] = df["Close"].pct_change()

        # 计算波动率
        df["Volatility"] = df["Return"].rolling(window=30).std() * np.sqrt(252)

        # 计算对数收益率
        df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

        processed_data[symbol] = df
        print(f"{symbol}数据预处理完成")

    return processed_data

def preprocess_index_data(index_data):
    \"\"\"预处理指数数据\"\"\"
    # 检查并填充缺失值
    index_data = index_data.fillna(method="ffill")

    # 计算收益率和波动率
    index_data["Return"] = index_data["Close"].pct_change()
    index_data["Volatility"] = index_data["Return"].rolling(window=30).std() * np.sqrt(252)

    print("指数数据预处理完成")
    return index_data

def merge_data(processed_stock, processed_index, macro_data):
    \"\"\"合并各类数据\"\"\"
    # 合并股票数据
    merged_data = pd.DataFrame()
    for symbol, df in processed_stock.items():
        temp = df[["Close", "Return", "Volatility"]].copy()
        temp.columns = [f"{symbol}_{col}" for col in temp.columns]
        if merged_data.empty:
            merged_data = temp
        else:
            merged_data = merged_data.join(temp, how="outer")

    # 合并指数数据
    merged_data = merged_data.join(
        processed_index[["Return", "Volatility"]].rename(
            columns={"Return": "SP500_Return", "Volatility": "SP500_Volatility"}
        ),
        how="outer"
    )

    # 合并宏观经济数据
    merged_data = merged_data.join(macro_data, how="outer")

    # 填充最终缺失值
    merged_data = merged_data.fillna(method="ffill").fillna(method="bfill")

    print(f"数据合并完成，最终形状: {merged_data.shape}")
    return merged_data

# 使用示例
# processed_stock = preprocess_stock_data(stock_data)
# processed_index = preprocess_index_data(index_data)
# final_data = merge_data(processed_stock, processed_index, macro_data)
""",
        "explanation": "此代码演示如何预处理风险管理数据，包括股票和指数数据的清洗、收益率计算、波动率计算，以及各类数据的合并。数据预处理是风险管理的关键步骤。"
    },
    {
        "topic_id": 3,
        "category_id": 3,
        "title": "风险分析方法",
        "code": """import pandas as pd
import numpy as np
from scipy.stats import norm, t

class RiskAnalyst:
    def __init__(self, data):
        self.data = data

    def calculate_var(self, returns, method="parametric", confidence=0.95, period=1):
        \"\"\"计算风险价值(VaR)\"\"\"
        if method == "parametric":
            mu = returns.mean()
            sigma = returns.std()
            VaR = mu * period - sigma * norm.ppf(confidence) * np.sqrt(period)
        elif method == "historical":
            VaR = -returns.quantile(1 - confidence)
        elif method == "monte_carlo":
            np.random.seed(42)
            sim_returns = np.random.normal(returns.mean(), returns.std(), 10000)
            VaR = -np.percentile(sim_returns, 100 * (1 - confidence))
        else:
            raise ValueError("无效的VaR计算方法")

        return VaR

    def calculate_cvar(self, returns, method="parametric", confidence=0.95):
        \"\"\"计算条件风险价值(CVaR)\"\"\"
        if method == "parametric":
            mu = returns.mean()
            sigma = returns.std()
            cvar = mu - sigma * norm.pdf(norm.ppf(1 - confidence)) / (1 - confidence)
        elif method == "historical":
            var = -returns.quantile(1 - confidence)
            cvar = -returns[returns <= -var].mean()
        else:
            raise ValueError("无效的CVaR计算方法")

        return cvar

    def analyze_portfolio_risk(self, weights, symbols, period=1):
        \"\"\"分析投资组合风险\"\"\"
        returns = np.array([self.data[f"{symbol}_Return"] for symbol in symbols]).T
        cov_matrix = np.cov(returns.T)

        # 计算投资组合收益率和波动率
        portfolio_return = np.dot(weights, np.array([self.data[f"{symbol}_Return"].mean() for symbol in symbols]))
        portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights))) * np.sqrt(period)

        # 计算VaR和CVaR
        portfolio_returns = np.dot(returns, weights)
        portfolio_VaR = self.calculate_var(portfolio_returns, method="parametric")
        portfolio_CVaR = self.calculate_cvar(portfolio_returns, method="parametric")

        risk_report = {
            "Return": portfolio_return,
            "Volatility": portfolio_volatility,
            "VaR": portfolio_VaR,
            "CVaR": portfolio_CVaR
        }

        return risk_report

# 使用示例
# analyst = RiskAnalyst(final_data)
# weights = [0.25, 0.25, 0.25, 0.25]
# risk_report = analyst.analyze_portfolio_risk(weights, symbols)
# print("投资组合风险分析报告:")
# for metric, value in risk_report.items():
#     print(f"{metric}: {value:.4f}")
""",
        "explanation": "此代码演示如何使用多种方法计算风险管理中的关键指标，包括VaR（风险价值）和CVaR（条件风险价值），并提供投资组合风险分析功能。"
    },
    {
        "topic_id": 3,
        "category_id": 4,
        "title": "风险可视化",
        "code": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_volatility_evolution(processed_data, symbols):
    \"\"\"绘制波动率演变\"\"\"
    plt.figure(figsize=(12, 6))
    for symbol in symbols:
        plt.plot(processed_data[symbol].index, processed_data[symbol]["Volatility"], label=f"{symbol}")

    plt.title("波动率演变")
    plt.xlabel("日期")
    plt.ylabel("年波动率")
    plt.legend()
    plt.grid(True)
    plt.savefig("volatility_evolution.png")
    plt.show()

def plot_correlation_matrix(final_data, symbols):
    \"\"\"绘制相关系数矩阵\"\"\"
    returns = [final_data[f"{symbol}_Return"] for symbol in symbols]
    returns.append(final_data["SP500_Return"])
    return_df = pd.DataFrame(returns).T
    return_df.columns = symbols + ["SP500"]

    corr_matrix = return_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title("收益率相关系数矩阵")
    plt.savefig("correlation_matrix.png")
    plt.show()

def plot_return_distribution(final_data, symbols):
    \"\"\"绘制收益率分布\"\"\"
    plt.figure(figsize=(12, 8))
    for i, symbol in enumerate(symbols, 1):
        plt.subplot(2, 2, i)
        sns.histplot(final_data[f"{symbol}_Return"].dropna(), kde=True)
        plt.title(f"{symbol}收益率分布")
        plt.xlabel("收益率")
        plt.ylabel("频率")

    plt.tight_layout()
    plt.savefig("return_distribution.png")
    plt.show()

def plot_risk_metrics_comparison(risk_reports):
    \"\"\"绘制风险指标比较\"\"\"
    methods = list(risk_reports.keys())
    VaRs = [report["VaR"] for report in risk_reports.values()]
    CVaRs = [report["CVaR"] for report in risk_reports.values()]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, VaRs, width, label="VaR")
    plt.bar(x + width/2, CVaRs, width, label="CVaR")

    plt.title("不同计算方法的VaR和CVaR比较")
    plt.xlabel("计算方法")
    plt.ylabel("风险价值")
    plt.xticks(x, methods)
    plt.legend()
    plt.grid(axis="y")
    plt.savefig("risk_metrics_comparison.png")
    plt.show()

# 使用示例
# plot_volatility_evolution(processed_stock, symbols)
# plot_correlation_matrix(final_data, symbols)
# plot_return_distribution(final_data, symbols)
""",
        "explanation": "此代码演示如何可视化风险管理过程中的各类数据和指标，包括波动率演变、相关系数矩阵、收益率分布和风险指标比较图表。可视化帮助分析师直观理解风险特征。"
    },
    {
        "topic_id": 3,
        "category_id": 5,
        "title": "机器学习在风险管理中的应用",
        "code": """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

class RiskPredictor:
    def __init__(self, data, threshold=0.02):
        self.data = data
        self.threshold = threshold
        self.scaler = StandardScaler()

    def create_features(self, symbol):
        \"\"\"创建特征数据\"\"\"
        features = pd.DataFrame()
        features["Return"] = self.data[f"{symbol}_Return"]
        features["Volatility"] = self.data[f"{symbol}_Volatility"]
        features["SP500_Return"] = self.data["SP500_Return"]
        features["SP500_Volatility"] = self.data["SP500_Volatility"]
        features["Inflation"] = self.data["Inflation"]
        features["InterestRate"] = self.data["InterestRate"]

        # 创建滞后特征
        for i in range(1, 6):
            features[f"Return_Lag{i}"] = self.data[f"{symbol}_Return"].shift(i)

        # 创建技术指标特征
        features = features.dropna()
        return features

    def create_labels(self, symbol):
        \"\"\"创建标签数据\"\"\"
        returns = self.data[f"{symbol}_Return"]
        labels = (returns < -self.threshold).astype(int)
        return labels

    def train_model(self, symbol):
        \"\"\"训练风险预测模型\"\"\"
        X = self.create_features(symbol)
        y = self.create_labels(symbol)[X.index]

        # 划分训练和测试数据
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 标准化数据
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 训练随机森林模型
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)

        # 预测
        y_pred = model.predict(X_test_scaled)

        # 评估模型
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)

        return model, report, matrix

    def predict_risk(self, model, symbol, future_dates):
        \"\"\"预测未来风险\"\"\"
        features = self.create_features(symbol)
        last_features = features.tail(1)
        last_features_scaled = self.scaler.transform(last_features)

        risk_prediction = model.predict(last_features_scaled)
        risk_probability = model.predict_proba(last_features_scaled)[0][1]

        return risk_prediction, risk_probability

# 使用示例
# predictor = RiskPredictor(final_data, threshold=0.02)
# model, report, matrix = predictor.train_model("AAPL")
# print("模型分类报告:")
# print(report)
# print("混淆矩阵:")
# print(matrix)

# risk_pred, risk_prob = predictor.predict_risk(model, "AAPL", [pd.Timestamp("2025-01-02")])
# print(f"预测风险发生: {risk_pred}")
# print(f"风险概率: {risk_prob:.4f}")
""",
        "explanation": "此代码演示如何使用机器学习方法预测金融风险事件，包括特征工程、模型训练、预测和评估。通过机器学习，我们可以更准确地识别潜在的风险信号。"
    },
    {
        "topic_id": 3,
        "category_id": 6,
        "title": "完整风险管理系统",
        "code": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RiskManagementSystem:
    def __init__(self):
        self.data = None
        self.stock_data = None
        self.index_data = None
        self.macro_data = None
        self.processed_data = None
        self.risk_analyst = None

    def load_and_prepare_data(self, symbols, start_date, end_date):
        \"\"\"加载和准备数据\"\"\"
        from data_fetcher import fetch_stock_data, fetch_index_data, fetch_macro_data
        from data_preprocessor import preprocess_stock_data, preprocess_index_data, merge_data

        print("开始加载数据...")
        self.stock_data = fetch_stock_data(symbols, start_date, end_date)
        self.index_data = fetch_index_data("^GSPC", start_date, end_date)
        self.macro_data = fetch_macro_data()

        print("开始预处理数据...")
        processed_stock = preprocess_stock_data(self.stock_data)
        processed_index = preprocess_index_data(self.index_data)

        print("开始合并数据...")
        self.processed_data = merge_data(processed_stock, processed_index, self.macro_data)

        return self.processed_data

    def initialize_risk_analyst(self):
        \"\"\"初始化风险分析师\"\"\"
        from risk_analyst import RiskAnalyst

        self.risk_analyst = RiskAnalyst(self.processed_data)
        return self.risk_analyst

    def run_portfolio_risk_analysis(self, weights, symbols):
        \"\"\"运行投资组合风险分析\"\"\"
        if self.risk_analyst is None:
            self.initialize_risk_analyst()

        print("开始投资组合风险分析...")
        return self.risk_analyst.analyze_portfolio_risk(weights, symbols)

    def run_stress_testing(self, scenario, symbols):
        \"\"\"运行压力测试\"\"\"
        # 这里可以添加不同的压力测试场景
        # 比如：金融危机、经济衰退、市场崩溃等
        print(f"运行压力测试场景: {scenario}")

        if scenario == "severe_recession":
            # 模拟严重衰退场景：市场下跌20%，波动率增加3倍
            stress_data = self.processed_data.copy()
            for symbol in symbols:
                stress_data[f"{symbol}_Return"] = stress_data[f"{symbol}_Return"] * 0.8
                stress_data[f"{symbol}_Volatility"] = stress_data[f"{symbol}_Volatility"] * 3
            stress_data["SP500_Return"] = stress_data["SP500_Return"] * 0.8
            stress_data["SP500_Volatility"] = stress_data["SP500_Volatility"] * 3
            stress_data["InterestRate"] = stress_data["InterestRate"] * 1.5

            # 在压力场景下重新评估风险
            stress_analyst = RiskAnalyst(stress_data)
            stress_report = stress_analyst.analyze_portfolio_risk([0.25, 0.25, 0.25, 0.25], symbols)

            return stress_report
        else:
            return {"Error": "未知压力测试场景"}

    def generate_risk_report(self, portfolio_report, stress_report=None):
        \"\"\"生成风险报告\"\"\"
        print("\\n=== 风险管理报告 ===")
        print("\\n投资组合基础风险:")
        for key, value in portfolio_report.items():
            if key in ["Return", "Volatility"]:
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value:.2%}")

        if stress_report and "Error" not in stress_report:
            print("\\n压力测试结果:")
            for key, value in stress_report.items():
                if key in ["Return", "Volatility"]:
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value:.2%}")

        return "Risk report generated successfully"

# 使用示例
if __name__ == "__main__":
    system = RiskManagementSystem()
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
    start_date = "2015-01-01"
    end_date = "2025-01-01"

    try:
        print("1. 加载和准备数据")
        system.load_and_prepare_data(symbols, start_date, end_date)

        print("\\n2. 初始化风险分析师")
        system.initialize_risk_analyst()

        print("\\n3. 投资组合风险分析")
        report = system.run_portfolio_risk_analysis([0.25, 0.25, 0.25, 0.25], symbols)

        print("\\n4. 压力测试")
        stress_report = system.run_stress_testing("severe_recession", symbols)

        print("\\n5. 生成风险报告")
        report_result = system.generate_risk_report(report, stress_report)
        print(report_result)

    except Exception as e:
        print(f"错误: {e}")
""",
        "explanation": "这是一个完整的风险管理系统，集成了数据获取、预处理、风险分析和报告功能。该系统展示了如何构建一个全面的风险管理平台，支持投资组合风险分析和压力测试。"
    },
    # 主题2：量化交易策略
    {
        "topic_id": 2,
        "category_id": 1,
        "title": "获取市场数据",
        "code": """import yfinance as yf
import pandas as pd
import numpy as np

# 获取股票数据
def get_stock_data(symbol, start_date, end_date):
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    return data

# 获取S&P 500指数数据
sp500_data = get_stock_data("^GSPC", "2015-01-01", "2025-01-01")
print("S&P 500指数数据形状:", sp500_data.shape)
print(sp500_data.head())

# 获取多个股票数据
symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
all_data = pd.DataFrame()
for symbol in symbols:
    try:
        data = get_stock_data(symbol, "2015-01-01", "2025-01-01")
        data["Symbol"] = symbol
        all_data = pd.concat([all_data, data])
    except Exception as e:
        print(f"Error getting data for {symbol}: {e}")

print("所有股票数据形状:", all_data.shape)
print("数据说明:", all_data.describe())
""",
        "explanation": "此代码演示如何使用yfinance库获取股票数据，包括单只股票和多只股票的数据获取方法，并对获取到的数据进行基本的查看和统计分析。量化交易策略的基础是可靠的数据获取。"
    },
    {
        "topic_id": 2,
        "category_id": 2,
        "title": "数据预处理",
        "code": """import pandas as pd
import numpy as np

# 加载数据
data = pd.read_csv("sp500_data.csv", index_col="Date", parse_dates=True)

# 数据清洗
print("缺失值检查:")
print(data.isnull().sum())

# 填充缺失值
data = data.fillna(method="ffill")

# 计算收益率
data["Return"] = data["Close"].pct_change()

# 计算移动平均线
data["MA5"] = data["Close"].rolling(window=5).mean()
data["MA20"] = data["Close"].rolling(window=20).mean()
data["MA50"] = data["Close"].rolling(window=50).mean()

# 计算布林带
data["UpperBB"] = data["MA20"] + 2 * data["Close"].rolling(window=20).std()
data["LowerBB"] = data["MA20"] - 2 * data["Close"].rolling(window=20).std()

# 计算RSI指标
def calculate_rsi(data, window=14):
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data["RSI"] = calculate_rsi(data)

# 保存处理后的数据
data.to_csv("sp500_processed_data.csv")
print("预处理完成")
print(data.head())
""",
        "explanation": "此代码演示如何对量化交易数据进行预处理，包括数据清洗、计算收益率、移动平均线、布林带和RSI等常用技术指标。这些处理步骤是量化策略开发的基础。"
    },
    {
        "topic_id": 2,
        "category_id": 3,
        "title": "简单移动平均线策略",
        "code": """import pandas as pd
import numpy as np

# 加载数据
data = pd.read_csv("sp500_processed_data.csv", index_col="Date", parse_dates=True)

# 策略参数
short_window = 5
long_window = 20

# 生成交易信号
data["Signal"] = 0.0
data["Signal"][short_window:] = np.where(
    data["Close"][short_window:] > data["MA20"][short_window:], 1.0, 0.0
)
data["Position"] = data["Signal"].diff()

# 回测策略
initial_capital = 100000.0
positions = pd.DataFrame(index=data.index).fillna(0.0)
positions["SP500"] = 100 * data["Signal"]
portfolio = positions.multiply(data["Close"], axis=0)

# 计算投资组合价值
pos_diff = positions.diff()
portfolio["Holdings"] = (positions.multiply(data["Close"], axis=0)).sum(axis=1)
portfolio["Cash"] = initial_capital - (pos_diff.multiply(data["Close"], axis=0)).sum(axis=1).cumsum()
portfolio["Total"] = portfolio["Cash"] + portfolio["Holdings"]
portfolio["Returns"] = portfolio["Total"].pct_change()

print("策略回测结果:")
print(portfolio.tail())
print(f"最终投资组合价值: ${portfolio['Total'][-1]:.2f}")
print(f"总收益率: {((portfolio['Total'][-1] / initial_capital) - 1) * 100:.2f}%")
""",
        "explanation": "此代码演示如何实现简单的移动平均线交叉策略，包括交易信号生成、投资组合管理和策略回测。这是量化交易策略中最基础的策略之一。"
    },
    {
        "topic_id": 2,
        "category_id": 4,
        "title": "策略可视化分析",
        "code": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 加载数据和回测结果
data = pd.read_csv("sp500_processed_data.csv", index_col="Date", parse_dates=True)
portfolio = pd.read_csv("portfolio.csv", index_col="Date", parse_dates=True)

# 绘制价格和移动平均线
plt.figure(figsize=(12, 6))
plt.plot(data["Close"], label="收盘价")
plt.plot(data["MA5"], label="5日均线")
plt.plot(data["MA20"], label="20日均线")

# 绘制交易信号
buy_signals = data[data["Position"] == 1.0]
sell_signals = data[data["Position"] == -1.0]
plt.scatter(buy_signals.index, data["Close"][buy_signals.index], marker="^", color="g", label="买入信号")
plt.scatter(sell_signals.index, data["Close"][sell_signals.index], marker="v", color="r", label="卖出信号")

plt.title("S&P 500价格走势和交易信号")
plt.xlabel("日期")
plt.ylabel("价格")
plt.legend()
plt.grid(True)
plt.savefig("price_with_signals.png")

# 绘制投资组合价值
plt.figure(figsize=(12, 6))
plt.plot(portfolio["Total"], label="投资组合价值")
plt.title("投资组合价值变化")
plt.xlabel("日期")
plt.ylabel("价值")
plt.legend()
plt.grid(True)
plt.savefig("portfolio_value.png")

# 绘制收益率直方图
plt.figure(figsize=(12, 6))
sns.histplot(portfolio["Returns"].dropna(), kde=True)
plt.title("投资组合收益率分布")
plt.xlabel("收益率")
plt.ylabel("频率")
plt.grid(True)
plt.savefig("returns_distribution.png")

plt.show()
""",
        "explanation": "此代码演示如何可视化量化交易策略的执行情况，包括价格走势、交易信号和投资组合价值变化。通过图表分析可以更好地理解策略的表现。"
    },
    {
        "topic_id": 2,
        "category_id": 5,
        "title": "机器学习交易策略",
        "code": """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv("sp500_processed_data.csv", index_col="Date", parse_dates=True)

# 准备特征和标签
data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
features = ["MA5", "MA20", "MA50", "UpperBB", "LowerBB", "RSI"]
X = data[features].dropna()
y = data["Target"].dropna()

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 特征重要性
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(feature_importance["Feature"], feature_importance["Importance"])
plt.title("特征重要性")
plt.xlabel("特征")
plt.ylabel("重要性")
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig("feature_importance.png")

# 保存预测结果
data["Prediction"] = 0
data.loc[X_test.index, "Prediction"] = y_pred
data.to_csv("predicted_data.csv")
""",
        "explanation": "此代码演示如何使用机器学习方法（随机森林）构建交易策略，包括数据准备、模型训练、预测和评估。机器学习可以帮助我们识别更复杂的交易模式。"
    },
    {
        "topic_id": 2,
        "category_id": 6,
        "title": "完整量化交易系统",
        "code": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf

class QuantitativeTradingSystem:
    def __init__(self, capital=100000.0, commission=0.001):
        self.capital = capital
        self.cash = capital
        self.shares_held = 0
        self.commission = commission
        self.trades = []

    def load_data(self, symbol, start_date, end_date):
        self.data = yf.download(symbol, start=start_date, end=end_date)
        self.calculate_indicators()

    def calculate_indicators(self):
        self.data["Return"] = self.data["Close"].pct_change()
        self.data["MA5"] = self.data["Close"].rolling(window=5).mean()
        self.data["MA20"] = self.data["Close"].rolling(window=20).mean()
        self.data["UpperBB"] = self.data["MA20"] + 2 * self.data["Close"].rolling(window=20).std()
        self.data["LowerBB"] = self.data["MA20"] - 2 * self.data["Close"].rolling(window=20).std()

    def generate_signals(self):
        self.data["Signal"] = 0.0
        self.data["Signal"] = np.where(
            (self.data["Close"] < self.data["LowerBB"]) & (self.data["MA5"] > self.data["MA20"]), 1.0, 0.0
        )
        self.data["Position"] = self.data["Signal"].diff()

    def backtest_strategy(self):
        portfolio = pd.DataFrame(index=self.data.index)
        portfolio["Holdings"] = self.shares_held * self.data["Close"]
        portfolio["Cash"] = self.cash
        portfolio["Total"] = portfolio["Holdings"] + portfolio["Cash"]

        for date in self.data.index:
            if self.data.loc[date, "Position"] == 1.0 and self.cash > 0:
                max_shares = int(self.cash / (self.data.loc[date, "Close"] * (1 + self.commission)))
                cost = max_shares * self.data.loc[date, "Close"] * (1 + self.commission)
                self.cash -= cost
                self.shares_held += max_shares
                self.trades.append({
                    "Date": date,
                    "Type": "Buy",
                    "Price": self.data.loc[date, "Close"],
                    "Shares": max_shares
                })

            elif self.data.loc[date, "Position"] == -1.0 and self.shares_held > 0:
                revenue = self.shares_held * self.data.loc[date, "Close"] * (1 - self.commission)
                self.cash += revenue
                self.trades.append({
                    "Date": date,
                    "Type": "Sell",
                    "Price": self.data.loc[date, "Close"],
                    "Shares": self.shares_held
                })
                self.shares_held = 0

            portfolio.loc[date, "Holdings"] = self.shares_held * self.data.loc[date, "Close"]
            portfolio.loc[date, "Cash"] = self.cash
            portfolio.loc[date, "Total"] = portfolio.loc[date, "Holdings"] + portfolio.loc[date, "Cash"]

        portfolio["Returns"] = portfolio["Total"].pct_change()
        return portfolio

    def print_performance(self, portfolio):
        final_value = portfolio["Total"][-1]
        total_return = ((final_value / self.capital) - 1) * 100
        annualized_return = (1 + total_return / 100) ** (252 / len(self.data)) - 1

        print(f"初始资本: ${self.capital:.2f}")
        print(f"最终价值: ${final_value:.2f}")
        print(f"总收益率: {total_return:.2f}%")
        print(f"年化收益率: {annualized_return * 100:.2f}%")
        print(f"交易次数: {len(self.trades)}")

    def plot_results(self, portfolio):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(portfolio["Total"], label="投资组合价值")
        plt.title("投资组合价值变化")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.data["Close"], label="价格")
        plt.plot(self.data["MA5"], label="5日均线")
        plt.plot(self.data["MA20"], label="20日均线")
        plt.plot(self.data["UpperBB"], label="上布林带")
        plt.plot(self.data["LowerBB"], label="下布林带")
        plt.title("价格和技术指标")
        plt.legend()

        plt.tight_layout()
        plt.savefig("trading_system_results.png")

if __name__ == "__main__":
    system = QuantitativeTradingSystem(capital=100000)
    system.load_data("^GSPC", "2015-01-01", "2025-01-01")
    system.generate_signals()
    portfolio = system.backtest_strategy()
    system.print_performance(portfolio)
    system.plot_results(portfolio)
    print("\\n交易详情:")
    for trade in system.trades:
        print(f"{trade['Date']} - {trade['Type']}: {trade['Shares']}股 @ {trade['Price']:.2f}")
""",
        "explanation": "这是一个完整的量化交易系统类，包含数据加载、指标计算、信号生成、策略回测和结果分析等核心功能。该系统实现了一个基于布林带和移动平均线的交易策略，展示了量化交易的完整流程。"
    },
    {
        "topic_id": 1,
        "category_id": 1,
        "title": "使用Yahoo Finance获取股票数据",
        "code": """import yfinance as yf
import pandas as pd

# 获取苹果公司股票数据
apple = yf.Ticker("AAPL")

# 获取历史数据
history = apple.history(period="1y")
print("苹果公司近1年股票数据:")
print(history.head())

# 获取公司基本信息
info = apple.info
print("\\n公司基本信息:")
print(f"公司名称: {info['longName']}")
print(f"当前价格: {info['currentPrice']}")
print(f"市值: {info['marketCap']}")
""",
        "explanation": "此代码演示如何使用yfinance库从Yahoo Finance获取股票历史数据和公司基本信息。yfinance是一个强大的金融数据获取库，支持获取全球股票、指数、ETF等金融产品的数据。"
    },
    {
        "topic_id": 1,
        "category_id": 2,
        "title": "股票数据预处理",
        "code": """import yfinance as yf
import pandas as pd

# 获取股票数据
df = yf.download("AAPL", start="2020-01-01", end="2023-12-31")

# 数据清洗
# 检查缺失值
print("缺失值数量:")
print(df.isnull().sum())

# 填充缺失值
df = df.fillna(method="ffill")

# 计算技术指标
df["MA5"] = df["Close"].rolling(window=5).mean()
df["MA20"] = df["Close"].rolling(window=20).mean()
df["Return"] = df["Close"].pct_change()

# 保存处理后的数据
df.to_csv("apple_stock_processed.csv")
print("数据处理完成，已保存到apple_stock_processed.csv")
""",
        "explanation": "此代码演示如何对股票数据进行预处理，包括检查和填充缺失值，计算移动平均线等技术指标，以及计算收益率。这些处理步骤是金融数据分析的基础。"
    },
    {
        "topic_id": 1,
        "category_id": 3,
        "title": "股票数据分析",
        "code": """import pandas as pd
import numpy as np
from scipy.stats import norm

# 读取处理后的数据
df = pd.read_csv("apple_stock_processed.csv", index_col="Date", parse_dates=True)

# 计算基本统计量
print("基本统计量:")
print(df["Close"].describe())

# 计算收益率分布
returns = df["Return"].dropna()
print("\\n收益率统计:")
print(returns.describe())

# 计算VaR (Value at Risk)
confidence_level = 0.95
VaR = norm.ppf(1 - confidence_level, returns.mean(), returns.std())
print(f"\\n95%置信水平的VaR: {VaR:.4f}")

# 计算最大回撤
def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

max_drawdown = calculate_max_drawdown(returns)
print(f"最大回撤: {max_drawdown:.4f}")
""",
        "explanation": "此代码演示如何对股票数据进行深入分析，包括计算基本统计量、收益率分布、风险价值(VaR)和最大回撤等重要的金融指标。这些分析对于投资决策至关重要。"
    },
    {
        "topic_id": 1,
        "category_id": 4,
        "title": "股票数据可视化",
        "code": """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 读取数据
df = pd.read_csv("apple_stock_processed.csv", index_col="Date", parse_dates=True)

# 创建图表
plt.figure(figsize=(12, 8))

# 收盘价和移动平均线
plt.subplot(2, 2, 1)
plt.plot(df["Close"], label="收盘价")
plt.plot(df["MA5"], label="5日均线")
plt.plot(df["MA20"], label="20日均线")
plt.title("苹果公司股票价格走势")
plt.legend()

# 成交量
plt.subplot(2, 2, 2)
plt.bar(df.index, df["Volume"])
plt.title("成交量")

# 收益率分布
plt.subplot(2, 2, 3)
sns.histplot(df["Return"].dropna(), kde=True)
plt.title("收益率分布")

# 相关系数矩阵
plt.subplot(2, 2, 4)
corr_matrix = df[["Open", "High", "Low", "Close", "Volume"]].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("相关系数矩阵")

plt.tight_layout()
plt.savefig("stock_analysis.png", dpi=300, bbox_inches="tight")
plt.show()
""",
        "explanation": "此代码演示如何使用matplotlib和seaborn库可视化股票数据，包括价格走势、成交量、收益率分布和相关系数矩阵。可视化是金融数据分析中重要的部分，可以帮助我们更好地理解数据。"
    },
    {
        "topic_id": 1,
        "category_id": 5,
        "title": "使用机器学习预测股票价格",
        "code": """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("apple_stock_processed.csv", index_col="Date", parse_dates=True)

# 创建特征和标签
features = ["Open", "High", "Low", "Volume", "MA5", "MA20"]
X = df[features]
y = df["Close"]

# 划分训练和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建和训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"均方误差: {mse:.2f}")
print(f"R²得分: {r2:.2f}")

# 可视化预测结果
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="真实价格")
plt.plot(y_test.index, y_pred, label="预测价格")
plt.title("股票价格预测")
plt.legend()
plt.savefig("prediction.png", dpi=300)
plt.show()

# 输出特征重要性
feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.coef_
}).sort_values(by="Importance", ascending=False)
print("\\n特征重要性:")
print(feature_importance)
""",
        "explanation": "此代码演示如何使用线性回归模型预测股票价格。我们使用历史价格、成交量和移动平均线作为特征，训练模型并评估其性能。机器学习在金融预测中具有广泛的应用前景。"
    },
    {
        "topic_id": 1,
        "category_id": 6,
        "title": "完整的股票分析应用",
        "code": """import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

class StockAnalyzer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self.data = None

    def download_data(self, period="1y"):
        \"\"\"下载股票历史数据\"\"\"
        self.data = self.ticker.history(period=period)
        return self.data

    def calculate_technical_indicators(self):
        \"\"\"计算技术指标\"\"\"
        if self.data is None:
            raise ValueError("请先下载数据")

        # 移动平均线
        self.data["MA5"] = self.data["Close"].rolling(window=5).mean()
        self.data["MA20"] = self.data["Close"].rolling(window=20).mean()

        # 收益率
        self.data["Return"] = self.data["Close"].pct_change()

        return self.data

    def analyze_risk(self, confidence_level=0.95):
        \"\"\"风险分析\"\"\"
        returns = self.data["Return"].dropna()

        # VaR计算
        VaR = norm.ppf(1 - confidence_level, returns.mean(), returns.std())

        # 最大回撤计算
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding(min_periods=1).max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        return {
            "VaR": VaR,
            "max_drawdown": max_drawdown,
            "mean_return": returns.mean(),
            "std_return": returns.std()
        }

    def visualize(self):
        \"\"\"数据可视化\"\"\"
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 价格走势
        axes[0, 0].plot(self.data.index, self.data["Close"], label="收盘价")
        axes[0, 0].plot(self.data.index, self.data["MA5"], label="5日均线")
        axes[0, 0].plot(self.data.index, self.data["MA20"], label="20日均线")
        axes[0, 0].set_title(f"{self.symbol}股票价格走势")
        axes[0, 0].legend()

        # 成交量
        axes[0, 1].bar(self.data.index, self.data["Volume"])
        axes[0, 1].set_title("成交量")

        # 收益率分布
        sns.histplot(self.data["Return"].dropna(), kde=True, ax=axes[1, 0])
        axes[1, 0].set_title("收益率分布")

        # 相关系数矩阵
        corr_matrix = self.data[["Open", "High", "Low", "Close", "Volume"]].corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=axes[1, 1])
        axes[1, 1].set_title("相关系数矩阵")

        plt.tight_layout()
        plt.savefig(f"{self.symbol}_analysis.png", dpi=300)
        plt.show()

# 使用示例
if __name__ == "__main__":
    analyzer = StockAnalyzer("AAPL")
    analyzer.download_data()
    analyzer.calculate_technical_indicators()

    # 显示基本信息
    info = analyzer.ticker.info
    print(f"公司名称: {info['longName']}")
    print(f"当前价格: {info['currentPrice']:.2f}")

    # 风险分析
    risk_info = analyzer.analyze_risk()
    print(f"\\n风险分析结果:")
    print(f"平均收益率: {risk_info['mean_return']:.4f}")
    print(f"收益率标准差: {risk_info['std_return']:.4f}")
    print(f"95%置信水平VaR: {risk_info['VaR']:.4f}")
    print(f"最大回撤: {risk_info['max_drawdown']:.4f}")

    # 可视化
    analyzer.visualize()
""",
        "explanation": "这是一个完整的股票分析应用程序，它将之前演示的所有功能整合到一个类中。这个应用程序可以下载数据、计算技术指标、进行风险分析和数据可视化。通过这个示例，您可以了解如何构建一个完整的金融应用项目。"
    },
    # 主题6：债券计算工具
    {
        "topic_id": 6,
        "category_id": 1,
        "title": "债券基本信息获取",
        "code": """import requests
import pandas as pd
from datetime import datetime

def get_treasury_rates():
    \"\"\"获取美国国债收益率数据\"\"\"
    try:
        # 使用Yahoo Finance获取10年期美国国债收益率数据
        url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ETNX"
        params = {
            "period1": int(datetime(2024, 1, 1).timestamp()),
            "period2": int(datetime.now().timestamp()),
            "interval": "1d",
            "includePrePost": "false"
        }

        response = requests.get(url, params=params)
        data = response.json()

        if 'chart' in data and 'result' in data['chart'] and len(data['chart']['result']) > 0:
            timestamps = data['chart']['result'][0]['timestamp']
            rates = data['chart']['result'][0]['indicators']['quote'][0]['close']

            df = pd.DataFrame({
                'Date': [datetime.fromtimestamp(ts) for ts in timestamps],
                'Rate': rates
            })
            df.set_index('Date', inplace=True)
            return df

        return pd.DataFrame()

    except Exception as e:
        print(f"获取美国国债收益率数据失败: {str(e)}")
        return pd.DataFrame()

def get_corporate_bond_data():
    \"\"\"获取公司债券数据\"\"\"
    try:
        # 创建一个示例公司债券数据
        data = {
            'Name': ['Apple Inc.', 'Microsoft Corp.', 'Amazon.com Inc.',
                    'Google LLC', 'Facebook Inc.', 'Tesla Inc.'],
            'Symbol': ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB', 'TSLA'],
            'Rating': ['AA+', 'AAA', 'AA', 'AA+', 'A+', 'B-'],
            'Coupon Rate (%)': [4.25, 3.75, 4.50, 4.00, 3.50, 5.75],
            'Maturity Date': ['2034-01-15', '2033-06-30', '2035-12-01',
                            '2034-09-30', '2033-03-15', '2032-11-15'],
            'YTM (%)': [4.50, 4.25, 4.75, 4.35, 4.10, 6.25],
            'Price': [98.50, 101.25, 97.75, 100.50, 102.75, 95.25]
        }

        df = pd.DataFrame(data)
        df['Maturity Date'] = pd.to_datetime(df['Maturity Date'])

        return df

    except Exception as e:
        print(f"获取公司债券数据失败: {str(e)}")
        return pd.DataFrame()

def get_bond_quote(symbol):
    \"\"\"获取单个债券报价\"\"\"
    try:
        # 这里使用示例数据
        bond_data = {
            'AAPL.BN': {
                'Name': 'Apple Inc. Bond 2034',
                'Rating': 'AA+',
                'Coupon': 4.25,
                'Maturity': '2034-01-15',
                'Price': 98.50,
                'YTM': 4.50
            },
            'MSFT.BN': {
                'Name': 'Microsoft Corp. Bond 2033',
                'Rating': 'AAA',
                'Coupon': 3.75,
                'Maturity': '2033-06-30',
                'Price': 101.25,
                'YTM': 4.25
            },
            'AMZN.BN': {
                'Name': 'Amazon.com Inc. Bond 2035',
                'Rating': 'AA',
                'Coupon': 4.50,
                'Maturity': '2035-12-01',
                'Price': 97.75,
                'YTM': 4.75
            }
        }

        if symbol in bond_data:
            return pd.DataFrame([bond_data[symbol]])
        else:
            return pd.DataFrame()

    except Exception as e:
        print(f"获取债券报价失败: {str(e)}")
        return pd.DataFrame()

# 使用示例
# 获取国债收益率数据
treasury_rates = get_treasury_rates()
print("美国国债收益率数据:\\n", treasury_rates.tail())

# 获取公司债券数据
corporate_bonds = get_corporate_bond_data()
print("\\n公司债券数据:\\n", corporate_bonds.head())

# 获取单个债券报价
apple_bond = get_bond_quote('AAPL.BN')
print("\\nApple公司债券报价:\\n", apple_bond)
""",
        "explanation": "此代码演示如何获取债券市场数据，包括美国国债收益率、公司债券数据和单个债券报价。我们使用模拟数据创建了一个示例，实际应用中需要访问专业的债券数据API。"
    },
    {
        "topic_id": 6,
        "category_id": 2,
        "title": "债券数据预处理",
        "code": """import pandas as pd
import numpy as np
from datetime import datetime

def calculate_time_to_maturity(maturity_date):
    \"\"\"计算剩余期限\"\"\"
    today = datetime.now()
    time_to_maturity = (maturity_date - today).days / 365
    return max(time_to_maturity, 0)

def calculate_accrued_interest(settlement_date, maturity_date, coupon_rate, face_value=1000, frequency=2):
    \"\"\"计算应计利息\"\"\"
    period_days = 365 / frequency
    last_coupon_date = maturity_date
    while last_coupon_date > settlement_date:
        last_coupon_date -= pd.DateOffset(months=6)

    next_coupon_date = last_coupon_date + pd.DateOffset(months=6)
    days_since_coupon = (settlement_date - last_coupon_date).days
    days_in_period = (next_coupon_date - last_coupon_date).days

    accrued_interest = (coupon_rate / 100 / frequency) * (days_since_coupon / days_in_period) * face_value

    return accrued_interest

def bond_data_preprocessing(bond_data):
    \"\"\"债券数据预处理\"\"\"
    processed_data = bond_data.copy()

    processed_data['Time to Maturity (Years)'] = processed_data['Maturity Date'].apply(calculate_time_to_maturity)

    processed_data['Accrued Interest'] = processed_data.apply(
        lambda row: calculate_accrued_interest(
            settlement_date=datetime.now(),
            maturity_date=row['Maturity Date'],
            coupon_rate=row['Coupon Rate (%)'],
            face_value=1000
        ),
        axis=1
    )

    processed_data['Duration'] = processed_data.apply(
        lambda row: (row['Coupon Rate (%)'] / 100) / row['YTM (%)'] * 100,
        axis=1
    )

    processed_data['Convexity'] = processed_data.apply(
        lambda row: ((1 + row['YTM (%)'] / 100) / (row['YTM (%)'] / 100)) ** 2,
        axis=1
    )

    processed_data['Price Change (%)'] = 0

    processed_data['Price Category'] = processed_data['Price'].apply(
        lambda x: '溢价' if x > 100 else '折价' if x < 100 else '平价'
    )

    return processed_data

# 使用示例
# 假设我们有之前获取的公司债券数据
# corporate_bonds = get_corporate_bond_data()
# processed_bonds = bond_data_preprocessing(corporate_bonds)
# print("\\n预处理后的公司债券数据:\\n", processed_bonds.head())
""",
        "explanation": "此代码演示如何对债券数据进行预处理，包括计算剩余期限、应计利息、久期和凸性等重要指标。这些指标对于债券分析和投资决策非常重要。"
    },
    {
        "topic_id": 6,
        "category_id": 3,
        "title": "债券定价与YTM计算",
        "code": """import math
from datetime import datetime
import pandas as pd

def calculate_bond_price(coupon_rate, face_value, ytm, time_to_maturity, frequency=2):
    \"\"\"计算债券价格\"\"\"
    price = 0
    coupon_payment = (coupon_rate / 100 / frequency) * face_value
    periods = time_to_maturity * frequency

    for i in range(1, int(periods) + 1):
        price += coupon_payment / (1 + (ytm / 100 / frequency)) ** i

    price += face_value / (1 + (ytm / 100 / frequency)) ** periods

    return price

def calculate_ytm(price, coupon_rate, face_value, time_to_maturity, frequency=2):
    \"\"\"计算到期收益率(YTM)\"\"\"
    tolerance = 1e-5
    max_iterations = 1000

    ytm_guess = coupon_rate

    for _ in range(max_iterations):
        price_guess = calculate_bond_price(coupon_rate, face_value, ytm_guess, time_to_maturity, frequency)
        price_diff = price_guess - price

        if abs(price_diff) < tolerance:
            return ytm_guess

        derivative = 0
        coupon_payment = (coupon_rate / 100 / frequency) * face_value
        periods = time_to_maturity * frequency

        for i in range(1, int(periods) + 1):
            derivative -= i * coupon_payment / (1 + (ytm_guess / 100 / frequency)) ** (i + 1)
            derivative -= i * face_value / (1 + (ytm_guess / 100 / frequency)) ** (i + 1)

        ytm_guess -= price_diff / derivative

    return None

def calculate_accrued_interest(settlement_date, maturity_date, coupon_rate, face_value=1000, frequency=2):
    \"\"\"计算应计利息\"\"\"
    period_days = 365 / frequency
    last_coupon_date = maturity_date
    while last_coupon_date > settlement_date:
        last_coupon_date -= pd.DateOffset(months=6)

    next_coupon_date = last_coupon_date + pd.DateOffset(months=6)
    days_since_coupon = (settlement_date - last_coupon_date).days
    days_in_period = (next_coupon_date - last_coupon_date).days

    accrued_interest = (coupon_rate / 100 / frequency) * (days_since_coupon / days_in_period) * face_value

    return accrued_interest

# 使用示例
# 计算债券价格
price = calculate_bond_price(4.25, 1000, 4.50, 10, 2)
print(f\"债券价格: ${price:.2f}\")

# 计算YTM
ytm = calculate_ytm(985, 4.25, 1000, 10, 2)
print(f\"到期收益率: {ytm:.2f}%\")

# 计算应计利息
settlement_date = datetime(2024, 4, 9)
maturity_date = datetime(2034, 1, 15)

accrued_interest = calculate_accrued_interest(settlement_date, maturity_date, 4.25, 1000, 2)
print(f\"应计利息: ${accrued_interest:.2f}\")
""",
        "explanation": "此代码演示如何计算债券价格和到期收益率(YTM)。债券价格是根据债券的票面利率、面值、到期收益率和剩余期限计算得出的。YTM是使债券未来现金流现值等于当前价格的贴现率。"
    },
    {
        "topic_id": 6,
        "category_id": 4,
        "title": "债券数据可视化",
        "code": """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_coupon_vs_price(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Coupon Rate (%)', y='Price')
    plt.title('票面利率与价格关系')
    plt.xlabel('票面利率 (%)')
    plt.ylabel('价格')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('coupon_vs_price.png')
    plt.show()

def plot_rating_vs_price(df):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Rating', y='Price')
    plt.title('信用评级与价格关系')
    plt.xlabel('信用评级')
    plt.ylabel('价格')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('rating_vs_price.png')
    plt.show()

def plot_duration_vs_price(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Duration', y='Price')
    plt.title('久期与价格关系')
    plt.xlabel('久期')
    plt.ylabel('价格')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('duration_vs_price.png')
    plt.show()

def plot_price_by_rating(df):
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='Rating', y='Price')
    sns.stripplot(data=df, x='Rating', y='Price', color='black', alpha=0.3)
    plt.title('按信用评级分组的价格分布')
    plt.xlabel('信用评级')
    plt.ylabel('价格')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('price_by_rating.png')
    plt.show()

def plot_bond_cash_flows(coupon_rate, face_value, time_to_maturity, frequency=2):
    cash_flows = []
    coupon_payment = (coupon_rate / 100 / frequency) * face_value
    periods = int(time_to_maturity * frequency)

    for i in range(1, periods + 1):
        cash_flows.append(coupon_payment)

    cash_flows[-1] += face_value

    plt.figure(figsize=(12, 6))
    plt.bar(range(1, periods + 1), cash_flows, color='skyblue')
    plt.title('债券现金流量图')
    plt.xlabel('付息期')
    plt.ylabel('现金流')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cash_flows.png')
    plt.show()

def plot_yield_curve(treasury_rates):
    if not treasury_rates.empty:
        plt.figure(figsize=(12, 8))
        plt.plot(treasury_rates.index, treasury_rates['Rate'])
        plt.title('美国国债收益率曲线')
        plt.xlabel('日期')
        plt.ylabel('收益率 (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('yield_curve.png')
        plt.show()

# 使用示例（假设已经有处理过的数据）
# corporate_bonds = get_corporate_bond_data()
# processed_bonds = bond_data_preprocessing(corporate_bonds)
# plot_coupon_vs_price(processed_bonds)
# plot_rating_vs_price(processed_bonds)
# plot_duration_vs_price(processed_bonds)
""",
        "explanation": "此代码演示如何可视化债券数据，包括票面利率与价格关系、信用评级与价格关系、久期与价格关系以及收益率曲线等图表。这些图表可以帮助我们更好地理解债券市场趋势。"
    },
    {
        "topic_id": 6,
        "category_id": 5,
        "title": "债券风险分析",
        "code": """import numpy as np
from scipy import stats

def calculate_bond_risk(price, duration, convexity, yield_change=0.01):
    duration_change = -duration * yield_change * price
    convexity_change = 0.5 * convexity * (yield_change ** 2) * price
    total_change = duration_change + convexity_change
    price_change_percent = (total_change / price) * 100

    return duration_change, convexity_change, total_change, price_change_percent

def calculate_credit_spread(rating):
    rating_spreads = {
        'AAA': 0.50, 'AA+': 0.75, 'AA': 1.00, 'AA-': 1.25,
        'A+': 1.50, 'A': 1.75, 'A-': 2.00, 'BBB+': 2.50,
        'BBB': 3.00, 'BBB-': 3.50, 'BB+': 4.50, 'BB': 5.50,
        'BB-': 6.50, 'B+': 7.50, 'B': 8.50, 'B-': 9.50,
        'CCC+': 10.50, 'CCC': 11.50, 'CCC-': 12.50
    }
    return rating_spreads.get(rating, 13.00)

def calculate_default_probability(rating):
    rating_default_rates = {
        'AAA': 0.00, 'AA+': 0.01, 'AA': 0.02, 'AA-': 0.03,
        'A+': 0.05, 'A': 0.07, 'A-': 0.10, 'BBB+': 0.15,
        'BBB': 0.20, 'BBB-': 0.25, 'BB+': 0.40, 'BB': 0.60,
        'BB-': 0.90, 'B+': 1.20, 'B': 1.60, 'B-': 2.10,
        'CCC+': 3.00, 'CCC': 4.00, 'CCC-': 5.00
    }
    return rating_default_rates.get(rating, 6.00)

def calculate_var(duration, convexity, yield_volatility, confidence_level=0.95, time_period=1):
    yield_change_std = yield_volatility * np.sqrt(time_period)
    z_score = stats.norm.ppf(1 - confidence_level)
    yield_change = z_score * yield_change_std

    duration_change = -duration * yield_change
    convexity_change = 0.5 * convexity * (yield_change ** 2)

    total_change = duration_change + convexity_change
    return total_change

def analyze_risk(bonds_data):
    bonds_data['Credit Spread'] = bonds_data['Rating'].apply(calculate_credit_spread)
    bonds_data['Default Probability'] = bonds_data['Rating'].apply(calculate_default_probability)
    bonds_data['Credit Risk Score'] = bonds_data['Default Probability'] * bonds_data['Duration']

    bonds_data['Price Change (%)'] = bonds_data.apply(
        lambda row: calculate_bond_risk(row['Price'], row['Duration'], row['Convexity'])[3],
        axis=1
    )

    return bonds_data

# 使用示例
# corporate_bonds = get_corporate_bond_data()
# processed_bonds = bond_data_preprocessing(corporate_bonds)
# risk_analysis = analyze_risk(processed_bonds)
# print(\"\\n风险分析结果:\\n\", risk_analysis[['Symbol', 'Name', 'Credit Spread', 'Default Probability', 'Credit Risk Score']])
#
# duration_change, convexity_change, total_change, price_change_percent = calculate_bond_risk(98.50, 8.50, 125.32, yield_change=0.01)
# print(f\"\\n价格变化分析:\\n久期效应: {duration_change:.2f}\\n凸性效应: {convexity_change:.2f}\\n总变化: {total_change:.2f}\\n变化百分比: {price_change_percent:.2f}%\")
""",
        "explanation": "此代码演示如何进行债券风险分析，包括信用利差、违约概率、VaR和价格变化敏感度等计算。债券风险分析对于投资者在购买债券前评估风险非常重要。"
    },
    {
        "topic_id": 6,
        "category_id": 6,
        "title": "债券投资策略",
        "code": """import pandas as pd

class BondPortfolio:
    def __init__(self, bonds_data):
        self.bonds_data = bonds_data
        self.positions = {}
        self.total_value = 0

    def add_bond(self, symbol, quantity):
        if symbol in self.bonds_data['Symbol'].values:
            bond_info = self.bonds_data[self.bonds_data['Symbol'] == symbol].iloc[0]
            self.positions[symbol] = {
                'Quantity': quantity,
                'Price': bond_info['Price'],
                'Coupon Rate (%)': bond_info['Coupon Rate (%)'],
                'Maturity Date': bond_info['Maturity Date'],
                'Duration': bond_info['Duration'],
                'Convexity': bond_info['Convexity']
            }
            self.update_total_value()
        else:
            print(f"债券 {symbol} 未找到")

    def remove_bond(self, symbol):
        if symbol in self.positions:
            del self.positions[symbol]
            self.update_total_value()
        else:
            print(f"债券 {symbol} 未在投资组合中找到")

    def update_total_value(self):
        total = 0

        for symbol, info in self.positions.items():
            total += info['Quantity'] * info['Price']

        self.total_value = total

    def get_portfolio_summary(self):
        if not self.positions:
            return pd.DataFrame()

        positions_data = []

        for symbol, info in self.positions.items():
            positions_data.append({
                'Symbol': symbol,
                'Quantity': info['Quantity'],
                'Price': info['Price'],
                'Value': info['Quantity'] * info['Price'],
                'Duration': info['Duration'],
                'Convexity': info['Convexity']
            })

        df = pd.DataFrame(positions_data)
        df['Weight'] = df['Value'] / df['Value'].sum()

        weighted_duration = (df['Duration'] * df['Weight']).sum()
        weighted_convexity = (df['Convexity'] * df['Weight']).sum()

        summary = {
            'Total Value': df['Value'].sum(),
            'Number of Bonds': len(self.positions),
            'Average Duration': weighted_duration,
            'Average Convexity': weighted_convexity
        }

        return df, summary

    def rebalance_portfolio(self, target_weights):
        if not self.positions:
            print("投资组合为空")
            return

        current_values = {symbol: info['Quantity'] * info['Price'] for symbol, info in self.positions.items()}
        total_value = sum(current_values.values())
        target_values = {symbol: total_value * weight for symbol, weight in target_weights.items()}

        for symbol, target_value in target_values.items():
            if symbol in self.positions:
                current_value = current_values[symbol]
                current_price = self.positions[symbol]['Price']
                target_quantity = int(target_value / current_price)
                self.positions[symbol]['Quantity'] = target_quantity

        self.update_total_value()

    def plot_duration_distribution(self):
        if not self.positions:
            print("投资组合为空")
            return

        positions_data = []

        for symbol, info in self.positions.items():
            positions_data.append({
                'Symbol': symbol,
                'Duration': info['Duration'],
                'Quantity': info['Quantity']
            })

        df = pd.DataFrame(positions_data)
        plt.figure(figsize=(12, 8))
        plt.hist(df['Duration'], bins=10, color='skyblue', edgecolor='black')
        plt.title('投资组合久期分布')
        plt.xlabel('久期')
        plt.ylabel('频率')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('duration_distribution.png')
        plt.show()

    def plot_coupon_distribution(self):
        if not self.positions:
            print("投资组合为空")
            return

        positions_data = []

        for symbol, info in self.positions.items():
            positions_data.append({
                'Symbol': symbol,
                'Coupon Rate': info['Coupon Rate (%)'],
                'Quantity': info['Quantity']
            })

        df = pd.DataFrame(positions_data)
        plt.figure(figsize=(12, 8))
        plt.hist(df['Coupon Rate'], bins=10, color='skyblue', edgecolor='black')
        plt.title('投资组合票面利率分布')
        plt.xlabel('票面利率 (%)')
        plt.ylabel('频率')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('coupon_distribution.png')
        plt.show()

# 使用示例
# portfolio = BondPortfolio(corporate_bonds)
# portfolio.add_bond('AAPL', 100)
# portfolio.add_bond('MSFT', 200)
# portfolio.add_bond('AMZN', 50)
# positions_df, summary = portfolio.get_portfolio_summary()
# print("投资组合摘要:\\n", positions_df)
# print("\\n投资组合统计:\\n", summary)
""",
        "explanation": "此代码演示如何创建一个债券投资组合类，用于管理债券持仓和分析投资组合风险。投资组合类提供了添加/移除债券、重新平衡投资组合和分析投资组合久期分布的功能。"
    },
    # 主题7：房地产投资分析
    {
        "topic_id": 7,
        "category_id": 1,
        "title": "房地产数据获取",
        "code": """import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

def get_house_listings(city, pages=1):
    \"\"\"获取房地产 listings 数据\"\"\"
    all_listings = []

    try:
        for page in range(1, pages + 1):
            # 这里使用模拟数据（实际应用中需要访问真实的房地产API）
            print(f"正在获取第{page}页数据...")

            # 创建模拟数据
            for i in range(10):
                listing = {
                    "id": f"{city}_{page}_{i}",
                    "title": f"{city}优质房源 {page}-{i}",
                    "price": 300000 + (page - 1) * 50000 + i * 10000,
                    "bedrooms": 2 + (i % 3),
                    "bathrooms": 1 + (i % 2),
                    "sqft": 800 + i * 100,
                    "address": f"{city}市朝阳区第{page}街道{i}号",
                    "listing_date": datetime(2024, 1, 1) + pd.Timedelta(days=(page - 1) * 30 + i)
                }
                all_listings.append(listing)

        df = pd.DataFrame(all_listings)
        return df

    except Exception as e:
        print(f"获取房地产数据失败: {str(e)}")
        return pd.DataFrame()

def get_property_details(property_id):
    \"\"\"获取单个房产详细信息\"\"\"
    try:
        # 模拟获取房产详细信息
        details = {
            "id": property_id,
            "property_type": "公寓",
            "year_built": 2015 + (int(property_id[-1]) % 5),
            "amenities": ["电梯", "停车位", "健身房", "游泳池"],
            "tax_assessment": 280000 + (int(property_id[-1]) % 10) * 5000,
            "last_sold_price": 295000 + (int(property_id[-1]) % 10) * 3000,
            "last_sold_date": datetime(2023, 1, 1) + pd.Timedelta(days=int(property_id[-1]) * 50)
        }

        return details

    except Exception as e:
        print(f"获取房产{property_id}详情失败: {str(e)}")
        return {}

def get_neighborhood_data(city):
    \"\"\"获取小区周边数据\"\"\"
    try:
        neighborhoods = [
            {
                "name": f"{city}小区A",
                "avg_price": 350000,
                "price_per_sqft": 350,
                "crime_rate": 0.02,
                "school_rating": 9,
                "walk_score": 85,
                "transit_score": 90
            },
            {
                "name": f"{city}小区B",
                "avg_price": 320000,
                "price_per_sqft": 320,
                "crime_rate": 0.03,
                "school_rating": 8,
                "walk_score": 80,
                "transit_score": 85
            },
            {
                "name": f"{city}小区C",
                "avg_price": 380000,
                "price_per_sqft": 380,
                "crime_rate": 0.01,
                "school_rating": 10,
                "walk_score": 90,
                "transit_score": 95
            }
        ]

        return pd.DataFrame(neighborhoods)

    except Exception as e:
        print(f"获取{city}小区数据失败: {str(e)}")
        return pd.DataFrame()

# 使用示例
city = "北京"
house_listings = get_house_listings(city, pages=2)
print(f"{city}房产列表数据形状: {house_listings.shape}")
print(house_listings[['title', 'price', 'bedrooms', 'bathrooms']].head())

# 获取单个房产详细信息
property_details = get_property_details("北京_1_0")
print("\\n房产详细信息:")
for key, value in property_details.items():
    print(f"{key}: {value}")

# 获取小区数据
neighborhood_data = get_neighborhood_data(city)
print("\\n小区数据:")
print(neighborhood_data[['name', 'avg_price', 'school_rating']])
""",
        "explanation": "此代码演示如何获取房地产市场数据，包括房产列表、单个房产详细信息和小区周边数据。由于访问真实房地产API可能需要权限，这里使用模拟数据演示获取方法。"
    },
    {
        "topic_id": 7,
        "category_id": 2,
        "title": "房地产数据预处理",
        "code": """import pandas as pd
import numpy as np
from datetime import datetime

def clean_house_data(df):
    \"\"\"清洁房产数据\"\"\"
    df_clean = df.copy()

    # 删除重复项
    df_clean = df_clean.drop_duplicates()

    # 处理缺失值
    df_clean['price'] = df_clean['price'].fillna(df_clean['price'].median())
    df_clean['sqft'] = df_clean['sqft'].fillna(df_clean['sqft'].mean())

    # 数据转换
    df_clean['listing_month'] = df_clean['listing_date'].dt.month
    df_clean['listing_year'] = df_clean['listing_date'].dt.year

    # 计算价格每平方英尺
    df_clean['price_per_sqft'] = df_clean['price'] / df_clean['sqft']

    return df_clean

def normalize_property_features(df):
    \"\"\"归一化房产特征\"\"\"
    df_normalized = df.copy()

    # 归一化数值特征
    numerical_features = ['price', 'sqft', 'price_per_sqft']

    for feature in numerical_features:
        df_normalized[feature] = (df_normalized[feature] - df_normalized[feature].min()) / \
                               (df_normalized[feature].max() - df_normalized[feature].min())

    return df_normalized

def calculate_price_statistics(df):
    \"\"\"计算价格统计信息\"\"\"
    price_stats = {
        "mean_price": df['price'].mean(),
        "median_price": df['price'].median(),
        "min_price": df['price'].min(),
        "max_price": df['price'].max(),
        "price_std": df['price'].std(),
        "count": len(df)
    }

    return price_stats

# 使用示例（假设已获取数据）
# house_listings = get_house_listings("北京")
# cleaned_data = clean_house_data(house_listings)
# normalized_data = normalize_property_features(cleaned_data)
# price_stats = calculate_price_statistics(cleaned_data)
#
# print("价格统计信息:")
# for key, value in price_stats.items():
#     if key != 'count':
#         print(f"{key}: ${value:.2f}")
#     else:
#         print(f"{key}: {value}")
#
# print("\\n归一化后的数据:")
# print(normalized_data[['title', 'price', 'sqft', 'price_per_sqft']].head())
""",
        "explanation": "此代码演示如何对房地产数据进行预处理，包括数据清洗、缺失值处理、特征工程和归一化。良好的数据预处理是进行房地产分析的基础。"
    },
    {
        "topic_id": 7,
        "category_id": 3,
        "title": "房地产估值模型",
        "code": """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def prepare_features_for_valuation(df):
    \"\"\"准备估值模型特征\"\"\"
    # 选择相关特征
    features = df[['bedrooms', 'bathrooms', 'sqft', 'listing_month', 'listing_year']]

    return features

def train_valuation_model(X, y, model_type='linear'):
    \"\"\"训练估值模型\"\"\"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("不支持的模型类型")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mse, r2

def predict_property_value(model, features):
    \"\"\"预测房产价值\"\"\"
    if isinstance(features, pd.DataFrame):
        return model.predict(features)
    elif isinstance(features, list):
        return model.predict([features])
    else:
        return model.predict([[features]])

def calculate_feature_importance(model, feature_names):
    \"\"\"计算特征重要性\"\"\"
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False)

        return importance
    elif hasattr(model, 'coef_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': abs(model.coef_)
        }).sort_values(by='importance', ascending=False)

        return importance
    else:
        return pd.DataFrame()

# 使用示例（假设已处理数据）
# X = prepare_features_for_valuation(cleaned_data)
# y = cleaned_data['price']
#
# linear_model, linear_mse, linear_r2 = train_valuation_model(X, y, 'linear')
# rf_model, rf_mse, rf_r2 = train_valuation_model(X, y, 'random_forest')
#
# print(f"线性回归 - MSE: {linear_mse:.2f}, R²: {linear_r2:.4f}")
# print(f"随机森林 - MSE: {rf_mse:.2f}, R²: {rf_r2:.4f}")
#
# # 预测示例
# example_features = [[3, 2, 1500, 6, 2024]]
# prediction = predict_property_value(rf_model, example_features)
# print(f"预测价格: ${prediction[0]:,.2f}")
#
# # 特征重要性
# importance = calculate_feature_importance(rf_model, X.columns)
# print("\\n特征重要性:")
# print(importance)
""",
        "explanation": "此代码演示如何构建和训练房地产估值模型，包括特征准备、模型训练、预测和特征重要性分析。机器学习模型可以帮助更准确地评估房产价值。"
    },
    {
        "topic_id": 7,
        "category_id": 4,
        "title": "房地产数据可视化",
        "code": """import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_price_distribution(df):
    \"\"\"绘制价格分布直方图\"\"\"
    plt.figure(figsize=(12, 6))
    sns.histplot(df['price'], kde=True, bins=30)
    plt.title('房产价格分布')
    plt.xlabel('价格 ($)')
    plt.ylabel('数量')
    plt.grid(True)
    plt.savefig('price_distribution.png', dpi=300)
    plt.show()

def plot_price_vs_sqft(df):
    \"\"\"绘制价格与面积关系\"\"\"
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='sqft', y='price')
    sns.regplot(data=df, x='sqft', y='price', scatter=False, color='red')
    plt.title('价格与面积关系')
    plt.xlabel('面积 (平方英尺)')
    plt.ylabel('价格 ($)')
    plt.grid(True)
    plt.savefig('price_vs_sqft.png', dpi=300)
    plt.show()

def plot_price_per_bedroom(df):
    \"\"\"绘制价格按卧室数量分组\"\"\"
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='bedrooms', y='price')
    plt.title('价格按卧室数量分组')
    plt.xlabel('卧室数量')
    plt.ylabel('价格 ($)')
    plt.grid(True)
    plt.savefig('price_per_bedroom.png', dpi=300)
    plt.show()

def plot_monthly_listings(df):
    \"\"\"绘制月度房源数量\"\"\"
    monthly_counts = df.groupby(['listing_year', 'listing_month']).size().unstack(fill_value=0)

    plt.figure(figsize=(12, 6))
    monthly_counts.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('月度房源数量')
    plt.xlabel('年份')
    plt.ylabel('房源数量')
    plt.legend(title='月份')
    plt.grid(True)
    plt.savefig('monthly_listings.png', dpi=300)
    plt.show()

def plot_correlation_matrix(df):
    \"\"\"绘制相关系数矩阵\"\"\"
    numerical_cols = ['price', 'bedrooms', 'bathrooms', 'sqft']

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('特征相关系数矩阵')
    plt.savefig('correlation_matrix.png', dpi=300)
    plt.show()

# 使用示例（假设已处理数据）
# plot_price_distribution(cleaned_data)
# plot_price_vs_sqft(cleaned_data)
# plot_price_per_bedroom(cleaned_data)
# plot_monthly_listings(cleaned_data)
# plot_correlation_matrix(cleaned_data)
""",
        "explanation": "此代码演示如何可视化房地产数据，包括价格分布、价格与面积关系、价格按卧室数量分组等图表。可视化帮助更好地理解房地产市场特征。"
    },
    {
        "topic_id": 7,
        "category_id": 5,
        "title": "房地产投资回报计算",
        "code": """import pandas as pd
import numpy as np
from datetime import datetime

def calculate_roi(purchase_price, sale_price, holding_period, down_payment, monthly_rent, expenses):
    \"\"\"计算投资回报率\"\"\"
    initial_investment = purchase_price * down_payment

    total_rental_income = monthly_rent * 12 * holding_period

    total_expenses = expenses * 12 * holding_period

    net_income = total_rental_income - total_expenses

    capital_gain = sale_price - purchase_price

    total_profit = net_income + capital_gain

    roi = total_profit / initial_investment

    annual_roi = (1 + roi) ** (1 / holding_period) - 1

    return roi, annual_roi

def calculate_cash_flow(purchase_price, down_payment, monthly_rent, monthly_expenses):
    \"\"\"计算现金流\"\"\"
    monthly_interest = (purchase_price * (1 - down_payment)) * (0.04 / 12)

    monthly_cash_flow = monthly_rent - monthly_expenses - monthly_interest

    cash_on_cash = (monthly_cash_flow * 12) / (purchase_price * down_payment)

    return monthly_cash_flow, cash_on_cash

def calculate_cap_rate(net_operating_income, property_value):
    \"\"\"计算资本化率\"\"\"
    if property_value > 0:
        return net_operating_income / property_value
    else:
        return 0

def analyze_investment_scenarios(purchase_price):
    \"\"\"分析不同投资场景\"\"\"
    scenarios = [
        {
            "name": "保守场景",
            "rent_growth": 0.02,
            "appreciation": 0.03,
            "vacancy_rate": 0.05,
            "expense_ratio": 0.35
        },
        {
            "name": "基准场景",
            "rent_growth": 0.03,
            "appreciation": 0.05,
            "vacancy_rate": 0.03,
            "expense_ratio": 0.30
        },
        {
            "name": "乐观场景",
            "rent_growth": 0.05,
            "appreciation": 0.08,
            "vacancy_rate": 0.02,
            "expense_ratio": 0.25
        }
    ]

    results = []

    for scenario in scenarios:
        monthly_rent = (purchase_price / 1000) * 0.8
        expenses = monthly_rent * scenario['expense_ratio']

        total_rent = monthly_rent * 12 * 5
        total_expenses = expenses * 12 * 5

        net_operating_income = total_rent - total_expenses

        future_value = purchase_price * (1 + scenario['appreciation']) ** 5

        roi, annual_roi = calculate_roi(purchase_price, future_value, 5, 0.2, monthly_rent, expenses)

        results.append({
            "scenario": scenario['name'],
            "total_rent": total_rent,
            "total_expenses": total_expenses,
            "net_operating_income": net_operating_income,
            "future_value": future_value,
            "roi": roi,
            "annual_roi": annual_roi
        })

    return results

# 使用示例
purchase_price = 350000
down_payment = 0.2
monthly_rent = 2500
monthly_expenses = 800

roi, annual_roi = calculate_roi(purchase_price, 420000, 5, down_payment, monthly_rent, monthly_expenses)
cash_flow, cash_on_cash = calculate_cash_flow(purchase_price, down_payment, monthly_rent, monthly_expenses)

print(f"投资回报率 (ROI): {roi:.2%}")
print(f"年化投资回报率: {annual_roi:.2%}")
print(f"月现金流: ${cash_flow:.2f}")
print(f"现金回报率: {cash_on_cash:.2%}")

scenario_results = analyze_investment_scenarios(purchase_price)
print("\\n投资场景分析:")
for result in scenario_results:
    print(f"\\n{result['scenario']}:")
    print(f"  总租金: ${result['total_rent']:.0f}")
    print(f"  总支出: ${result['total_expenses']:.0f}")
    print(f"  净运营收入: ${result['net_operating_income']:.0f}")
    print(f"  未来价值: ${result['future_value']:.0f}")
    print(f"  投资回报率: {result['roi']:.2%}")
    print(f"  年化回报率: {result['annual_roi']:.2%}")
""",
        "explanation": "此代码演示如何计算房地产投资回报率，包括ROI、年化ROI、现金流、资本化率等指标，并提供不同投资场景的分析。这些指标帮助评估投资物业的潜力。"
    },
    {
        "topic_id": 7,
        "category_id": 6,
        "title": "完整房地产投资分析系统",
        "code": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RealEstateInvestmentSystem:
    def __init__(self, city):
        self.city = city
        self.listings = None
        self.analytics = None

    def load_data(self, pages=1):
        \"\"\"加载数据\"\"\"
        from data_fetcher import get_house_listings

        self.listings = get_house_listings(self.city, pages)

        return self.listings

    def preprocess_data(self):
        \"\"\"预处理数据\"\"\"
        from data_preprocessor import clean_house_data, normalize_property_features

        if self.listings is None:
            raise ValueError("请先加载数据")

        self.analytics = clean_house_data(self.listings)

        return self.analytics

    def analyze_market(self):
        \"\"\"市场分析\"\"\"
        from market_analyzer import (
            calculate_price_statistics,
            prepare_features_for_valuation,
            train_valuation_model,
            calculate_feature_importance
        )

        if self.analytics is None:
            raise ValueError("请先预处理数据")

        price_stats = calculate_price_statistics(self.analytics)

        X = prepare_features_for_valuation(self.analytics)
        y = self.analytics['price']

        model, mse, r2 = train_valuation_model(X, y, 'linear')

        feature_importance = calculate_feature_importance(model, X.columns)

        price_range = {
            "low": price_stats['mean_price'] - price_stats['price_std'],
            "high": price_stats['mean_price'] + price_stats['price_std']
        }

        return {
            "price_stats": price_stats,
            "price_range": price_range,
            "model_performance": {"mse": mse, "r2": r2},
            "feature_importance": feature_importance
        }

    def visualize_data(self):
        \"\"\"可视化数据\"\"\"
        from visualizer import (
            plot_price_distribution,
            plot_price_vs_sqft,
            plot_price_per_bedroom,
            plot_monthly_listings,
            plot_correlation_matrix
        )

        if self.analytics is None:
            raise ValueError("请先预处理数据")

        plot_price_distribution(self.analytics)
        plot_price_vs_sqft(self.analytics)
        plot_price_per_bedroom(self.analytics)
        plot_monthly_listings(self.analytics)
        plot_correlation_matrix(self.analytics)

    def analyze_investment(self, property_details):
        \"\"\"投资分析\"\"\"
        from investment_analyzer import (
            calculate_roi,
            calculate_cash_flow,
            calculate_cap_rate,
            analyze_investment_scenarios
        )

        if self.analytics is None:
            raise ValueError("请先预处理数据")

        purchase_price = property_details['price']
        monthly_rent = (purchase_price / 1000) * 0.8
        monthly_expenses = monthly_rent * 0.3

        roi, annual_roi = calculate_roi(purchase_price, 420000, 5, 0.2, monthly_rent, monthly_expenses)
        cash_flow, cash_on_cash = calculate_cash_flow(purchase_price, 0.2, monthly_rent, monthly_expenses)
        cap_rate = calculate_cap_rate((monthly_rent - monthly_expenses) * 12, purchase_price)

        return {
            "roi": roi,
            "annual_roi": annual_roi,
            "cash_flow": cash_flow,
            "cash_on_cash": cash_on_cash,
            "cap_rate": cap_rate
        }

    def generate_report(self):
        \"\"\"生成分析报告\"\"\"
        market_analysis = self.analyze_market()

        report = []

        report.append(f"# {self.city}房地产市场分析报告")
        report.append(f"生成时间: {pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
        report.append("")
        report.append("## 市场概览")
        report.append(f"房源数量: {market_analysis['price_stats']['count']}")
        report.append(f"平均价格: ${market_analysis['price_stats']['mean_price']:.0f}")
        report.append(f"中位价格: ${market_analysis['price_stats']['median_price']:.0f}")
        report.append(f"价格标准差: ${market_analysis['price_stats']['price_std']:.0f}")
        report.append(f"价格范围: ${market_analysis['price_range']['low']:.0f} - ${market_analysis['price_range']['high']:.0f}")
        report.append("")
        report.append("## 模型性能")
        report.append(f"均方误差 (MSE): {market_analysis['model_performance']['mse']:.2f}")
        report.append(f"决定系数 (R²): {market_analysis['model_performance']['r2']:.4f}")
        report.append("")
        report.append("## 特征重要性")

        for index, row in market_analysis['feature_importance'].iterrows():
            report.append(f"- {row['feature']}: {row['importance']:.4f}")

        return "\\n".join(report)

# 使用示例
if __name__ == "__main__":
    system = RealEstateInvestmentSystem("北京")
    print("1. 加载数据")
    system.load_data(pages=2)

    print("2. 预处理数据")
    system.preprocess_data()

    print("3. 市场分析")
    market_analysis = system.analyze_market()

    print("4. 可视化数据")
    system.visualize_data()

    print("5. 投资分析")
    sample_property = {
        "id": "sample_1",
        "price": 350000,
        "bedrooms": 3,
        "bathrooms": 2,
        "sqft": 1200
    }
    investment_analysis = system.analyze_investment(sample_property)

    print("6. 生成报告")
    report = system.generate_report()
    with open("real_estate_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\\n分析完成！报告已保存到 real_estate_report.md")
""",
        "explanation": "这是一个完整的房地产投资分析系统类，集成了数据获取、预处理、市场分析、投资分析和可视化功能。系统提供了全面的市场分析和投资评估方法。"
    },
    {
        "topic_id": 7,
        "category_id": 6,
        "title": "实战案例 - 房地产市场分析报告",
        "code": """import pandas as pd
from real_estate_system import RealEstateInvestmentSystem

def run_case_study():
    \"\"\"房地产市场分析实战案例\"\"\"
    system = RealEstateInvestmentSystem("北京")

    try:
        print("1. 加载和预处理数据")
        system.load_data(pages=2)
        system.preprocess_data()

        print("2. 市场分析")
        analysis = system.analyze_market()

        print("\\n=== 市场概览 ===")
        print(f"房源数量: {analysis['price_stats']['count']}")
        print(f"平均价格: ${analysis['price_stats']['mean_price']:.0f}")
        print(f"中位价格: ${analysis['price_stats']['median_price']:.0f}")
        print(f"价格范围: ${analysis['price_range']['low']:.0f} - ${analysis['price_range']['high']:.0f}")

        print("\\n=== 模型性能 ===")
        print(f"均方误差 (MSE): {analysis['model_performance']['mse']:.2f}")
        print(f"决定系数 (R²): {analysis['model_performance']['r2']:.4f}")

        print("\\n=== 特征重要性 ===")
        for index, row in analysis['feature_importance'].iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")

        print("\\n3. 可视化")
        system.visualize_data()

        print("\\n4. 投资分析")
        sample_property = {
            "id": "case_01",
            "price": 350000,
            "bedrooms": 3,
            "bathrooms": 2,
            "sqft": 1200
        }
        investment = system.analyze_investment(sample_property)

        print("\\n=== 投资分析 ===")
        print(f"投资回报率: {investment['roi']:.2%}")
        print(f"年化投资回报率: {investment['annual_roi']:.2%}")
        print(f"月现金流: ${investment['cash_flow']:.2f}")
        print(f"现金回报率: {investment['cash_on_cash']:.2%}")
        print(f"资本化率: {investment['cap_rate']:.2%}")

        print("\\n5. 生成报告")
        report = system.generate_report()
        with open("beijing_market_analysis.md", "w", encoding="utf-8") as f:
            f.write(report)

        print("\\n=== 报告生成完成 ===")
        print("报告已保存到 beijing_market_analysis.md")

        return True

    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== 房地产市场分析实战案例 ===")

    success = run_case_study()

    if success:
        print("\\n✅ 分析完成！")
    else:
        print("\\n❌ 分析过程中出现错误")
""",
        "explanation": "这是一个完整的房地产市场分析实战案例，展示了从数据获取、预处理、市场分析到报告生成的完整流程。实战案例帮助理解如何在实际应用中使用这些功能。"
    },
    # 主题8：金融风险管理
    {
        "topic_id": 8,
        "category_id": 1,
        "title": "风险数据获取与处理",
        "code": """import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def fetch_risk_data(symbols, start_date, end_date):
    \"\"\"获取风险数据\"\"\"
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            history = ticker.history(start=start_date, end=end_date)
            data[symbol] = history
            print(f"成功获取{symbol}数据")
        except Exception as e:
            print(f"获取{symbol}数据失败: {e}")
            data[symbol] = pd.DataFrame()
    return data

def calculate_returns(data):
    \"\"\"计算收益率\"\"\"
    returns = {}
    for symbol, df in data.items():
        if not df.empty:
            returns[symbol] = df['Close'].pct_change().dropna()
    return returns

def calculate_volatility(returns, window=252):
    \"\"\"计算波动率\"\"\"
    volatility = {}
    for symbol, ret in returns.items():
        volatility[symbol] = ret.std() * np.sqrt(window)
    return volatility

def calculate_value_at_risk(returns, confidence_level=0.95):
    \"\"\"计算风险价值(VaR)\"\"\"
    VaR = {}
    for symbol, ret in returns.items():
        VaR[symbol] = np.percentile(ret, 100 * (1 - confidence_level))
    return VaR

def calculate_expected_shortfall(returns, confidence_level=0.95):
    \"\"\"计算预期损失(ES)\"\"\"
    ES = {}
    for symbol, ret in returns.items():
        VaR = np.percentile(ret, 100 * (1 - confidence_level))
        ES[symbol] = ret[ret <= VaR].mean()
    return ES

def prepare_risk_report(data, returns, volatility, VaR, ES):
    \"\"\"准备风险报告\"\"\"
    report = []

    for symbol in data.keys():
        if not data[symbol].empty:
            report.append({
                'Symbol': symbol,
                'Start Date': data[symbol].index[0].strftime('%Y-%m-%d'),
                'End Date': data[symbol].index[-1].strftime('%Y-%m-%d'),
                'Daily Return': returns[symbol].mean() if symbol in returns else np.nan,
                'Volatility': volatility[symbol] if symbol in volatility else np.nan,
                'VaR (95%)': VaR[symbol] if symbol in VaR else np.nan,
                'Expected Shortfall (95%)': ES[symbol] if symbol in ES else np.nan
            })

    return pd.DataFrame(report)

# 使用示例
# symbols = ['SPY', 'AAPL', 'MSFT']
# start_date = '2020-01-01'
# end_date = '2023-12-31'
#
# data = fetch_risk_data(symbols, start_date, end_date)
# returns = calculate_returns(data)
# volatility = calculate_volatility(returns)
# VaR = calculate_value_at_risk(returns)
# ES = calculate_expected_shortfall(returns)
#
# report = prepare_risk_report(data, returns, volatility, VaR, ES)
# print(report)
""",
        "explanation": "此代码演示如何从Yahoo Finance获取风险数据，并计算关键风险指标：收益率、波动率、VaR（风险价值）和预期损失(ES)。这些指标是金融风险管理的基础。"
    },
    {
        "topic_id": 8,
        "category_id": 2,
        "title": "风险识别与评估",
        "code": """import pandas as pd
import numpy as np
from scipy.stats import norm

def identify_market_risk_factors(data):
    \"\"\"识别市场风险因素\"\"\"
    risk_factors = []

    # 计算价格变化
    for symbol, df in data.items():
        if not df.empty:
            df['Price Change'] = df['Close'].pct_change()
            df['Volatility'] = df['Price Change'].rolling(window=30).std() * np.sqrt(252)

            # 检测价格异常波动
            threshold = df['Price Change'].std() * 3
            outliers = df[np.abs(df['Price Change']) > threshold]

            if not outliers.empty:
                risk_factors.extend([
                    {
                        'Symbol': symbol,
                        'Date': idx.strftime('%Y-%m-%d'),
                        'Type': 'Price Volatility',
                        'Magnitude': abs(change),
                        'Volatility': vol
                    } for idx, change, vol in zip(outliers.index, outliers['Price Change'], outliers['Volatility'])
                ])

    return pd.DataFrame(risk_factors)

def assess_credit_risk(rating, market_value, exposure):
    \"\"\"评估信用风险\"\"\"
    # 简化的信用风险评估模型
    default_probabilities = {
        'AAA': 0.001, 'AA': 0.002, 'A': 0.005, 'BBB': 0.01,
        'BB': 0.03, 'B': 0.08, 'CCC': 0.20, 'D': 1.00
    }

    recovery_rates = {
        'AAA': 0.90, 'AA': 0.85, 'A': 0.80, 'BBB': 0.70,
        'BB': 0.60, 'B': 0.45, 'CCC': 0.30, 'D': 0.10
    }

    default_prob = default_probabilities.get(rating, 0.25)
    recovery_rate = recovery_rates.get(rating, 0.40)

    expected_loss = default_prob * (1 - recovery_rate) * exposure

    return {
        'Default Probability': default_prob,
        'Recovery Rate': recovery_rate,
        'Expected Loss': expected_loss
    }

def evaluate_operational_risk(incidents, business_units):
    \"\"\"评估操作风险\"\"\"
    risk_scores = {}

    for unit in business_units:
        unit_incidents = incidents[incidents['Business Unit'] == unit]

        if not unit_incidents.empty:
            # 简化的操作风险评分
            frequency_score = len(unit_incidents)
            severity_score = unit_incidents['Severity'].mean()
            risk_scores[unit] = frequency_score * severity_score

    return risk_scores

def calculate_portfolio_risk_exposure(weights, cov_matrix):
    \"\"\"计算投资组合风险暴露\"\"\"
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_volatility

def generate_risk_identification_report(risk_factors, credit_risk, operational_risk):
    \"\"\"生成风险识别报告\"\"\"
    report = {
        'Market Risk Factors': risk_factors,
        'Credit Risk Assessment': pd.DataFrame([credit_risk]),
        'Operational Risk Scores': pd.DataFrame(list(operational_risk.items()),
                                              columns=['Business Unit', 'Risk Score'])
    }

    return report

# 使用示例
# incidents = pd.DataFrame({
#     'Business Unit': ['Trading', 'Operations', 'Compliance', 'Trading'],
#     'Incident Date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05'],
#     'Severity': [3, 5, 2, 4],
#     'Description': ['系统故障', '人为错误', '合规违规', '市场波动']
# })
#
# business_units = ['Trading', 'Operations', 'Compliance']
#
# risk_factors = identify_market_risk_factors(data)
# credit_risk = assess_credit_risk('BBB', 1000000, 500000)
# operational_risk = evaluate_operational_risk(incidents, business_units)
#
# report = generate_risk_identification_report(risk_factors, credit_risk, operational_risk)
# print("市场风险因素数量:", len(report['Market Risk Factors']))
# print("信用风险评估:")
# print(report['Credit Risk Assessment'])
""",
        "explanation": "此代码演示风险识别与评估方法，包括市场风险因素识别、信用风险评估和操作风险评估，为风险管理提供基础。"
    },
    {
        "topic_id": 8,
        "category_id": 3,
        "title": "风险测量与量化",
        "code": """import pandas as pd
import numpy as np
from scipy.stats import norm, t
from scipy.optimize import minimize

def var_historical(returns, confidence_level=0.95):
    \"\"\"历史模拟法计算VaR\"\"\"
    return np.percentile(returns, 100 * (1 - confidence_level))

def var_parametric(returns, confidence_level=0.95, distribution='normal'):
    \"\"\"参数法计算VaR\"\"\"
    mean = returns.mean()
    std = returns.std()

    if distribution == 'normal':
        VaR = mean - std * norm.ppf(confidence_level)
    elif distribution == 't':
        # 假设自由度为4的t分布
        VaR = mean - std * t.ppf(confidence_level, df=4)
    else:
        raise ValueError("不支持的分布类型")

    return VaR

def var_monte_carlo(returns, confidence_level=0.95, simulations=10000):
    \"\"\"蒙特卡洛模拟计算VaR\"\"\"
    np.random.seed(42)
    mean = returns.mean()
    std = returns.std()

    simulated_returns = np.random.normal(mean, std, simulations)

    return np.percentile(simulated_returns, 100 * (1 - confidence_level))

def calculate_covar(returns, confidence_level=0.95):
    \"\"\"计算条件VaR(CVaR)\"\"\"
    VaR = np.percentile(returns, 100 * (1 - confidence_level))
    return returns[returns <= VaR].mean()

def portfolio_optimization(returns, risk_free_rate=0.02):
    \"\"\"投资组合优化\"\"\"
    # 计算协方差矩阵
    cov_matrix = returns.cov()

    # 定义目标函数（最小化波动率）
    def minimize_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # 定义约束条件
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, 1) for _ in range(len(returns.columns))]

    # 初始猜测
    initial_weights = np.ones(len(returns.columns)) / len(returns.columns)

    # 优化
    result = minimize(minimize_volatility, initial_weights,
                     method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

def sensitivity_analysis(parameters, base_value, percentage_change):
    \"\"\"敏感性分析\"\"\"
    results = {}

    for param, value in parameters.items():
        # 计算参数上下波动对结果的影响
        for direction in ['up', 'down']:
            if direction == 'up':
                new_value = value * (1 + percentage_change)
                scenario = f"{param}_up"
            else:
                new_value = value * (1 - percentage_change)
                scenario = f"{param}_down"

            # 简化的计算逻辑
            sensitivity = (new_value - value) / value
            results[scenario] = {
                'New Value': new_value,
                'Change': direction,
                'Sensitivity': sensitivity
            }

    return results

# 使用示例
# symbols = ['AAPL', 'MSFT', 'SPY']
# start_date = '2020-01-01'
# end_date = '2023-12-31'
#
# data = fetch_risk_data(symbols, start_date, end_date)
# returns = calculate_returns(data)
# aapl_returns = returns['AAPL']
#
# # 计算VaR使用不同方法
# historical_var = var_historical(aapl_returns)
# parametric_var = var_parametric(aapl_returns)
# monte_carlo_var = var_monte_carlo(aapl_returns)
# cvar = calculate_covar(aapl_returns)
#
# print(f"历史VaR: {historical_var:.4f}")
# print(f"参数VaR: {parametric_var:.4f}")
# print(f"蒙特卡洛VaR: {monte_carlo_var:.4f}")
# print(f"CVaR: {cvar:.4f}")
""",
        "explanation": "此代码演示风险测量与量化方法，包括多种VaR计算方法、CVaR计算、投资组合优化和敏感性分析，为风险决策提供数据支持。"
    },
    {
        "topic_id": 8,
        "category_id": 4,
        "title": "风险控制与管理",
        "code": """import pandas as pd
import numpy as np
from datetime import datetime

class RiskController:
    \"\"\"风险控制器\"\"\"

    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.risk_limits = {
            'total_exposure': 1000000,
            'single_position': 200000,
            'max_drawdown': 0.20,
            'var_95': -0.03
        }

    def check_position_limits(self):
        \"\"\"检查头寸限制\"\"\"
        violations = []

        total_value = self.portfolio['Value'].sum()
        if total_value > self.risk_limits['total_exposure']:
            violations.append(f"总头寸暴露超过限制: ${total_value:.0f} > ${self.risk_limits['total_exposure']:.0f}")

        for idx, position in self.portfolio.iterrows():
            if position['Value'] > self.risk_limits['single_position']:
                violations.append(f"{position['Symbol']}头寸过大: ${position['Value']:.0f} > ${self.risk_limits['single_position']:.0f}")

        return violations

    def check_var_limit(self, VaR):
        \"\"\"检查VaR限制\"\"\"
        violations = []

        if VaR < self.risk_limits['var_95']:
            violations.append(f"VaR超过限制: {VaR:.4f} < {self.risk_limits['var_95']:.4f}")

        return violations

    def calculate_stop_loss(self, entry_price, stop_loss_pct=0.05):
        \"\"\"计算止损价格\"\"\"
        return entry_price * (1 - stop_loss_pct)

    def calculate_take_profit(self, entry_price, take_profit_pct=0.10):
        \"\"\"计算止盈价格\"\"\"
        return entry_price * (1 + take_profit_pct)

    def optimize_portfolio_hedging(self, hedge_instruments, correlation_matrix):
        \"\"\"优化投资组合对冲\"\"\"
        optimal_hedge_ratios = {}

        # 简化的对冲优化逻辑
        for instrument, correlations in correlation_matrix.items():
            if instrument in hedge_instruments:
                optimal_hedge_ratios[instrument] = -correlations['Portfolio']

        return optimal_hedge_ratios

    def stress_test_portfolio(self, scenarios):
        \"\"\"压力测试投资组合\"\"\"
        results = []

        for scenario_name, impact in scenarios.items():
            scenario_value = self.portfolio['Value'].sum() * (1 + impact)
            results.append({
                'Scenario': scenario_name,
                'Impact': impact,
                'Portfolio Value': scenario_value,
                'Loss': scenario_value - self.portfolio['Value'].sum()
            })

        return pd.DataFrame(results)

def implement_risk_monitoring_system(data_source, monitoring_rules):
    \"\"\"实现风险监控系统\"\"\"
    violations = []

    for rule in monitoring_rules:
        data = fetch_risk_data([rule['Symbol']], rule['Start'], rule['End'])
        returns = calculate_returns(data)

        if rule['Symbol'] in returns:
            metric_value = var_historical(returns[rule['Symbol']], rule['Confidence'])

            if metric_value < rule['Threshold']:
                violations.append({
                    'Rule': rule['Name'],
                    'Symbol': rule['Symbol'],
                    'Metric': 'VaR',
                    'Value': metric_value,
                    'Threshold': rule['Threshold'],
                    'Violation': 'Below Threshold'
                })

    return pd.DataFrame(violations)

def manage_counterparty_risk(exposures, ratings, limits):
    \"\"\"管理对手方风险\"\"\"
    violations = []

    for counterparty, exposure in exposures.items():
        if counterparty in ratings:
            limit = limits.get(counterparty, 100000)

            if exposure > limit:
                violations.append({
                    'Counterparty': counterparty,
                    'Rating': ratings[counterparty],
                    'Exposure': exposure,
                    'Limit': limit,
                    'Overrun': exposure - limit
                })

    return pd.DataFrame(violations)

# 使用示例
# portfolio = pd.DataFrame({
#     'Symbol': ['AAPL', 'MSFT', 'SPY'],
#     'Quantity': [100, 50, 20],
#     'Price': [180, 400, 450],
#     'Value': [18000, 20000, 9000]
# })
#
# controller = RiskController(portfolio)
# violations = controller.check_position_limits()
# print("头寸限制违规:")
# for violation in violations:
#     print(violation)
""",
        "explanation": "此代码演示风险控制与管理方法，包括风险控制器类、风险监控系统和对手方风险管理，为风险控制提供完整框架。"
    },
    {
        "topic_id": 8,
        "category_id": 5,
        "title": "风险管理策略",
        "code": """import pandas as pd
import numpy as np
from datetime import datetime

class RiskManagementStrategy:
    \"\"\"风险管理策略基类\"\"\"

    def __init__(self):
        self.risk_tolerance = 'medium'
        self.strategy_name = 'Base Strategy'

    def evaluate_strategy(self, market_conditions):
        \"\"\"评估策略\"\"\"
        raise NotImplementedError("子类必须实现此方法")

    def rebalance(self, portfolio):
        \"\"\"再平衡投资组合\"\"\"
        raise NotImplementedError("子类必须实现此方法")

class DiversificationStrategy(RiskManagementStrategy):
    \"\"\"分散化策略\"\"\"

    def __init__(self, max_sector_weight=0.3):
        super().__init__()
        self.strategy_name = 'Diversification'
        self.max_sector_weight = max_sector_weight

    def evaluate_strategy(self, market_conditions):
        \"\"\"评估策略\"\"\"
        if market_conditions['Volatility'] > 0.25:
            return 'Increase Diversification'
        elif market_conditions['Volatility'] < 0.15:
            return 'Maintain Diversification'
        else:
            return 'Monitor Closely'

    def rebalance(self, portfolio):
        \"\"\"再平衡投资组合\"\"\"
        sectors = portfolio.groupby('Sector').sum()['Value']
        total_value = portfolio['Value'].sum()

        sector_weights = sectors / total_value

        adjustments = []

        for sector, weight in sector_weights.items():
            if weight > self.max_sector_weight:
                excess = weight - self.max_sector_weight
                target_value = sectors[sector] - (total_value * excess)
                adjustments.append({
                    'Sector': sector,
                    'Current Weight': weight,
                    'Target Weight': self.max_sector_weight,
                    'Adjustment': -total_value * excess
                })

        return pd.DataFrame(adjustments)

class HedgingStrategy(RiskManagementStrategy):
    \"\"\"对冲策略\"\"\"

    def __init__(self, hedge_ratio=0.5):
        super().__init__()
        self.strategy_name = 'Hedging'
        self.hedge_ratio = hedge_ratio

    def evaluate_strategy(self, market_conditions):
        \"\"\"评估策略\"\"\"
        if market_conditions['Trend'] == 'Down':
            return 'Increase Hedge Ratio'
        elif market_conditions['Trend'] == 'Up':
            return 'Decrease Hedge Ratio'
        else:
            return 'Maintain Current Hedge'

    def rebalance(self, portfolio):
        \"\"\"再平衡投资组合\"\"\"
        # 简化的对冲计算
        hedge_amount = portfolio['Value'].sum() * self.hedge_ratio

        return {
            'Hedge Ratio': self.hedge_ratio,
            'Hedge Amount': hedge_amount,
            'Implementation': 'Use S&P 500 futures'
        }

class RiskParityStrategy(RiskManagementStrategy):
    \"\"\"风险平价策略\"\"\"

    def __init__(self, target_risk_allocation):
        super().__init__()
        self.strategy_name = 'Risk Parity'
        self.target_risk_allocation = target_risk_allocation

    def evaluate_strategy(self, market_conditions):
        \"\"\"评估策略\"\"\"
        return 'Maintain Risk Parity'

    def rebalance(self, portfolio):
        \"\"\"再平衡投资组合\"\"\"
        # 简化的风险平价计算
        current_risk = {
            'Equities': 0.45,
            'Fixed Income': 0.30,
            'Commodities': 0.25
        }

        adjustments = {}

        for asset_class, target in self.target_risk_allocation.items():
            current = current_risk.get(asset_class, 0)
            if abs(target - current) > 0.02:
                adjustments[asset_class] = target - current

        return adjustments

def simulate_strategy_performance(strategy, market_data):
    \"\"\"模拟策略性能\"\"\"
    returns = []
    for period, data in market_data.items():
        strategy_action = strategy.evaluate_strategy(data)
        period_return = np.random.normal(0.01, 0.02)  # 简化的回报模拟

        if strategy_action == 'Increase Diversification':
            period_return *= 0.95
        elif strategy_action == 'Increase Hedge Ratio':
            period_return *= 0.85

        returns.append(period_return)

    return pd.Series(returns)

def backtest_risk_strategies(strategies, historical_data):
    \"\"\"回测风险策略\"\"\"
    results = []

    for strategy_name, strategy in strategies.items():
        returns = simulate_strategy_performance(strategy, historical_data)

        results.append({
            'Strategy': strategy_name,
            'Mean Return': returns.mean(),
            'Standard Deviation': returns.std(),
            'Sharpe Ratio': (returns.mean() - 0.02) / returns.std(),
            'Max Drawdown': min(returns)
        })

    return pd.DataFrame(results)

# 使用示例
# strategy = DiversificationStrategy()
# backtest_results = backtest_risk_strategies({
#     'Diversification': strategy
# }, historical_data)
# print("策略回测结果:")
# print(backtest_results)
""",
        "explanation": "此代码演示风险管理策略实现，包括分散化策略、对冲策略和风险平价策略，并提供策略评估和回测框架，为风险管理决策提供支持。"
    },
    {
        "topic_id": 8,
        "category_id": 6,
        "title": "完整风险管理系统架构",
        "code": """import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FinancialRiskManagementSystem:
    \"\"\"金融风险管理系统\"\"\"

    def __init__(self):
        self.data_providers = {}
        self.risk_models = {}
        self.risk_limits = {}
        self.monitoring_alerts = []
        self.report_generators = []

    def register_data_provider(self, name, provider):
        \"\"\"注册数据提供商\"\"\"
        self.data_providers[name] = provider

    def register_risk_model(self, name, model):
        \"\"\"注册风险模型\"\"\"
        self.risk_models[name] = model

    def set_risk_limits(self, limits):
        \"\"\"设置风险限制\"\"\"
        self.risk_limits = limits

    def generate_risk_report(self, portfolio, time_period):
        \"\"\"生成风险报告\"\"\"
        # 收集数据
        data = {name: provider.fetch_data(time_period) for name, provider in self.data_providers.items()}

        # 计算风险指标
        risk_metrics = {name: model.calculate_risk(data) for name, model in self.risk_models.items()}

        # 评估风险暴露
        risk_exposures = {}
        for metric, value in risk_metrics.items():
            if metric in self.risk_limits:
                risk_exposures[metric] = value['Value'] / self.risk_limits[metric]

        # 生成报告内容
        report = {
            'Timestamp': datetime.now(),
            'Portfolio Value': portfolio['Value'].sum(),
            'Risk Metrics': risk_metrics,
            'Risk Exposures': risk_exposures,
            'Compliance Status': self.check_compliance(risk_exposures)
        }

        return pd.DataFrame([report])

    def check_compliance(self, risk_exposures):
        \"\"\"检查合规性\"\"\"
        violations = []

        for metric, exposure in risk_exposures.items():
            if exposure > 1.0:
                violations.append(f"{metric}暴露超过限制: {exposure:.2f}")

        return 'Compliant' if not violations else 'Non-compliant'

    def trigger_alerts(self, portfolio, time_period):
        \"\"\"触发警报\"\"\"
        risk_report = self.generate_risk_report(portfolio, time_period)

        if risk_report['Compliance Status'].iloc[0] == 'Non-compliant':
            self.monitoring_alerts.append({
                'Alert Time': datetime.now(),
                'Portfolio Value': risk_report['Portfolio Value'].iloc[0],
                'Compliance Status': risk_report['Compliance Status'].iloc[0],
                'Details': risk_report['Risk Exposures'].iloc[0]
            })

    def optimize_risk_allocation(self, portfolio, target_risk):
        \"\"\"优化风险分配\"\"\"
        # 简化的风险分配优化
        asset_classes = portfolio.groupby('Asset Class')['Value'].sum()
        total_value = portfolio['Value'].sum()

        current_allocation = asset_classes / total_value

        # 简化的目标风险分配逻辑
        target_allocation = {
            'Equities': min(current_allocation.get('Equities', 0.5) - 0.05, 0.4),
            'Fixed Income': max(current_allocation.get('Fixed Income', 0.3) + 0.03, 0.35),
            'Cash': 0.25
        }

        return pd.DataFrame([{
            'Asset Class': asset_class,
            'Current Allocation': current_allocation.get(asset_class, 0),
            'Target Allocation': target,
            'Change': target - current_allocation.get(asset_class, 0)
        } for asset_class, target in target_allocation.items()])

def create_risk_management_workflow():
    \"\"\"创建风险管理工作流程\"\"\"
    system = FinancialRiskManagementSystem()

    # 注册数据提供商
    system.register_data_provider('Market Data', MarketDataProvider())
    system.register_data_provider('Credit Data', CreditDataProvider())
    system.register_data_provider('Operational Data', OperationalDataProvider())

    # 注册风险模型
    system.register_risk_model('Market Risk', MarketRiskModel())
    system.register_risk_model('Credit Risk', CreditRiskModel())
    system.register_risk_model('Operational Risk', OperationalRiskModel())

    # 设置风险限制
    system.set_risk_limits({
        'Market Risk': 0.03,
        'Credit Risk': 0.02,
        'Operational Risk': 0.01
    })

    return system

def run_risk_management_cycle(system, portfolio, time_period):
    \"\"\"运行风险管理循环\"\"\"
    system.trigger_alerts(portfolio, time_period)

    optimization_result = system.optimize_risk_allocation(portfolio, target_risk=0.025)

    if not optimization_result.empty:
        print("风险分配优化建议:")
        print(optimization_result)

    return system.generate_risk_report(portfolio, time_period)

class MarketDataProvider:
    \"\"\"市场数据提供商\"\"\"
    def fetch_data(self, time_period):
        return fetch_risk_data(['SPY', 'AAPL', 'MSFT'], time_period['Start'], time_period['End'])

class CreditDataProvider:
    \"\"\"信用数据提供商\"\"\"
    def fetch_data(self, time_period):
        return assess_credit_risk('BBB', 1000000, 500000)

class OperationalDataProvider:
    \"\"\"操作数据提供商\"\"\"
    def fetch_data(self, time_period):
        return evaluate_operational_risk([], ['Trading', 'Operations'])

class MarketRiskModel:
    \"\"\"市场风险模型\"\"\"
    def calculate_risk(self, data):
        return {'Value': var_historical(data['Market Data'])}

class CreditRiskModel:
    \"\"\"信用风险模型\"\"\"
    def calculate_risk(self, data):
        return {'Value': data['Credit Data']['Expected Loss']}

class OperationalRiskModel:
    \"\"\"操作风险模型\"\"\"
    def calculate_risk(self, data):
        return {'Value': max(data['Operational Data'].values())}

# 使用示例
# system = create_risk_management_workflow()
#
# time_period = {
#     'Start': '2020-01-01',
#     'End': '2023-12-31'
# }
#
# portfolio = pd.DataFrame({
#     'Symbol': ['AAPL', 'MSFT', 'SPY', 'TLT'],
#     'Asset Class': ['Equities', 'Equities', 'Equities', 'Fixed Income'],
#     'Sector': ['Technology', 'Technology', 'Broad Market', 'Fixed Income'],
#     'Quantity': [100, 50, 20, 50],
#     'Price': [180, 400, 450, 95],
#     'Value': [18000, 20000, 9000, 4750]
# })
#
# risk_report = run_risk_management_cycle(system, portfolio, time_period)
# print("风险报告:")
# print(risk_report)
""",
        "explanation": "此代码演示完整的金融风险管理系统架构，包括数据收集、风险评估、合规性检查和优化建议。该架构提供了全面的风险管理框架，适用于实际的金融风险管理需求。"
    },
    # 主题9：外汇交易系统
    {
        "topic_id": 9,
        "category_id": 1,
        "title": "外汇数据获取与处理",
        "code": """import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta

def fetch_exchange_rate(base_currency, target_currency, start_date, end_date):
    \"\"\"获取汇率数据\"\"\"
    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
        response = requests.get(url)
        data = response.json()
        rate = data['rates'][target_currency]

        print(f"当前汇率: 1 {base_currency} = {rate} {target_currency}")
        return rate
    except Exception as e:
        print(f"获取汇率数据失败: {e}")
        return None

def parse_forex_data(raw_data):
    \"\"\"解析外汇原始数据\"\"\"
    try:
        data = json.loads(raw_data)

        if 'Time Series FX (Daily)' in data:
            dates = []
            rates = []

            for date, values in data['Time Series FX (Daily)'].items():
                dates.append(date)
                rates.append(float(values['4. close']))

            df = pd.DataFrame({'Date': dates, 'Rate': rates})
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')

            return df
        else:
            print("数据格式不正确")
            return pd.DataFrame()

    except Exception as e:
        print(f"解析外汇数据失败: {e}")
        return pd.DataFrame()

def calculate_forex_returns(prices):
    \"\"\"计算汇率收益率\"\"\"
    returns = []

    for i in range(1, len(prices)):
        daily_return = (prices[i] - prices[i-1]) / prices[i-1]
        returns.append(daily_return)

    return returns

def preprocess_forex_data(df):
    \"\"\"预处理外汇数据\"\"\"
    # 检查缺失值
    if df.isnull().any().any():
        print("数据包含缺失值，将进行处理")
        df = df.dropna()

    # 计算收益率
    df['Return'] = df['Rate'].pct_change()

    # 计算移动平均
    df['MA5'] = df['Rate'].rolling(window=5).mean()
    df['MA20'] = df['Rate'].rolling(window=20).mean()

    # 计算波动率
    df['Volatility'] = df['Return'].rolling(window=20).std() * np.sqrt(252)

    return df

def get_forex_data_from_file(file_path):
    \"\"\"从文件读取外汇数据\"\"\"
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])

        return df
    except Exception as e:
        print(f"从文件读取数据失败: {e}")
        return pd.DataFrame()

# 使用示例
if __name__ == "__main__":
    # 获取汇率数据
    exchange_rate = fetch_exchange_rate("USD", "CNY")

    # 从文件读取数据
    data_file = "forex_data.csv"
    forex_data = get_forex_data_from_file(data_file)

    if not forex_data.empty:
        processed_data = preprocess_forex_data(forex_data)
        print(processed_data.head())
""",
        "explanation": "此代码演示外汇交易系统中的数据获取与处理，包括从API获取实时汇率、解析历史数据、计算收益率、预处理数据以及从文件读取数据等功能。这些功能是外汇交易系统的基础。"
    },
    {
        "topic_id": 9,
        "category_id": 2,
        "title": "汇率分析与预测",
        "code": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def analyze_exchange_rate_trends(data, currency_pair):
    \"\"\"分析汇率趋势\"\"\"
    print(f"汇率趋势分析 ({currency_pair}):")

    # 计算基础统计数据
    mean_rate = data['Rate'].mean()
    std_rate = data['Rate'].std()
    max_rate = data['Rate'].max()
    min_rate = data['Rate'].min()

    print(f"平均汇率: {mean_rate:.4f}")
    print(f"汇率标准差: {std_rate:.4f}")
    print(f"最高汇率: {max_rate:.4f}")
    print(f"最低汇率: {min_rate:.4f}")

    return {'mean': mean_rate, 'std': std_rate, 'max': max_rate, 'min': min_rate}

def build_forex_prediction_model(data, model_type='linear'):
    \"\"\"构建汇率预测模型\"\"\"
    # 准备特征
    data['Lag1'] = data['Rate'].shift(1)
    data['Lag2'] = data['Rate'].shift(2)
    data['Lag3'] = data['Rate'].shift(3)
    data = data.dropna()

    X = data[['Lag1', 'Lag2', 'Lag3', 'MA5', 'MA20']]
    y = data['Rate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError("不支持的模型类型")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"模型性能 - MSE: {mse:.4f}, R²: {r2:.4f}")

    return model, X_test, y_test, y_pred

def plot_exchange_rate_predictions(data, actual, predicted):
    \"\"\"绘制汇率预测结果\"\"\"
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[-len(actual):], actual, label='实际汇率')
    plt.plot(data.index[-len(predicted):], predicted, label='预测汇率', linestyle='--')
    plt.title('汇率预测')
    plt.xlabel('日期')
    plt.ylabel('汇率')
    plt.legend()
    plt.grid(True)
    plt.savefig('forex_prediction.png')
    plt.show()

def identify_forex_correlations(data, other_currency_data):
    \"\"\"识别汇率相关性\"\"\"
    # 计算相关性
    correlation = data['Rate'].corr(other_currency_data['Rate'])
    print(f"汇率相关性: {correlation:.4f}")

    # 绘制相关图
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Rate'], other_currency_data['Rate'])
    plt.title('汇率相关性')
    plt.xlabel('汇率1')
    plt.ylabel('汇率2')
    plt.grid(True)
    plt.savefig('forex_correlation.png')
    plt.show()

    return correlation

def forecast_exchange_rate(model, features):
    \"\"\"预测未来汇率\"\"\"
    try:
        prediction = model.predict(features)
        return prediction
    except Exception as e:
        print(f"预测失败: {e}")
        return None

# 使用示例
# data_file = "forex_data.csv"
# forex_data = get_forex_data_from_file(data_file)
# processed_data = preprocess_forex_data(forex_data)
#
# # 趋势分析
# analyze_exchange_rate_trends(processed_data, "USD/CNY")
#
# # 构建预测模型
# model, X_test, y_test, y_pred = build_forex_prediction_model(processed_data)
# plot_exchange_rate_predictions(processed_data, y_test, y_pred)
""",
        "explanation": "此代码演示汇率分析与预测功能，包括趋势分析、构建预测模型、绘制预测结果、识别汇率相关性以及预测未来汇率等。这些功能帮助外汇交易者做出更明智的决策。"
    },
    {
        "topic_id": 9,
        "category_id": 3,
        "title": "外汇交易策略实现",
        "code": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class ForexTradingStrategy:
    \"\"\"外汇交易策略基类\"\"\"

    def __init__(self, name):
        self.name = name
        self.signals = []
        self.positions = []

    def generate_signals(self, data):
        \"\"\"生成交易信号\"\"\"
        raise NotImplementedError("子类必须实现此方法")

    def plot_signals(self, data):
        \"\"\"绘制交易信号\"\"\"
        plt.figure(figsize=(12, 6))
        plt.plot(data['Date'], data['Rate'], label='汇率')

        for i, signal in enumerate(self.signals):
            date = data.iloc[i]['Date']
            rate = data.iloc[i]['Rate']

            if signal == 1:
                plt.scatter(date, rate, color='green', marker='^', label='买入信号')
            elif signal == -1:
                plt.scatter(date, rate, color='red', marker='v', label='卖出信号')

        plt.title(f"{self.name} - 交易信号")
        plt.xlabel('日期')
        plt.ylabel('汇率')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.name}_signals.png')
        plt.show()

class MovingAverageCrossoverStrategy(ForexTradingStrategy):
    \"\"\"移动平均交叉策略\"\"\"

    def __init__(self, short_window=5, long_window=20):
        super().__init__("移动平均交叉策略")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        \"\"\"生成移动平均交叉信号\"\"\"
        self.signals = np.zeros(len(data))

        for i in range(self.long_window, len(data)):
            short_avg = data['Rate'][i - self.short_window:i].mean()
            long_avg = data['Rate'][i - self.long_window:i].mean()

            if short_avg > long_avg and self.signals[i-1] != 1:
                self.signals[i] = 1
            elif short_avg < long_avg and self.signals[i-1] != -1:
                self.signals[i] = -1
            else:
                self.signals[i] = self.signals[i-1]

        return self.signals

class BollingerBandsStrategy(ForexTradingStrategy):
    \"\"\"布林带策略\"\"\"

    def __init__(self, window=20, num_std=2):
        super().__init__("布林带策略")
        self.window = window
        self.num_std = num_std

    def generate_signals(self, data):
        \"\"\"生成布林带交易信号\"\"\"
        self.signals = np.zeros(len(data))

        for i in range(self.window, len(data)):
            prices = data['Rate'][i - self.window:i]
            mean = prices.mean()
            std = prices.std()
            upper_band = mean + self.num_std * std
            lower_band = mean - self.num_std * std

            current_price = data['Rate'][i]

            if current_price < lower_band and self.signals[i-1] != 1:
                self.signals[i] = 1
            elif current_price > upper_band and self.signals[i-1] != -1:
                self.signals[i] = -1
            else:
                self.signals[i] = self.signals[i-1]

        return self.signals

class RSIOverboughtOversoldStrategy(ForexTradingStrategy):
    \"\"\"RSI超买超卖策略\"\"\"

    def __init__(self, window=14, overbought=70, oversold=30):
        super().__init__("RSI超买超卖策略")
        self.window = window
        self.overbought = overbought
        self.oversold = oversold

    def calculate_rsi(self, data):
        \"\"\"计算RSI指标\"\"\"
        delta = data['Rate'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signals(self, data):
        \"\"\"生成RSI交易信号\"\"\"
        rsi = self.calculate_rsi(data)
        self.signals = np.zeros(len(data))

        for i in range(self.window, len(data)):
            if rsi[i] < self.oversold and self.signals[i-1] != 1:
                self.signals[i] = 1
            elif rsi[i] > self.overbought and self.signals[i-1] != -1:
                self.signals[i] = -1
            else:
                self.signals[i] = self.signals[i-1]

        return self.signals

def backtest_strategy(data, strategy):
    \"\"\"回测交易策略\"\"\"
    # 生成信号
    signals = strategy.generate_signals(data)

    # 模拟交易
    position = 0
    positions = []
    portfolio_value = [10000]  # 初始资金

    for i in range(len(data)):
        if signals[i] == 1 and position == 0:
            position = 1
            shares = portfolio_value[-1] / data['Rate'][i]
            print(f"买入: {data['Date'][i]}, 价格: {data['Rate'][i]:.4f}")
        elif signals[i] == -1 and position == 1:
            position = 0
            portfolio_value.append(shares * data['Rate'][i])
            print(f"卖出: {data['Date'][i]}, 价格: {data['Rate'][i]:.4f}")

        positions.append(position)

    final_value = portfolio_value[-1]
    total_return = (final_value - 10000) / 10000 * 100
    print(f"最终价值: ${final_value:.2f}")
    print(f"总收益率: {total_return:.2f}%")

    return portfolio_value

# 使用示例
if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv("forex_data.csv")
    data['Date'] = pd.to_datetime(data['Date'])

    # 使用移动平均交叉策略
    strategy = MovingAverageCrossoverStrategy()
    signals = strategy.generate_signals(data)

    # 回测策略
    backtest_strategy(data, strategy)

    # 绘制信号图
    strategy.plot_signals(data)
""",
        "explanation": "此代码演示外汇交易策略的实现，包括移动平均交叉策略、布林带策略和RSI超买超卖策略。还包括策略回测和信号可视化功能，帮助交易者优化和评估交易策略。"
    },
    {
        "topic_id": 9,
        "category_id": 4,
        "title": "外汇风险控制与资金管理",
        "code": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ForexRiskManager:
    \"\"\"外汇风险管理器\"\"\"

    def __init__(self, initial_capital, risk_per_trade):
        \"\"\"
        初始化风险管理器
        参数:
            initial_capital: 初始资金
            risk_per_trade: 每笔交易风险比例（0-1）
        \"\"\"
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = 0
        self.drawdown = 0
        self.peak_value = initial_capital

    def calculate_position_size(self, entry_price, stop_loss_price):
        \"\"\"计算仓位大小\"\"\"
        risk_amount = self.current_capital * self.risk_per_trade

        risk_per_unit = entry_price - stop_loss_price

        if risk_per_unit <= 0:
            print("止损价格必须低于入场价格")
            return 0

        position_size = risk_amount / risk_per_unit

        return position_size

    def update_risk_metrics(self, current_portfolio_value):
        \"\"\"更新风险指标\"\"\"
        # 更新峰值和最大回撤
        if current_portfolio_value > self.peak_value:
            self.peak_value = current_portfolio_value
            self.drawdown = 0
        else:
            self.drawdown = (self.peak_value - current_portfolio_value) / self.peak_value

            if self.drawdown > self.max_drawdown:
                self.max_drawdown = self.drawdown

        # 更新当前资金
        self.current_capital = current_portfolio_value

    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        \"\"\"计算夏普比率\"\"\"
        excess_returns = np.array(returns) - risk_free_rate

        mean_excess_return = np.mean(excess_returns)
        std_excess_return = np.std(excess_returns)

        if std_excess_return == 0:
            return 0

        sharpe_ratio = mean_excess_return / std_excess_return

        return sharpe_ratio

    def calculate_sortino_ratio(self, returns, risk_free_rate=0.02, target_return=0):
        \"\"\"计算索提诺比率\"\"\"
        excess_returns = np.array(returns) - risk_free_rate

        downside_returns = np.where(excess_returns < target_return, excess_returns, 0)

        mean_downside = np.mean(downside_returns)
        std_downside = np.std(downside_returns)

        if std_downside == 0:
            return 0

        sortino_ratio = (np.mean(excess_returns) - target_return) / std_downside

        return sortino_ratio

    def print_risk_summary(self):
        \"\"\"打印风险概要\"\"\"
        print("风险概要:")
        print(f"当前资金: {self.current_capital:.2f}")
        print(f"初始资金: {self.initial_capital:.2f}")
        print(f"最大回撤: {self.max_drawdown:.2%}")
        print(f"风险/交易: {self.risk_per_trade:.2%}")

def plot_drawdown(portfolio_values):
    \"\"\"绘制回撤曲线\"\"\"
    peak = portfolio_values[0]
    drawdowns = []

    for value in portfolio_values:
        if value > peak:
            peak = value

        drawdown = (peak - value) / peak
        drawdowns.append(drawdown)

    plt.figure(figsize=(12, 6))
    plt.plot(drawdowns)
    plt.title('最大回撤')
    plt.xlabel('时间')
    plt.ylabel('回撤')
    plt.grid(True)
    plt.savefig('drawdown.png')
    plt.show()

def calculate_drawdown(portfolio_values):
    \"\"\"计算最大回撤\"\"\"
    max_value = portfolio_values[0]
    max_drawdown = 0

    for value in portfolio_values:
        if value > max_value:
            max_value = value
        else:
            drawdown = (max_value - value) / max_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    return max_drawdown

def risk_adjusted_position_sizing(volatility, risk_per_trade, current_price, initial_capital):
    \"\"\"风险调整仓位规模\"\"\"
    daily_volatility = volatility / np.sqrt(252)

    stop_loss_distance = current_price * daily_volatility * 2

    risk_amount = initial_capital * risk_per_trade

    position_size = risk_amount / stop_loss_distance

    return position_size

def implement_trailing_stop_loss(data, entry_price, trail_distance):
    \"\"\"实现追踪止损\"\"\"
    stop_loss_prices = []

    highest_price = entry_price

    for i in range(len(data)):
        current_price = data['Rate'][i]

        if current_price > highest_price:
            highest_price = current_price

        stop_loss_price = highest_price * (1 - trail_distance)
        stop_loss_prices.append(stop_loss_price)

    return stop_loss_prices

# 使用示例
if __name__ == "__main__":
    # 初始化风险管理器
    risk_manager = ForexRiskManager(10000, 0.02)

    # 计算仓位大小
    entry_price = 6.90
    stop_loss_price = 6.85
    position_size = risk_manager.calculate_position_size(entry_price, stop_loss_price)
    print(f"建议仓位大小: {position_size:.2f}")

    # 更新风险指标
    portfolio_values = [10000, 10500, 10300, 11000, 10800, 12000]
    for value in portfolio_values:
        risk_manager.update_risk_metrics(value)

    # 打印风险概要
    risk_manager.print_risk_summary()
""",
        "explanation": "此代码演示外汇风险控制与资金管理，包括风险管理器类、仓位大小计算、风险指标跟踪、回撤分析、风险调整仓位规模以及追踪止损实现。这些功能帮助交易者管理风险，保护资金，并优化交易策略的表现。"
    },
    {
        "topic_id": 9,
        "category_id": 5,
        "title": "外汇交易执行与监控",
        "code": """import pandas as pd
import numpy as np
from datetime import datetime
import time
from collections import deque

class ForexTradingSystem:
    \"\"\"外汇交易系统\"\"\"

    def __init__(self, api_interface, strategy, risk_manager):
        self.api_interface = api_interface
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.trade_history = []
        self.pending_orders = []
        self.current_position = 0

    def connect_to_exchange(self):
        \"\"\"连接到交易所\"\"\"
        if self.api_interface.connect():
            print("成功连接到交易所")
            return True
        else:
            print("连接到交易所失败")
            return False

    def execute_order(self, order_type, price, quantity):
        \"\"\"执行订单\"\"\"
        try:
            if order_type == "buy":
                order_id = self.api_interface.place_buy_order(price, quantity)
                print(f"买入订单执行成功，订单ID: {order_id}")
            elif order_type == "sell":
                order_id = self.api_interface.place_sell_order(price, quantity)
                print(f"卖出订单执行成功，订单ID: {order_id}")

            # 添加到订单历史
            self.trade_history.append({
                "order_id": order_id,
                "order_type": order_type,
                "price": price,
                "quantity": quantity,
                "timestamp": datetime.now()
            })

            return order_id
        except Exception as e:
            print(f"订单执行失败: {e}")
            return None

    def monitor_positions(self):
        \"\"\"监控持仓\"\"\"
        positions = self.api_interface.get_positions()
        self.current_position = positions.get("USD/CNY", 0)

        print(f"当前持仓: {self.current_position:.2f}")

        return self.current_position

    def monitor_pending_orders(self):
        \"\"\"监控待处理订单\"\"\"
        for order in self.pending_orders.copy():
            status = self.api_interface.get_order_status(order['order_id'])

            if status == "filled":
                print(f"订单 {order['order_id']} 已完成")
                self.trade_history.append(order)
                self.pending_orders.remove(order)
            elif status == "canceled":
                print(f"订单 {order['order_id']} 已取消")
                self.pending_orders.remove(order)
            elif status == "expired":
                print(f"订单 {order['order_id']} 已过期")
                self.pending_orders.remove(order)

    def run_trading_loop(self):
        \"\"\"运行交易循环\"\"\"
        while True:
            # 获取最新数据
            latest_data = self.api_interface.get_latest_data()

            # 生成信号
            signal = self.strategy.generate_signals(latest_data)

            # 监控持仓和订单
            positions = self.monitor_positions()
            self.monitor_pending_orders()

            # 根据信号执行交易
            if signal == 1 and positions == 0:
                entry_price = latest_data['Rate'].iloc[-1]
                stop_loss_price = entry_price * 0.99

                position_size = self.risk_manager.calculate_position_size(entry_price, stop_loss_price)

                if position_size > 0:
                    order_id = self.execute_order("buy", entry_price, position_size)

                    if order_id:
                        self.pending_orders.append({
                            "order_id": order_id,
                            "order_type": "buy",
                            "price": entry_price,
                            "quantity": position_size,
                            "timestamp": datetime.now()
                        })

            elif signal == -1 and positions > 0:
                exit_price = latest_data['Rate'].iloc[-1]
                order_id = self.execute_order("sell", exit_price, positions)

                if order_id:
                    self.pending_orders.append({
                        "order_id": order_id,
                        "order_type": "sell",
                        "price": exit_price,
                        "quantity": positions,
                        "timestamp": datetime.now()
                    })

            # 暂停
            time.sleep(60)

class MockAPI:
    \"\"\"模拟API接口\"\"\"

    def __init__(self, initial_price=6.90):
        self.connection_status = False
        self.latest_price = initial_price
        self.orders = []
        self.positions = {}
        self.order_counter = 0

    def connect(self):
        \"\"\"模拟连接\"\"\"
        self.connection_status = True
        return True

    def disconnect(self):
        \"\"\"模拟断开连接\"\"\"
        self.connection_status = False
        return True

    def place_buy_order(self, price, quantity):
        \"\"\"模拟买入订单\"\"\"
        order_id = f"BUY{self.order_counter:04d}"
        self.order_counter += 1

        self.orders.append({
            "order_id": order_id,
            "type": "buy",
            "price": price,
            "quantity": quantity,
            "status": "filled"
        })

        if "USD/CNY" not in self.positions:
            self.positions["USD/CNY"] = 0

        self.positions["USD/CNY"] += quantity

        return order_id

    def place_sell_order(self, price, quantity):
        \"\"\"模拟卖出订单\"\"\"
        order_id = f"SELL{self.order_counter:04d}"
        self.order_counter += 1

        self.orders.append({
            "order_id": order_id,
            "type": "sell",
            "price": price,
            "quantity": quantity,
            "status": "filled"
        })

        if "USD/CNY" in self.positions:
            self.positions["USD/CNY"] -= quantity

            if self.positions["USD/CNY"] < 0:
                self.positions["USD/CNY"] = 0

        return order_id

    def get_positions(self):
        \"\"\"获取持仓\"\"\"
        return self.positions

    def get_order_status(self, order_id):
        \"\"\"获取订单状态\"\"\"
        for order in self.orders:
            if order["order_id"] == order_id:
                return order["status"]

        return "not_found"

    def get_latest_data(self):
        \"\"\"获取最新数据\"\"\"
        # 模拟汇率变化
        random_change = (np.random.random() - 0.5) * 0.01
        self.latest_price *= (1 + random_change)

        # 构造数据
        data = pd.DataFrame({
            "Date": [datetime.now()],
            "Rate": [self.latest_price],
            "Volume": [10000]
        })

        return data

# 使用示例
if __name__ == "__main__":
    # 初始化组件
    api = MockAPI()
    strategy = MovingAverageCrossoverStrategy()
    risk_manager = ForexRiskManager(10000, 0.02)

    # 初始化交易系统
    trading_system = ForexTradingSystem(api, strategy, risk_manager)

    # 连接到交易所
    trading_system.connect_to_exchange()

    # 获取最新数据
    latest_data = api.get_latest_data()

    # 监控持仓
    positions = trading_system.monitor_positions()
    print(f"当前持仓: {positions}")

    # 运行交易循环（模拟）
    try:
        for i in range(10):
            print(f"\\n=== 交易循环 {i+1} ===")

            # 获取最新数据
            latest_data = api.get_latest_data()
            print(f"最新汇率: {latest_data['Rate'].iloc[-1]:.4f}")

            # 模拟交易
            if i % 3 == 0 and positions == 0:
                entry_price = latest_data['Rate'].iloc[-1]
                stop_loss_price = entry_price * 0.99
                position_size = risk_manager.calculate_position_size(entry_price, stop_loss_price)
                order_id = trading_system.execute_order("buy", entry_price, position_size)
                print(f"买入订单: {order_id}")
            elif i % 5 == 0 and positions > 0:
                exit_price = latest_data['Rate'].iloc[-1]
                order_id = trading_system.execute_order("sell", exit_price, positions)
                print(f"卖出订单: {order_id}")

            # 更新风险指标
            trading_system.monitor_positions()

            time.sleep(1)
    except KeyboardInterrupt:
        print("交易循环已停止")
""",
        "explanation": "此代码演示外汇交易执行与监控系统，包括交易系统架构、订单执行、风险管理和交易监控。还包含模拟API接口，可用于测试和演示交易系统功能。"
    },
    {
        "topic_id": 9,
        "category_id": 6,
        "title": "完整外汇交易系统项目",
        "code": """import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

class CompleteForexTradingSystem:
    \"\"\"完整的外汇交易系统\"\"\"

    def __init__(self, config_file="config.json"):
        self.config = self.load_config(config_file)
        self.api = None
        self.strategy = None
        self.risk_manager = None
        self.data_processor = None

    def load_config(self, config_file):
        \"\"\"加载配置文件\"\"\"
        if not os.path.exists(config_file):
            print("配置文件不存在")
            return self.default_config()

        try:
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return self.default_config()

    def default_config(self):
        \"\"\"默认配置\"\"\"
        return {
            "api": {
                "type": "mock",
                "key": "test_key"
            },
            "strategy": {
                "type": "ma_crossover",
                "parameters": {
                    "short_window": 5,
                    "long_window": 20
                }
            },
            "risk": {
                "initial_capital": 10000,
                "risk_per_trade": 0.02,
                "max_drawdown": 0.2
            }
        }

    def initialize_components(self):
        \"\"\"初始化各个组件\"\"\"
        print("初始化系统组件...")

        # 初始化API接口
        if self.config["api"]["type"] == "mock":
            class MockAPI:
                def __init__(self, initial_price=6.90):
                    self.connection_status = False
                    self.latest_price = initial_price
                    self.orders = []
                    self.positions = {}
                    self.order_counter = 0

                def connect(self):
                    self.connection_status = True
                    return True

                def get_latest_data(self):
                    return pd.DataFrame({
                        "Date": [datetime.now()],
                        "Rate": [self.latest_price],
                        "Volume": [10000]
                    })

            self.api = MockAPI()
        else:
            raise NotImplementedError("实际API接口未实现")

        # 初始化策略
        if self.config["strategy"]["type"] == "ma_crossover":
            class MovingAverageCrossoverStrategy:
                def __init__(self, short_window=5, long_window=20):
                    self.short_window = short_window
                    self.long_window = long_window

                def generate_signals(self, data):
                    return 1 if np.random.random() > 0.5 else -1

            self.strategy = MovingAverageCrossoverStrategy()

        # 初始化风险管理器
        class ForexRiskManager:
            def __init__(self, initial_capital, risk_per_trade):
                self.initial_capital = initial_capital
                self.current_capital = initial_capital
                self.risk_per_trade = risk_per_trade
                self.max_drawdown = 0
                self.drawdown = 0
                self.peak_value = initial_capital

            def calculate_position_size(self, entry_price, stop_loss_price):
                return 1000

            def update_risk_metrics(self, current_portfolio_value):
                pass

        self.risk_manager = ForexRiskManager(
            self.config["risk"]["initial_capital"],
            self.config["risk"]["risk_per_trade"]
        )

    def connect(self):
        \"\"\"连接到交易接口\"\"\"
        print("连接到交易接口...")
        return self.api.connect()

    def load_data(self):
        \"\"\"加载历史数据\"\"\"
        print("加载历史数据...")

        data = pd.DataFrame({
            "Date": pd.date_range(start="2023-01-01", periods=30),
            "Rate": np.random.uniform(6.8, 7.2, 30)
        })

        return data

    def backtest_strategy(self, data):
        \"\"\"回测策略\"\"\"
        print("开始回测策略...")

        # 预处理数据
        data['Return'] = data['Rate'].pct_change()
        data['MA5'] = data['Rate'].rolling(window=5).mean()
        data['MA20'] = data['Rate'].rolling(window=20).mean()

        # 生成交易信号
        signals = [0] * len(data)
        for i in range(20, len(data)):
            if data['MA5'][i] > data['MA20'][i] and signals[i-1] != 1:
                signals[i] = 1
            elif data['MA5'][i] < data['MA20'][i] and signals[i-1] != -1:
                signals[i] = -1
            else:
                signals[i] = signals[i-1]

        # 回测策略
        portfolio_values = [10000]
        position = 0
        shares = 0

        for i in range(len(data)):
            if signals[i] == 1 and position == 0:
                position = 1
                shares = portfolio_values[-1] / data['Rate'][i]
                print(f"买入: {data['Date'][i]}, 价格: {data['Rate'][i]:.4f}")
            elif signals[i] == -1 and position == 1:
                position = 0
                portfolio_values.append(shares * data['Rate'][i])
                print(f"卖出: {data['Date'][i]}, 价格: {data['Rate'][i]:.4f}")

        return data, signals, portfolio_values

    def run_live_trading(self):
        \"\"\"运行实时交易\"\"\"
        print("启动实时交易...")

        try:
            while True:
                # 获取最新数据
                latest_data = self.api.get_latest_data()

                # 预处理数据
                latest_data['Return'] = latest_data['Rate'].pct_change()
                latest_data['MA5'] = latest_data['Rate'].rolling(window=5).mean()
                latest_data['MA20'] = latest_data['Rate'].rolling(window=20).mean()

                # 生成交易信号
                signal = self.strategy.generate_signals(latest_data)

                # 执行交易
                self.execute_trade(signal, latest_data)

                # 更新风险指标
                self.risk_manager.update_risk_metrics(self.api.get_positions())

                # 打印状态
                self.print_status()

                # 暂停
                import time
                time.sleep(self.config.get("trading_interval", 60))

        except KeyboardInterrupt:
            print("交易已停止")
            return

    def execute_trade(self, signal, data):
        \"\"\"执行交易\"\"\"
        if signal == 1 and self.api.get_positions().get("USD/CNY", 0) == 0:
            entry_price = data['Rate'].iloc[-1]
            stop_loss_price = entry_price * 0.99

            position_size = self.risk_manager.calculate_position_size(entry_price, stop_loss_price)

            if position_size > 0:
                order_id = self.api.place_buy_order(entry_price, position_size)

                if order_id:
                    self.pending_orders.append({
                        "order_id": order_id,
                        "order_type": "buy",
                        "price": entry_price,
                        "quantity": position_size,
                        "timestamp": datetime.now()
                    })

        elif signal == -1 and self.api.get_positions().get("USD/CNY", 0) > 0:
            exit_price = data['Rate'].iloc[-1]
            order_id = self.api.place_sell_order(exit_price, self.api.get_positions().get("USD/CNY", 0))

            if order_id:
                self.pending_orders.append({
                    "order_id": order_id,
                    "order_type": "sell",
                    "price": exit_price,
                    "quantity": self.api.get_positions().get("USD/CNY", 0),
                    "timestamp": datetime.now()
                })

    def print_status(self):
        \"\"\"打印状态\"\"\"
        positions = self.api.get_positions()
        print(f"\\n=== 系统状态 ===\\n时间: {datetime.now()}\\n持仓: {positions.get('USD/CNY', 0):.2f}")

def main():
    \"\"\"主函数\"\"\"
    # 创建交易系统
    system = CompleteForexTradingSystem()

    # 初始化系统
    system.initialize_components()

    # 连接到交易接口
    connected = system.connect()
    if not connected:
        print("连接失败，程序退出")
        return

    # 加载历史数据
    data = system.load_data()
    if data.empty:
        print("数据加载失败，程序退出")
        return

    # 回测策略
    processed_data, signals, portfolio_values = system.backtest_strategy(data)

    # 生成报告
    system.generate_report(processed_data, signals, portfolio_values)

    # 根据参数决定是否运行实时交易
    if "--live" in sys.argv:
        system.run_live_trading()

if __name__ == "__main__":
    main()
""",
        "explanation": "此代码演示完整的外汇交易系统项目，包括系统架构、组件初始化、历史数据回测、实时交易运行和报告生成。该项目提供了完整的代码结构，可作为实际外汇交易系统开发的基础框架。"
    },
    # 主题10：数据分析可视化
    {
        "topic_id": 10,
        "category_id": 1,
        "title": "金融数据可视化基础",
        "code": """import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def create_basic_plot():
    \"\"\"创建基础图表\"\"\"
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y)
    plt.title("基础折线图")
    plt.xlabel("X轴")
    plt.ylabel("Y轴")
    plt.grid(True)
    plt.savefig("basic_plot.png")
    plt.show()

def plot_multiple_data():
    \"\"\"绘制多组数据\"\"\"
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label="sin(x)", color="blue", linewidth=2)
    plt.plot(x, y2, label="cos(x)", color="red", linewidth=2, linestyle="--")

    plt.title("多个函数的可视化")
    plt.xlabel("X轴")
    plt.ylabel("Y轴")
    plt.legend()
    plt.grid(True)
    plt.savefig("multiple_data_plot.png")
    plt.show()

def create_scatter_plot():
    \"\"\"创建散点图\"\"\"
    np.random.seed(42)
    x = np.random.rand(100)
    y = np.random.rand(100)
    size = 1000 * np.random.rand(100)
    color = np.random.rand(100)

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, s=size, c=color, alpha=0.5, cmap="viridis")

    plt.title("彩色散点图")
    plt.xlabel("X轴")
    plt.ylabel("Y轴")
    plt.colorbar(label="颜色")
    plt.savefig("scatter_plot.png")
    plt.show()

def basic_statistical_plot():
    \"\"\"创建统计图表\"\"\"
    np.random.seed(42)
    data = np.random.randn(1000)

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, alpha=0.7, density=True)

    mu, sigma = np.mean(data), np.std(data)
    x = np.linspace(data.min(), data.max(), 100)
    y = (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-0.5*((x - mu)/sigma)**2)
    plt.plot(x, y, color="red", linewidth=2, label="正态分布")

    plt.title("直方图与概率密度曲线")
    plt.xlabel("值")
    plt.ylabel("频率")
    plt.legend()
    plt.savefig("statistical_plot.png")
    plt.show()

# 使用示例
if __name__ == "__main__":
    print("创建基础图表")
    create_basic_plot()

    print("绘制多组数据")
    plot_multiple_data()

    print("创建散点图")
    create_scatter_plot()

    print("创建统计图表")
    basic_statistical_plot()
""",
        "explanation": "此代码演示了金融数据可视化的基础内容，包括绘制基础图表、多组数据、散点图和统计图表等基本方法。这些方法适用于各种金融数据分析场景。"
    },
    {
        "topic_id": 10,
        "category_id": 2,
        "title": "金融数据可视化进阶",
        "code": """import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def financial_time_series_plot():
    \"\"\"绘制金融时间序列图表\"\"\"
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
    values = np.random.randn(365).cumsum()

    df = pd.DataFrame({"Date": dates, "Value": values})

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Value"], color="blue", linewidth=1.5)

    plt.title("金融时间序列图表")
    plt.xlabel("日期")
    plt.ylabel("值")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.savefig("time_series_plot.png")
    plt.show()

def candlestick_chart():
    \"\"\"创建K线图\"\"\"
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    open_price = 100 + np.random.randn(100).cumsum()
    high_price = open_price + np.random.rand(100) * 2
    low_price = open_price - np.random.rand(100) * 2
    close_price = open_price + np.random.randn(100)

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Close": close_price
    })

    plt.figure(figsize=(12, 6))

    for i in range(len(df)):
        color = "green" if df["Close"][i] >= df["Open"][i] else "red"
        plt.vlines(df["Date"][i], df["Low"][i], df["High"][i], color=color, linewidth=1)
        plt.hlines(df["Open"][i], df["Date"][i] - pd.Timedelta(days=0.25),
                  df["Date"][i] + pd.Timedelta(days=0.25), color=color, linewidth=2)
        plt.hlines(df["Close"][i], df["Date"][i] - pd.Timedelta(days=0.25),
                  df["Date"][i] + pd.Timedelta(days=0.25), color=color, linewidth=2)

    plt.title("K线图")
    plt.xlabel("日期")
    plt.ylabel("价格")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.savefig("candlestick_chart.png")
    plt.show()

def financial_heatmap():
    \"\"\"创建金融数据热力图\"\"\"
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
    returns = np.random.randn(len(tickers), len(dates))

    df = pd.DataFrame(returns, index=tickers, columns=dates)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df, annot=True, cmap="RdYlGn", center=0)

    plt.title("股票收益率热力图")
    plt.xlabel("日期")
    plt.ylabel("股票代码")

    plt.savefig("financial_heatmap.png")
    plt.show()

def financial_box_plot():
    \"\"\"创建金融数据箱线图\"\"\"
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    data = []

    for _ in range(len(tickers)):
        data.append(np.random.randn(100) + np.random.rand())

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=tickers)

    plt.title("股票收益率箱线图")
    plt.xlabel("股票代码")
    plt.ylabel("收益率")
    plt.grid(True, alpha=0.3)

    plt.savefig("financial_box_plot.png")
    plt.show()

# 使用示例
if __name__ == "__main__":
    print("绘制金融时间序列图表")
    financial_time_series_plot()

    print("创建K线图")
    candlestick_chart()

    print("创建金融数据热力图")
    financial_heatmap()

    print("创建金融数据箱线图")
    financial_box_plot()
""",
        "explanation": "此代码演示了金融数据可视化的进阶内容，包括时间序列图表、K线图、热力图和箱线图等。这些图表在金融数据分析中具有广泛的应用。"
    },
    {
        "topic_id": 10,
        "category_id": 3,
        "title": "交互式数据可视化",
        "code": """import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def interactive_time_series():
    \"\"\"创建交互式时间序列图表\"\"\"
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
    values = np.random.randn(365).cumsum()

    df = pd.DataFrame({"Date": dates, "Value": values})

    fig = px.line(df, x="Date", y="Value",
                  title="交互式时间序列图表",
                  labels={"Date": "日期", "Value": "值"},
                  hover_data={"Date": "|%Y-%m-%d", "Value": ":.2f"})

    fig.update_traces(line_color="blue", line_width=1.5)
    fig.update_layout(
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    )

    fig.write_html("interactive_time_series.html")
    fig.show()

def interactive_candlestick():
    \"\"\"创建交互式K线图\"\"\"
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    open_price = 100 + np.random.randn(100).cumsum()
    high_price = open_price + np.random.rand(100) * 2
    low_price = open_price - np.random.rand(100) * 2
    close_price = open_price + np.random.randn(100)

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_price,
        "High": high_price,
        "Low": low_price,
        "Close": close_price
    })

    fig = go.Figure(data=[go.Candlestick(x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"])])

    fig.update_layout(
        title="交互式K线图",
        yaxis_title="价格",
        xaxis_rangeslider_visible=False
    )

    fig.write_html("interactive_candlestick.html")
    fig.show()

def interactive_scatter_plot():
    \"\"\"创建交互式散点图\"\"\"
    np.random.seed(42)
    x = np.random.rand(100)
    y = np.random.rand(100)
    size = 1000 * np.random.rand(100)
    color = np.random.rand(100)

    df = pd.DataFrame({"X": x, "Y": y, "Size": size, "Color": color})

    fig = px.scatter(df, x="X", y="Y", size="Size", color="Color",
                     title="交互式散点图",
                     labels={"X": "X轴", "Y": "Y轴", "Size": "大小", "Color": "颜色"},
                     size_max=30,
                     color_continuous_scale="viridis")

    fig.update_layout(
        hovermode="closest",
        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)")
    )

    fig.write_html("interactive_scatter.html")
    fig.show()

def interactive_heatmap():
    \"\"\"创建交互式热力图\"\"\"
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    dates = pd.date_range(start="2020-01-01", periods=5, freq="D")
    returns = np.random.randn(len(tickers), len(dates))

    df = pd.DataFrame(returns, index=tickers, columns=dates)

    fig = px.imshow(df,
                    x=dates,
                    y=tickers,
                    title="交互式热力图",
                    labels={"x": "日期", "y": "股票代码", "color": "收益率"},
                    color_continuous_scale="RdYlGn",
                    zmin=-2, zmax=2)

    fig.update_xaxes(tickangle=45)
    fig.update_layout(width=1000, height=600)

    fig.write_html("interactive_heatmap.html")
    fig.show()

# 使用示例
if __name__ == "__main__":
    print("创建交互式时间序列图表")
    interactive_time_series()

    print("创建交互式K线图")
    interactive_candlestick()

    print("创建交互式散点图")
    interactive_scatter_plot()

    print("创建交互式热力图")
    interactive_heatmap()
""",
        "explanation": "此代码演示了交互式数据可视化技术，使用Plotly库创建了可交互的图表，包括时间序列、K线图、散点图和热力图。这些图表支持缩放、平移、悬停等交互功能。"
    },
    {
        "topic_id": 10,
        "category_id": 4,
        "title": "金融数据可视化最佳实践",
        "code": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def choose_appropriate_chart():
    \"\"\"选择合适的图表类型\"\"\"
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "Arial",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10
    })

    np.random.seed(42)
    data = np.random.randn(1000)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(data, bins=30, alpha=0.7, density=True)
    axes[0, 0].set_title("直方图")
    axes[0, 0].set_xlabel("值")
    axes[0, 0].set_ylabel("频率")

    axes[0, 1].boxplot(data)
    axes[0, 1].set_title("箱线图")
    axes[0, 1].set_ylabel("值")

    axes[1, 0].scatter(range(len(data)), data, alpha=0.3, s=2)
    axes[1, 0].set_title("散点图")
    axes[1, 0].set_xlabel("索引")
    axes[1, 0].set_ylabel("值")

    sns.kdeplot(data, ax=axes[1, 1])
    axes[1, 1].set_title("核密度估计")
    axes[1, 1].set_xlabel("值")
    axes[1, 1].set_ylabel("密度")

    plt.tight_layout()
    plt.savefig("chart_comparison.png")
    plt.show()

def color_usage():
    \"\"\"颜色使用最佳实践\"\"\"
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "Arial"
    })

    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(x, y1, color="red", label="sin(x)", linewidth=2)
    axes[0].plot(x, y2, color="blue", label="cos(x)", linewidth=2)
    axes[0].plot(x, y3, color="green", label="tan(x)", linewidth=2)
    axes[0].set_title("不良颜色搭配")
    axes[0].legend()

    axes[1].plot(x, y1, color="#1f77b4", label="sin(x)", linewidth=2)
    axes[1].plot(x, y2, color="#ff7f0e", label="cos(x)", linewidth=2)
    axes[1].plot(x, y3, color="#2ca02c", label="tan(x)", linewidth=2)
    axes[1].set_title("良好颜色搭配")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("color_usage.png")
    plt.show()

def label_and_annotation():
    \"\"\"标签和注释最佳实践\"\"\"
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "Arial"
    })

    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")
    values = np.random.randn(365).cumsum()

    df = pd.DataFrame({"Date": dates, "Value": values})

    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Value"], color="blue", linewidth=1.5)

    plt.title("金融时间序列图表 - 标签和注释示例")
    plt.xlabel("日期")
    plt.ylabel("值")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # 添加重要点的标签
    important_dates = [
        pd.Timestamp("2020-03-15"),
        pd.Timestamp("2020-06-30"),
        pd.Timestamp("2020-09-01")
    ]

    for date in important_dates:
        value = df.loc[df["Date"] == date, "Value"].iloc[0]
        plt.annotate(f"重要日期",
                    xy=(date, value),
                    xytext=(10, 10),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="red"))

    plt.tight_layout()
    plt.savefig("annotation_example.png")
    plt.show()

def layout_optimization():
    \"\"\"布局优化最佳实践\"\"\"
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "Arial"
    })

    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)

    fig = plt.figure(figsize=(12, 8))

    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(x, y1, color="#1f77b4", linewidth=2)
    ax1.set_title("sin(x)")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(x, y2, color="#ff7f0e", linewidth=2)
    ax2.set_title("cos(x)")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(x, y3, color="#2ca02c", linewidth=2)
    ax3.set_title("tan(x)")
    ax3.set_ylim(-10, 10)

    plt.tight_layout()
    plt.savefig("layout_optimization.png")
    plt.show()

# 使用示例
if __name__ == "__main__":
    print("图表类型选择")
    choose_appropriate_chart()

    print("颜色使用最佳实践")
    color_usage()

    print("标签和注释示例")
    label_and_annotation()

    print("布局优化最佳实践")
    layout_optimization()
""",
        "explanation": "此代码演示了金融数据可视化的最佳实践，包括图表类型选择、颜色使用、标签和注释、布局优化等。遵循这些最佳实践可以创建出更清晰、更有效的可视化图表。"
    },
    {
        "topic_id": 10,
        "category_id": 5,
        "title": "机器学习结果可视化",
        "code": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def data_reduction_visualization():
    \"\"\"数据降维可视化\"\"\"
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 3, size=100)

    # 使用PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 使用t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
    plt.title("PCA降维可视化")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap="viridis")
    plt.title("t-SNE降维可视化")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("dimensionality_reduction.png")
    plt.show()

def cluster_visualization():
    \"\"\"聚类结果可视化\"\"\"
    np.random.seed(42)
    X = np.random.rand(100, 2)
    y = np.random.randint(0, 3, size=100)

    # 使用KMeans聚类
    kmeans = KMeans(n_clusters=3, random_state=42)
    y_pred = kmeans.fit_predict(X)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis")
    plt.title("真实标签")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="viridis")
    plt.title("KMeans聚类结果")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig("cluster_visualization.png")
    plt.show()

def feature_importance():
    \"\"\"特征重要性可视化\"\"\"
    np.random.seed(42)
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=100)

    # 使用随机森林分类器
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    # 获取特征重要性
    feature_importances = clf.feature_importances_
    features = [f"Feature {i}" for i in range(10)]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(features)), feature_importances, color="#1f77b4")
    plt.yticks(range(len(features)), features)
    plt.title("特征重要性")
    plt.xlabel("重要性")

    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()

def confusion_matrix_visualization():
    \"\"\"混淆矩阵可视化\"\"\"
    np.random.seed(42)
    y_true = np.random.randint(0, 3, size=100)
    y_pred = np.random.randint(0, 3, size=100)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="viridis")
    plt.title("混淆矩阵")
    plt.colorbar()

    # 添加标签
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white", fontsize=12)

    plt.xlabel("预测标签")
    plt.ylabel("真实标签")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()

# 使用示例
if __name__ == "__main__":
    print("数据降维可视化")
    data_reduction_visualization()

    print("聚类结果可视化")
    cluster_visualization()

    print("特征重要性可视化")
    feature_importance()

    print("混淆矩阵可视化")
    confusion_matrix_visualization()
""",
        "explanation": "此代码演示了机器学习结果的可视化方法，包括数据降维可视化、聚类结果可视化、特征重要性可视化和混淆矩阵可视化等。这些可视化方法可以帮助我们更好地理解机器学习模型的行为和性能。"
    },
    {
        "topic_id": 10,
        "category_id": 6,
        "title": "金融数据分析可视化实战",
        "code": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

def stock_price_analysis():
    \"\"\"股票价格分析\"\"\"
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "Arial"
    })

    # 获取苹果公司股票数据
    apple = yf.Ticker("AAPL")
    df = apple.history(start="2020-01-01", end="2021-01-01")

    # 数据预处理
    df["Date"] = df.index
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(window=20).std() * np.sqrt(252)
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA20"] = df["Close"].rolling(window=20).mean()

    plt.figure(figsize=(12, 8))

    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])

    ax1 = plt.subplot(gs[0])
    ax1.plot(df["Date"], df["Close"], label="价格", color="#1f77b4", linewidth=1.5)
    ax1.plot(df["Date"], df["MA5"], label="5日均线", color="#ff7f0e", linewidth=1)
    ax1.plot(df["Date"], df["MA20"], label="20日均线", color="#2ca02c", linewidth=1)
    ax1.set_title("苹果公司股票价格分析")
    ax1.legend(loc="upper left")

    ax2 = plt.subplot(gs[1])
    ax2.bar(df["Date"], df["Volume"], label="成交量", color="#1f77b4")
    ax2.set_title("成交量")
    ax2.set_ylabel("成交量")

    ax3 = plt.subplot(gs[2])
    ax3.plot(df["Date"], df["Volatility"], label="波动率", color="#1f77b4")
    ax3.set_title("波动率")
    ax3.set_ylabel("波动率")

    plt.tight_layout()
    plt.savefig("stock_analysis.png")
    plt.show()

def portfolio_analysis():
    \"\"\"投资组合分析\"\"\"
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "Arial"
    })

    # 获取几只股票的数据
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    data = pd.DataFrame()

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start="2020-01-01", end="2021-01-01")
            data[ticker] = df["Close"]
        except Exception as e:
            print(f"获取{ticker}数据失败: {e}")

    # 计算投资组合价值
    weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    returns = data.pct_change()

    # 计算投资组合收益率和风险
    portfolio_return = (returns * weights).sum(axis=1)
    portfolio_value = 100000 * (1 + portfolio_return).cumprod()

    plt.figure(figsize=(12, 8))

    gs = plt.GridSpec(2, 2, height_ratios=[2, 1])

    ax1 = plt.subplot(gs[0, :])
    for ticker in tickers:
        ax1.plot(data.index, data[ticker]/data[ticker].iloc[0], label=ticker)
    ax1.plot(data.index, portfolio_value/portfolio_value.iloc[0], label="投资组合", linewidth=2)
    ax1.set_title("投资组合分析")
    ax1.legend(loc="upper left")

    ax2 = plt.subplot(gs[1, 0])
    sns.heatmap(returns.corr(), annot=True, cmap="RdYlGn", center=0)
    ax2.set_title("相关性矩阵")

    ax3 = plt.subplot(gs[1, 1])
    ax3.scatter(returns.std() * np.sqrt(252), returns.mean() * 252, s=100, color="#1f77b4")

    for i, ticker in enumerate(tickers):
        ax3.annotate(ticker, (returns.std()[i] * np.sqrt(252), returns.mean()[i] * 252),
                    textcoords="offset points", xytext=(10, 5), fontsize=10)

    ax3.set_title("风险收益分析")
    ax3.set_xlabel("风险 (波动率)")
    ax3.set_ylabel("收益 (年化收益率)")

    plt.tight_layout()
    plt.savefig("portfolio_analysis.png")
    plt.show()

def risk_return_analysis():
    \"\"\"风险收益分析\"\"\"
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "Arial"
    })

    # 获取几只股票的数据
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    data = pd.DataFrame()

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start="2020-01-01", end="2021-01-01")
            data[ticker] = df["Close"]
        except Exception as e:
            print(f"获取{ticker}数据失败: {e}")

    returns = data.pct_change()

    # 计算风险和收益
    mean_returns = returns.mean() * 252
    std_returns = returns.std() * np.sqrt(252)

    plt.figure(figsize=(10, 6))
    plt.scatter(std_returns, mean_returns, s=100, color="#1f77b4")

    # 添加标签
    for i, ticker in enumerate(tickers):
        plt.annotate(ticker, (std_returns[i], mean_returns[i]),
                    textcoords="offset points", xytext=(10, 5), fontsize=10)

    plt.title("风险收益分析")
    plt.xlabel("风险 (波动率)")
    plt.ylabel("收益 (年化收益率)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("risk_return.png")
    plt.show()

def correlation_analysis():
    \"\"\"相关性分析\"\"\"
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "Arial"
    })

    # 获取几只股票的数据
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    data = pd.DataFrame()

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start="2020-01-01", end="2021-01-01")
            data[ticker] = df["Close"]
        except Exception as e:
            print(f"获取{ticker}数据失败: {e}")

    returns = data.pct_change()

    plt.figure(figsize=(10, 8))
    sns.heatmap(returns.corr(), annot=True, cmap="RdYlGn", center=0, square=True)
    plt.title("股票收益率相关性")

    plt.tight_layout()
    plt.savefig("correlation_matrix.png")
    plt.show()

# 使用示例
if __name__ == "__main__":
    print("股票价格分析")
    stock_price_analysis()

    print("投资组合分析")
    portfolio_analysis()

    print("风险收益分析")
    risk_return_analysis()

    print("相关性分析")
    correlation_analysis()
""",
        "explanation": "此代码演示了金融数据分析可视化的实战案例，包括股票价格分析、投资组合分析、风险收益分析和相关性分析。这些案例展示了如何使用可视化技术帮助理解金融数据和做出投资决策。"
    },
    {
        "topic_id": 10,
        "category_id": 6,
        "title": "完整的数据可视化系统",
        "code": """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta

class FinancialVisualizationSystem:
    \"\"\"完整的金融数据可视化系统\"\"\"

    def __init__(self):
        self.data = {}
        self.results = {}

    def fetch_data(self, tickers, start_date, end_date):
        \"\"\"获取数据\"\"\"
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date)
                self.data[ticker] = df
                print(f"成功获取{ticker}数据")
            except Exception as e:
                print(f"获取{ticker}数据失败: {e}")
                self.data[ticker] = pd.DataFrame()

    def preprocess_data(self):
        \"\"\"预处理数据\"\"\"
        for ticker, df in self.data.items():
            if not df.empty:
                df["Date"] = df.index
                df["Return"] = df["Close"].pct_change()
                df["Volatility"] = df["Return"].rolling(window=20).std() * np.sqrt(252)
                df["MA5"] = df["Close"].rolling(window=5).mean()
                df["MA20"] = df["Close"].rolling(window=20).mean()

    def create_stock_plots(self):
        \"\"\"创建股票图表\"\"\"
        for ticker, df in self.data.items():
            if not df.empty:
                plt.figure(figsize=(12, 8))

                gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])

                ax1 = plt.subplot(gs[0])
                ax1.plot(df["Date"], df["Close"], label="价格", color="#1f77b4", linewidth=1.5)
                ax1.plot(df["Date"], df["MA5"], label="5日均线", color="#ff7f0e", linewidth=1)
                ax1.plot(df["Date"], df["MA20"], label="20日均线", color="#2ca02c", linewidth=1)
                ax1.set_title(f"{ticker}股票价格分析")
                ax1.legend(loc="upper left")

                ax2 = plt.subplot(gs[1])
                ax2.bar(df["Date"], df["Volume"], label="成交量", color="#1f77b4")
                ax2.set_title("成交量")
                ax2.set_ylabel("成交量")

                ax3 = plt.subplot(gs[2])
                ax3.plot(df["Date"], df["Volatility"], label="波动率", color="#1f77b4")
                ax3.set_title("波动率")
                ax3.set_ylabel("波动率")

                plt.tight_layout()
                plt.savefig(f"{ticker}_analysis.png")
                plt.close()

                print(f"已保存{ticker}分析图表")

    def create_portfolio_plots(self):
        \"\"\"创建投资组合图表\"\"\"
        tickers = list(self.data.keys())

        # 准备数据
        returns = pd.DataFrame()

        for ticker in tickers:
            if not self.data[ticker].empty:
                returns[ticker] = self.data[ticker]["Return"]

        if len(returns.columns) > 0:
            # 计算投资组合价值
            weights = np.array([1/len(tickers) for _ in range(len(tickers))])
            portfolio_return = (returns * weights).sum(axis=1)
            initial_investment = 100000
            portfolio_value = initial_investment * (1 + portfolio_return).cumprod()

            # 计算风险和收益
            mean_returns = returns.mean() * 252
            std_returns = returns.std() * np.sqrt(252)

            plt.figure(figsize=(12, 8))

            gs = plt.GridSpec(2, 2, height_ratios=[2, 1])

            ax1 = plt.subplot(gs[0, :])
            for ticker in tickers:
                if not self.data[ticker].empty:
                    ax1.plot(self.data[ticker]["Date"], self.data[ticker]["Close"]/self.data[ticker]["Close"].iloc[0],
                            label=ticker)
            ax1.plot(self.data[tickers[0]]["Date"], portfolio_value/portfolio_value.iloc[0],
                    label="投资组合", linewidth=2)
            ax1.set_title("投资组合分析")
            ax1.legend(loc="upper left")

            ax2 = plt.subplot(gs[1, 0])
            sns.heatmap(returns.corr(), annot=True, cmap="RdYlGn", center=0)
            ax2.set_title("相关性矩阵")

            ax3 = plt.subplot(gs[1, 1])
            ax3.scatter(std_returns, mean_returns, s=100, color="#1f77b4")

            for i, ticker in enumerate(tickers):
                ax3.annotate(ticker, (std_returns[i], mean_returns[i]),
                            textcoords="offset points", xytext=(10, 5), fontsize=10)

            ax3.set_title("风险收益分析")
            ax3.set_xlabel("风险 (波动率)")
            ax3.set_ylabel("收益 (年化收益率)")

            plt.tight_layout()
            plt.savefig("portfolio_analysis.png")
            plt.close()

            print("已保存投资组合分析图表")

    def create_summary_report(self):
        \"\"\"创建汇总报告\"\"\"
        if not self.data:
            print("无数据可生成报告")
            return

        report = []

        report.append("# 金融数据可视化系统报告")
        report.append("## 数据获取")
        report.append(f"- 时间范围: {list(self.data.values())[0]['Date'].iloc[0].strftime('%Y-%m-%d')} 至 {list(self.data.values())[0]['Date'].iloc[-1].strftime('%Y-%m-%d')}")
        report.append(f"- 股票数量: {len(self.data)}")

        report.append("## 主要结果")

        for ticker, df in self.data.items():
            if not df.empty:
                max_price = df["Close"].max()
                min_price = df["Close"].min()
                mean_volume = df["Volume"].mean()

                report.append(f"### {ticker}")
                report.append(f"- 最高价格: ${max_price:.2f}")
                report.append(f"- 最低价格: ${min_price:.2f}")
                report.append(f"- 平均成交量: {mean_volume:.0f}")

        # 保存报告
        with open("financial_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(report))

        print("已生成报告: financial_report.md")

    def run(self, tickers, start_date, end_date):
        \"\"\"运行整个系统\"\"\"
        print("开始金融数据可视化分析")

        self.fetch_data(tickers, start_date, end_date)

        if any(df.empty for df in self.data.values()):
            print("数据获取失败，无法继续")
            return

        self.preprocess_data()

        print("创建股票图表")
        self.create_stock_plots()

        print("创建投资组合图表")
        self.create_portfolio_plots()

        print("创建汇总报告")
        self.create_summary_report()

        print("分析完成")

# 使用示例
if __name__ == "__main__":
    system = FinancialVisualizationSystem()

    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    system.run(tickers, start_date, end_date)
""",
        "explanation": "此代码演示了完整的金融数据可视化系统，包括数据获取、预处理、图表生成和报告生成。该系统提供了一个完整的架构，可以根据需要进行扩展和改进。"
    }
]

@app.route("/")
def index():
    return render_template("index.html", topics=FINANCIAL_TOPICS)

@app.route("/topic/<int:topic_id>")
def topic_detail(topic_id):
    topic = next((t for t in FINANCIAL_TOPICS if t["id"] == topic_id), None)
    if not topic:
        return "主题未找到", 404
    return render_template("topic_detail.html", topic=topic, categories=TOPIC_CATEGORIES)

@app.route("/topic/<int:topic_id>/category/<int:category_id>")
def category_detail(topic_id, category_id):
    topic = next((t for t in FINANCIAL_TOPICS if t["id"] == topic_id), None)
    category = next((c for c in TOPIC_CATEGORIES if c["id"] == category_id), None)

    if not topic or not category:
        return "页面未找到", 404

    examples = [e for e in EXAMPLE_CODES if e["topic_id"] == topic_id and e["category_id"] == category_id]

    return render_template("category_detail.html", topic=topic, category=category, examples=examples)

if __name__ == "__main__":
    app.run(debug=True)
else:
    # For Vercel serverless deployment
    application = app
