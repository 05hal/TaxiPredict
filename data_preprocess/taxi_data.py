import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ====================================
# 图像风格设置
# ====================================
sns.set_theme(
    style="whitegrid",
    palette="deep",
    context="talk"
)

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

# 去掉顶部和右侧边框
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# 字体大小
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# ====================================
# 1. 读取 Uber 数据
# ====================================
print("读取 Uber 数据...")

may = pd.read_csv("data/uber-raw-data-may14.csv")
jun = pd.read_csv("data/uber-raw-data-jun14.csv")

# 合并两个月
df = pd.concat([may, jun], ignore_index=True)

print("Uber 数据读取完成")
print(df.head())
print(df.shape)

# ====================================
# 2. 时间处理
# ====================================
print("处理时间字段...")

df['Date/Time'] = pd.to_datetime(df['Date/Time'])

# 小时
df['hour'] = df['Date/Time'].dt.hour

# 星期
df['weekday'] = df['Date/Time'].dt.weekday

# 日期
df['day'] = df['Date/Time'].dt.day

# 月份
df['month'] = df['Date/Time'].dt.month

# 是否周末
df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

print("时间特征完成")

# ====================================
# 3. 经纬度清洗
# ====================================
print("清洗经纬度数据...")

df = df[
    (df['Lat'] > 40.0) &
    (df['Lat'] < 41.0) &
    (df['Lon'] > -75.0) &
    (df['Lon'] < -73.0)
]

print("清洗后数据量:", len(df))

# ====================================
# 4. 空间网格划分
# ====================================
print("进行空间网格划分...")

GRID_SIZE = 0.01

df['lat_grid'] = (df['Lat'] / GRID_SIZE).astype(int)
df['lon_grid'] = (df['Lon'] / GRID_SIZE).astype(int)

# 区域 ID
df['region_id'] = (
    df['lat_grid'].astype(str)
    + "_"
    + df['lon_grid'].astype(str)
)

print("区域数量:", df['region_id'].nunique())

# ====================================
# 5. 构建时间窗口
# ====================================
print("构建时间窗口...")

df['time_bin'] = df['Date/Time'].dt.floor('h')

# ====================================
# 6. 聚合订单量
# ====================================
print("统计订单量...")

traffic = (
    df.groupby([
        'time_bin',
        'region_id',
        'hour',
        'weekday',
        'is_weekend'
    ])
    .size()
    .reset_index(name='order_count')
)

print("聚合完成")
print(traffic.head())

# ====================================
# 7. 读取天气数据
# ====================================
print("读取天气数据...")

weather = pd.read_csv("data/nyc-weather-data.csv")

print(weather.head())

# ====================================
# 8. 处理天气数据
# ====================================
print("处理天气字段...")

# NOAA 日期格式转换
weather['DATE'] = pd.to_datetime(
    weather['DATE'],
    format='%Y%m%d'
)

# 保留需要字段
weather = weather[
    [
        'DATE',
        'PRCP',
        'TMAX',
        'TMIN',
        'SNOW',
        'AWND'
    ]
]

# 重命名
weather.columns = [
    'date',
    'precipitation',
    'temp_max',
    'temp_min',
    'snow',
    'avg_wind'
]

# NOAA 温度单位转换
weather['temperature'] = (
    weather['temp_max'] +
    weather['temp_min']
) / 20

print(weather.head())

# ====================================
# 9. 融合天气数据
# ====================================
print("融合天气数据...")

traffic['date'] = traffic['time_bin'].dt.date
weather['date'] = weather['date'].dt.date

traffic = traffic.merge(
    weather,
    on='date',
    how='left'
)

print("融合完成")
print(traffic.head())

# ====================================
# 10. 缺失值处理
# ====================================
print("处理缺失值...")

traffic = traffic.fillna(0)

# ====================================
# 11. 保存最终 CSV
# ====================================
output_path = "output/processed_traffic_data.csv"

traffic.to_csv(
    output_path,
    index=False
)

print("CSV 保存完成")

# ====================================
# 12. 每小时订单量图
# ====================================
print("生成：每小时订单量图")

hourly_orders = (
    df.groupby('hour')
    .size()
    .reset_index(name='orders')
)

plt.figure(figsize=(12, 6))

sns.lineplot(
    data=hourly_orders,
    x='hour',
    y='orders',
    marker='o',
    linewidth=3,
    markersize=8
)

plt.fill_between(
    hourly_orders['hour'],
    hourly_orders['orders'],
    alpha=0.3
)

plt.title(
    "Uber Orders by Hour",
    fontsize=20,
    fontweight='bold'
)

plt.xlabel(
    "Hour of Day",
    fontsize=14
)

plt.ylabel(
    "Number of Orders",
    fontsize=14
)

plt.xticks(range(24))

plt.tight_layout()

plt.savefig(
    "output/hourly_orders.png"
)

plt.close()

# ====================================
# 13. 星期订单量图
# ====================================
print("生成：星期订单量图")

weekday_map = {
    0: 'Mon',
    1: 'Tue',
    2: 'Wed',
    3: 'Thu',
    4: 'Fri',
    5: 'Sat',
    6: 'Sun'
}

df['weekday_name'] = df['weekday'].map(weekday_map)

weekday_orders = (
    df.groupby('weekday_name')
    .size()
    .reset_index(name='orders')
)

weekday_orders['weekday_name'] = pd.Categorical(
    weekday_orders['weekday_name'],
    categories=[
        'Mon',
        'Tue',
        'Wed',
        'Thu',
        'Fri',
        'Sat',
        'Sun'
    ],
    ordered=True
)

weekday_orders = weekday_orders.sort_values(
    'weekday_name'
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=weekday_orders,
    x='weekday_name',
    y='orders'
)

plt.title(
    "Uber Orders by Weekday",
    fontsize=20,
    fontweight='bold'
)

plt.xlabel(
    "Weekday",
    fontsize=14
)

plt.ylabel(
    "Number of Orders",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "output/weekday_orders.png"
)

plt.close()

# ====================================
# 14. 温度与订单关系图
# ====================================
print("生成：温度与订单关系图")

temp_orders = (
    traffic.groupby('temperature')['order_count']
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=temp_orders,
    x='temperature',
    y='order_count',
    marker='o',
    linewidth=3,
    markersize=8
)

sns.regplot(
    data=temp_orders,
    x='temperature',
    y='order_count',
    scatter_kws={'s':60},
    line_kws={'linewidth':3}
)

plt.title(
    "Temperature vs Average Orders",
    fontsize=20,
    fontweight='bold'
)

plt.xlabel(
    "Temperature (°C)",
    fontsize=14
)

plt.ylabel(
    "Average Order Count",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "output/temperature_orders.png"
)

plt.close()

# ====================================
# 15. 降雨与订单关系图
# ====================================
print("生成：降雨与订单关系图")

rain_orders = (
    traffic.groupby('precipitation')['order_count']
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 6))

sns.lineplot(
    data=rain_orders,
    x='precipitation',
    y='order_count',
    marker='o',
    linewidth=3,
    markersize=8
)

plt.title(
    "Precipitation vs Average Orders",
    fontsize=20,
    fontweight='bold'
)

plt.xlabel(
    "Precipitation",
    fontsize=14
)

plt.ylabel(
    "Average Order Count",
    fontsize=14
)

plt.tight_layout()

plt.savefig(
    "output/precipitation_orders.png"
)

plt.close()

# ====================================
# 16. 热门区域 Top10
# ====================================
print("生成：热门区域图")

top_regions = (
    traffic.groupby('region_id')['order_count']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

# 简化区域名称
top_regions['short_region'] = [
    f'R{i+1}'
    for i in range(len(top_regions))
]

plt.figure(figsize=(14, 7))

# 使用渐变色
colors = sns.color_palette(
    "Blues_r",
    len(top_regions)
)

ax = sns.barplot(
    data=top_regions,
    x='short_region',
    y='order_count',
    hue='short_region',
    palette=colors,
    legend=False
)
# 在柱子顶部显示数值
for i, row in top_regions.iterrows():

    ax.text(
        i,
        row['order_count'] + 500,
        f"{int(row['order_count'])}",
        ha='center',
        fontsize=11,
        fontweight='bold'
    )

plt.title(
    "Top 10 High Demand Regions",
    fontsize=22,
    fontweight='bold',
    pad=20
)

plt.xlabel(
    "Region",
    fontsize=15
)

plt.ylabel(
    "Total Orders",
    fontsize=15
)

# 去掉上边框和右边框
sns.despine()

plt.tight_layout()

plt.savefig(
    "output/top_regions.png"
)

plt.close()

# ====================================
# 17. 最终输出
# ====================================
print("====================================")
print("数据处理完成")
print("输出文件:")
print("1. processed_traffic_data.csv")
print("2. hourly_orders.png")
print("3. weekday_orders.png")
print("4. temperature_orders.png")
print("5. precipitation_orders.png")
print("6. top_regions.png")
print("最终数据形状:", traffic.shape)
print("====================================")