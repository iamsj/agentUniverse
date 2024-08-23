import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('E:\\SynologyDrive\\mba\\论文\\数据\\上证主非ST非金融非公共事业测试数据\\股市因子\\TRD_Dalyr.xlsx')

# 转换交易日期为日期格式
if 'Trddt' in df.columns:
    df['Trddt'] = pd.to_datetime(df['Trddt'], errors='coerce')
else:
    print("Column 'Trddt' not found in the DataFrame.")

# 提取2023年8月15日和12月27日的数据
df_815 = df[df['Trddt'] == '2023-08-15'][['Stkcd', 'Clsprc']]
df_1227 = df[df['Trddt'] == '2023-12-27'][['Stkcd', 'Clsprc']]

# 重命名列名以便后续合并
df_815 = df_815.rename(columns={'Clsprc': '8/15 Close Price'})
df_1227 = df_1227.rename(columns={'Clsprc': '12/27 Close Price'})

# 按证券代码合并两天的数据
result_df = pd.merge(df_815, df_1227, on='Stkcd', how='outer')

# 输出到新的 Excel 文件
result_df.to_excel('E:\\SynologyDrive\\mba\\论文\\数据\\上证主非ST非金融非公共事业测试数据\\股市因子\\output_file.xlsx',
                   index=False)
