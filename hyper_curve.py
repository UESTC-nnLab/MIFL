import matplotlib.pyplot as plt

# Hyper
x = [4,8,16,25,50,100]
betanmi = [7.13, 25.88,	34.76, 44.02, 50.15, 53.58] #DAUB
gamanmi = [2.81, 18.40,	25.44, 37.25, 50.56, 57.63] #ITSDT

# 创建图形和坐标轴对象“"
# fig, ax = plt.subplots()
plt.figure(figsize=(7, 5))

# 绘制蓝色折线和圆形数据点
plt.plot(x, betanmi, color='blue', marker='o', markersize=7, linewidth=2.5, label='with prompt')
# 绘制红色折线和正方形数据点
plt.plot(x, gamanmi, color='red', marker="s", markersize=6, linewidth=2.5, label='without prompt')



# 设置 x 轴和 y 轴标题
plt.ylabel('mAP$_{50}$ (%)') #plt.ylabel('Metric (%)', fontsize=20)
plt.xlabel('The numbers of shots')

plt.xticks([4,8,16,25,50,100],["4", "8", "16", "25","50","100"])#,fontsize=20
plt.yticks([0,10,20,30,40,50,60,70],["0","10","20","30","40","50","60","70"]) #,fontsize=20
plt.grid(True,linestyle='--')

# 添加图例
plt.legend(loc='upper right', ncol=2, ) #,fontsize=20 bbox_to_anchor=(1, 0.9),
plt.title('Results on ITSDT')

# 显示图形
plt.savefig(f'channel.jpg', format='jpg')

# 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中