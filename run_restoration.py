import pysd
import matplotlib.pyplot as plt

print("✅ Python 脚本开始执行")

# 读取 Restoration 模型
model = pysd.read_vensim("Restoration.mdl")
print("✅ Restoration 模型加载完成")

# 运行模型
result = model.run()
print("✅ Restoration 模型运行完成")

# 保存 CSV 文件
result.to_csv("restoration_result.csv", index=True)

# 画图：画需要的两个变量
plt.plot(result["Material collected"], label="Material collected")
plt.plot(result["Collecting sand"], label="Collecting sand")
plt.xlabel("Time (Month)")
plt.ylabel("Value")
plt.legend()

# 保存和显示图像
plt.savefig("restoration_result.png", dpi=300)
plt.show()

