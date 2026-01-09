import pysd
import matplotlib.pyplot as plt

print
model = pysd.read_vensim("Degradation.mdl")
print
result = model.run()
print
# print(result.columns.tolist())

time = result.index
material = result["Material collected"]
sand = result["Collecting sand"]

plt.figure(figsize=(8,5))
plt.plot(time, material, label="Material collected (ton)", color="blue")
plt.plot(time, sand, label="Sand collected (ton)", color="orange")

plt.xlabel("Month")
plt.ylabel("Tons")
plt.title("Material and Sand Collected Over Time")
plt.legend()
plt.grid(True)
plt.savefig("material_sand.png", dpi=300)
plt.show()


