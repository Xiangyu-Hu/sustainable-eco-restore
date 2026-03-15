from pysd import read_vensim
model = read_vensim("Vensim System Dynamics Model.mdl")
print(model.doc)
result = model.run()
print(result.head())
print(result.tail())
result.to_csv("Vensim_System_Dynamics_Model_output.csv", index=False)

print("Finished. Output saved to Vensim_System_Dynamics_Model_output.csv")