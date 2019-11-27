import pandas as pd 
import numpy as np 

'''
Recombines results back intno thehir appropriate positions.
Only set up for myVolts as that was the only company which we founds ones for.
'''


results = pd.read_csv('results.csv')
test = pd.read_csv('test.csv')


totalResults = pd.DataFrame(0, index=np.arange(9145), columns=['set_clicked'])

index = 0
for i in range(0, 9145):
    if (test['organization_id'].iloc[i] == 4):
        totalResults['set_clicked'].iloc[i] = results['set_clicked'].iloc[index]
        index = index + 1

print("Index reached: " , index)
print("Results size: ", results.size)
print(totalResults.size)

totalResults.to_csv('totalresults.csv', index=False, header=False)