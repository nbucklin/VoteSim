import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
plt.style.use('seaborn')
scaler = MinMaxScaler()

# Setting up parameters 
sim_num = 200
weeks = 44
random_national = .002
random_state = .001

# Reading in the data
data_states = pd.read_csv(r'/Users/Natalie/Desktop/Priorities USA/Materials/state_forecasts.csv')
    
# Making a simulation grid for the national swing. It is impossible to enforce the same
# national swing across all states within the next loop since I'm looping through on a 
# state by state basis. For each state, I will look up the same national swing values based on
# the week number and simulation number. 
var_sim = pd.DataFrame()
for x in range(sim_num):
    var_list = []
    t = weeks
    for t in range(t):
        # Pulling from normal distribution
        var = np.random.normal(0,random_national)
        var_list.append(var)
        var_df = pd.DataFrame(var_list).T
    var_sim = var_sim.append(var_df)
var_sim = var_sim.reset_index(drop=True)

state_sim = {}
for index,row in data_states.iterrows():
    state = data_states['code']
    sim = pd.DataFrame()
    for x in range(sim_num):
        # Calculating the high and low confidence intervals
        vote_high = row['dem_two_way'] + (1.96 * row['sd'])
        vote_low = row['dem_two_way'] - (1.96 * row['sd'])
        # Pulling from the confidence interval. I used a uniform distribution but am unsure if it should be normal.
        vote = np.random.uniform(low=vote_low,high=vote_high)
        rs = random_state
        t = weeks
        sim_list = []
        for t in range(t):
            # Calculating simulated vote for a given week based on the vote, state swing, and national swing
            sim_vote = vote + np.random.normal(0,rs) + var_sim.iloc[x,t]
            sim_list.append(sim_vote)
            vote = sim_vote
            sim_df = pd.DataFrame(sim_list).T
        sim = sim.append(sim_df)
    sim = sim.reset_index(drop=True)
    state_sim[row['code']] = sim

# Selecting the last week of simulated vote share for each simulation
result_sim = pd.DataFrame()
for key, value in state_sim.items():
    result_df = pd.DataFrame(state_sim[key].loc[:,43]).T
    result_df.rename(index={43:key},inplace=True)
    result_sim = result_sim.append(result_df)
 
# Calcuating probabilities of seeing vote shares above 48%, 50%, and 52%
result_sim['average'] = result_sim.mean(axis=1)
result_sim['p_48'] = (result_sim.iloc[:,0:(sim_num-1)] > .48).sum(axis=1) / sim_num
result_sim['p_50'] = (result_sim.iloc[:,0:(sim_num-1)] > .5).sum(axis=1) / sim_num
result_sim['p_52'] = (result_sim.iloc[:,0:(sim_num-1)] > .52).sum(axis=1) / sim_num
result_sim['diff'] = abs(result_sim['average'] - .5)
result_sim = result_sim.sort_values(by=['average'])

# Plotting
plt.figure(figsize=(14,5))
plt.grid('on')
plt.axhspan(52,48,color='r',alpha=.5)
plt.xticks(rotation=45)
plt.ylabel("Democratic Vote Share %",weight='bold')
plt.title("Simulated Democratic Vote Share",weight='bold')
plt.scatter(result_sim.index,y=(result_sim['average']* 100))   
plt.savefig(r'/Users/Natalie/Desktop/Priorities USA/State Sim.svg',format='svg',dpi=1000)
plt.show()

result_target = result_sim[(result_sim['p_50'] <.95) & (result_sim['p_50'] > 0 )]
result_target = result_target.sort_values(by=['diff'],ascending=False)
result_target['p_50'] = result_target['p_50'] * 100
result_target['diff'] = result_target['diff'] * 100

result_target = result_target.sort_values(by=['p_50'],ascending=False)
ax1 = result_target['diff'].plot(secondary_y=True,color='r')
ax2 = result_target['p_50'].plot(kind='bar')
ax1.set_ylabel('Mean Difference from 50% (line)',weight='bold')
ax2.set_ylabel('Probability Greater than 50% (bar)',weight='bold')
plt.title("Probability of Winning vs Simulated Difference",weight='bold')
plt.savefig(r'/Users/Natalie/Desktop/Priorities USA/Target States.svg',format='svg',dpi=100)
plt.figure(figsize=(10,5))
plt.show()

# Attempting to come up with a score by scaling the p_50 and the electoral votes from 0 to 1
result_final = result_target[(result_target['p_50'] > 20) & (result_target['p_50'] < 70)]
result_final['code'] = result_final.index
result_final = pd.merge(result_final,data_states[['code','votes']],how='inner',on='code')
result_final['score'] = scaler.fit_transform(result_final[['votes']]) + scaler.fit_transform(result_final[['p_50']])
