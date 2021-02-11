import pandas as pd
import numpy as np
import os
import requests
import json
import datetime
import time
import tqdm
pd.set_option('display.max_rows', 100)
##

BUFFER = 1

base_url = "https://www.kaggle.com/requests/EpisodeService/"
get_url = base_url + "GetEpisodeReplay"
list_url = base_url + "ListEpisodes"

th_min = 1100
th_max = 1300
rank_min = 1
rank_max = 30
##

# inital team list

# r = requests.post(list_url, json = {"submissionId":  18998567}) # ID is sample value
r = requests.post(list_url, json = {"submissionId":  18703109}) # ID is sample value

rj = r.json()

teams_df = pd.DataFrame(rj['result']['teams'])
teams_df = teams_df.sort_values('publicLeaderboardRank').reset_index().drop(["index"],axis = 1)
# sids = teams_df["publicLeaderboardSubmissionId"].head(20).copy()
sids = [18806907,18778644,18998567,19191552,19003099,19218541,19097985]
teams_df.head(70)
##

for sid in sids:
    time.sleep(2)
#     print(sid,type(int(sid)))
    r = requests.post(list_url, json = {"submissionId":  int(sid)}) # ID is sample value
    rj = r.json()

    teams_df_t = pd.DataFrame(rj['result']['teams'])
    teams_df = teams_df.append(teams_df_t)
    teams_df = teams_df.drop_duplicates('id')
teams_df = teams_df.sort_values('publicLeaderboardRank').reset_index().drop(["index"],axis = 1)
##

def log_training(result, n_machines):
    """Records training data from each machine, each agent, each round
    
    Generates a training dataset to support prediction of the current
    payout ratio for a given machine.
    
    Args:
       result ([[dict]]) - output from all rounds provided as output of 
                           env.run([agent1, agent2])
       n_machines (int) - number of machines
                           
    Returns:
       training_data (pd.DataFrame) - training data, including:
           "round_num"      : round number
           "machine_id"     : machine data applies to
           "agent_id"       : player data applies to (0 or 1)
           "n_pulls_self"   : number of pulls on this machine so far by agent_id
           "n_success_self" : number of rewards from this machine by agent_id
           "n_pulls_opp"    : number of pulls on this machine by the other player
           "success_rate"
           "payout"         : actual payout ratio for this machine
    
    """
    # Initialize machine and agent states
    machine_state = [{'n_pulls_0': 0, 'n_success_0': 0,
                      'n_pulls_1': 0, 'n_success_1': 0,
                      'payout': None}
                     for ii in range(n_machines)]
    agent_state = {'reward_0': 0, 'reward_1': 0, 'last_reward_0': 0,'last_reward_1': 0}

    # Initialize training dataframe
    # - In the first round, store records for all n_machines
    # - In subsequent rounds, just store the two machines that updated
    training_data = pd.DataFrame(
            index=range(n_machines + 4 * (len(result) - 1)),
            columns=['round_num', 'machine_id', 'agent_id',
                     'n_pulls_self', 'n_success_self','n_pulls_opp', 'success_rate','payout'])
    
    # Log training data from each round
    for round_num, res in enumerate(result):
        # Get current threshold values
        thresholds = res[0]['observation']['thresholds']

        # Update agent state
        for agent_ii in range(2):
            agent_state['last_reward_%i' % agent_ii] = (res[agent_ii]['reward']- agent_state['reward_%i' % agent_ii])
            agent_state['reward_%i' % agent_ii] = res[agent_ii]['reward']        

        # Update most recent machine state
        if res[0]['observation']['lastActions']:
            for agent_ii, r_obs in enumerate(res):
                action = r_obs['action']
                machine_state[action]['n_pulls_%i' % agent_ii] += 1
                machine_state[action]['n_success_%i' % agent_ii] += agent_state['last_reward_%i' % agent_ii]
                machine_state[action]['payout'] = thresholds[action]
        else:
            # Initialize machine states
            for mach_ii in range(n_machines):
                machine_state[mach_ii]['payout'] = thresholds[mach_ii]
            
        # Record training records
        # -- Each record includes:
        #       round_num, n_pulls_self, n_success_self, n_pulls_opp
        if res[0]['observation']['lastActions']:
            # Add results for most recent moves
            for agent_ii, r_obs in enumerate(res):
                action = r_obs['action']

                # Add row for agent who acted
                row_ii = n_machines + 4 * (round_num - 1) + 2 * agent_ii 
                training_data.at[row_ii, 'round_num'] = round_num
                training_data.at[row_ii, 'machine_id'] = action
                training_data.at[row_ii, 'agent_id'] = agent_ii
                training_data.at[row_ii, 'n_pulls_self'] = (machine_state[action]['n_pulls_%i' % agent_ii])
                training_data.at[row_ii, 'n_success_self'] = (machine_state[action]['n_success_%i' % agent_ii])
                training_data.at[row_ii, 'n_pulls_opp'] = (machine_state[action]['n_pulls_%i' % ((agent_ii + 1) % 2)])
                training_data.at[row_ii, 'success_rate'] = training_data.at[row_ii, 'n_success_self'] /training_data.at[row_ii, 'n_pulls_self'] if training_data.at[row_ii, 'n_pulls_self']>0 else np.nan
                training_data.at[row_ii, 'payout'] = (machine_state[action]['payout'] / 100)

                # Add row for other agent
                row_ii = n_machines + 4 * (round_num - 1) + 2 * agent_ii + 1
                other_agent = (agent_ii + 1) % 2
                training_data.at[row_ii, 'round_num'] = round_num
                training_data.at[row_ii, 'machine_id'] = action
                training_data.at[row_ii, 'agent_id'] = other_agent
                training_data.at[row_ii, 'n_pulls_self'] = (machine_state[action]['n_pulls_%i' % other_agent])
                training_data.at[row_ii, 'n_success_self'] = (machine_state[action]['n_success_%i' % other_agent])
                training_data.at[row_ii, 'n_pulls_opp'] = (machine_state[action]['n_pulls_%i' % agent_ii])
                training_data.at[row_ii, 'success_rate'] = training_data.at[row_ii, 'n_success_self'] /training_data.at[row_ii, 'n_pulls_self'] if training_data.at[row_ii, 'n_pulls_self']>0 else np.nan
                training_data.at[row_ii, 'payout'] = (machine_state[action]['payout'] / 100)
                
#         else:
            # Add initial data for all machines
            for action in range(n_machines):
                row_ii = action
                training_data.at[row_ii, 'round_num'] = round_num
                training_data.at[row_ii, 'machine_id'] = action
                training_data.at[row_ii, 'agent_id'] = -1
                training_data.at[row_ii, 'n_pulls_self'] = 0
                training_data.at[row_ii, 'n_success_self'] = 0
                training_data.at[row_ii, 'n_pulls_opp'] = 0
                training_data.at[row_ii, 'success_rate'] = np.nan
                training_data.at[row_ii, 'payout'] = (machine_state[action]['payout'] / 100)
            
    return training_data
##
    
# Create results
training_data = []

for subid in range(rank_min-1,rank_max):
    sub = teams_df["publicLeaderboardSubmissionId"][subid]
    start_time = datetime.datetime.now()
    r = BUFFER;
    result = requests.post(list_url, json = {"submissionId":  int(sub)})
    team_json = result.json()
    team_df = pd.DataFrame(team_json['result']['episodes'])
    print('{} - {} games for {} ({})'.format(subid,len(team_df), sub,teams_df["teamName"][subid]))
    team_df = team_df.head(50)

    for i in tqdm.tqdm(range(len(team_df))):
        epid = team_df.id.iloc[i]
        ps = [r for r in team_json['result']['episodes'] if r['id']==epid][0]['agents']
        p1 = ps[0]['initialScore']
        p2 = ps[1]['initialScore'] 
        if p1 is not None and p2 is not None and p1 > th_min and p2 > th_min and p1 < th_max and p2 < th_max:
            re = requests.post(get_url, json = {"EpisodeId": int(epid)})
            if re.headers.get('content-type') is not None and 'json' in re.headers.get('content-type'):
                jstr = re.json()['result']['replay']
                jj = json.loads(jstr)
                result = jj['steps']
                if(len(result)) == 2000:
                    training_data.append(log_training(result, len(result[0][0]['observation']['thresholds'])))
            #         print('agent 1 : %i, agent 2 : %i' % ( result[-1][0]['reward'], result[-1][1]['reward']))

# Save training data
training_data = pd.concat(training_data, axis=0)
training_data.to_parquet('training_data_lb.parquet', index=False)    
