import re,time,os,string,random,json_repair
from collections import defaultdict
import pandas as pd 
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from utils import *
from prompt_template_env import *


def student_simulation_experiment(student_id,agent_config):
    dataset = pd.read_csv(agent_config['dataset_path'])
    data_student = dataset[dataset['username']==student_id]
    sim_strategy = agent_config['sim_strategy']

    sim_config_name = agent_config['sim_config_list']
    sim_config_set = [str(agent_config[sim_item]) for sim_item in sim_config_name]
    root_folder = agent_config['result_path'] + '/' + '_'.join(sim_config_set) + '/' + str(int(student_id))
    out_sim_csv = f'{root_folder}/sim_{student_id}.csv'

    result_header = ['username', 'trial_id_global', 'user_accuracy', 'response_time', 'reasoning']
    data_result_df = pd.DataFrame(columns = result_header)

    student_simulator_agent = Student_Simulator_Agent(backbone_type = agent_config['gpt_type'], log_folder = root_folder, print_log = agent_config['print_log'], temperature = 0)

    past_table = data_student[data_student['session'] == 'calib']
    future_table = data_student[data_student['session'] == 'formal']

    future_trial_list = sorted(list(future_table['trial_id_global'].unique()))

    student_simulator_agent.fit(past_table)

    for trial_id in future_trial_list:
        future_table_trial = future_table[future_table['trial_id_global'] == trial_id]
        student_predict_item = student_simulator_agent.test_each(future_table_trial, sim_strategy)
        student_predict_item = student_predict_item[0]
        student_predict_item.update({'username': student_id})
        # Ensure column order matches result_header
        ordered_row = [student_predict_item[col] for col in result_header]
        item_df = pd.DataFrame([ordered_row], columns=result_header)
        data_result_df = pd.concat([data_result_df, item_df], ignore_index=True)
        data_result_df.to_csv(out_sim_csv, index = False)

class Student_Simulator_Agent(object):
    def __init__(self,backbone_type = 3,log_folder = './log/',print_log = True,temperature = 0):
        self.backbone_type = backbone_type
        self.agent_name = 'student_simulator_agent'
        if log_folder is None: 
            self.log_file = None
        else:
            self.log_file = f'{log_folder}/{self.agent_name}.txt'
        self.print_log = print_log
        self.temperature = temperature

        self.sys_prompt = sys_prompt_template_dict[self.agent_name]
        
        self.sys_message = {"role": "system","content": self.sys_prompt}
        self.task_prompt = task_prompt_template_dict[self.agent_name]

        if self.log_file is not None: store_log(f'{self.agent_name} agent: system: \n {self.sys_prompt}\n',self.log_file,color = 'white',print = self.print_log)

        self.message_list = [self.sys_message]
    
    def get_trial_result_str(self, data_table, context_type):
        assert context_type in ['past', 'future']
        trial_list = sorted(list(data_table['trial_id_global'].unique()))
        trial_str = ''
        for trial_id in trial_list:
            data_trial = data_table[data_table['trial_id_global'] == trial_id]
            math_question = data_trial['task_detail'].values[0]
            user_accuracy = data_trial['user_accuracy'].values[0]
            response_time = data_trial['response_time'].values[0]
            if context_type == 'past':
                trial_str += f'trial_id_global: {trial_id}, math question: {math_question}, student accuracy: {user_accuracy}, response time: {response_time};\n'
            else:
                trial_str += f'trial_id_global: {trial_id}, math question: {math_question};\n'

        return trial_str

    def fit(self, past_table):
        edu_str = past_table['education'].values[0]
        past_history_str = self.get_trial_result_str(past_table, 'past')

        past_student_report = f'''
        # education background #: {edu_str}
        # past math question answering performance #: \n\n {past_history_str}.
        '''

        future_simulation_str = f'''
        Based on the student unique educational background and past math question answering history, please mimic this student to simulate accuracy and response time in each of future math questions that I will input one by one later.

        {output_format_template_dict[self.agent_name]}
        '''

        user_prompt = f'{self.task_prompt} \n\n Now consider simulating the following student below: \n\n {past_student_report} \n\n {future_simulation_str}. '
        self.message_list.append({"role": "user","content": user_prompt})
        t1 = time.time()
        llm_response = 'Got it! Please continue the future math question one by one for my simulation.'
        self.message_list.append({"role": "assistant","content": llm_response})
        if self.log_file is not None: store_log(f'{self.agent_name} agent: user: \n {user_prompt}\n',self.log_file,color = 'green',print = self.print_log)
        if self.log_file is not None: store_log(f'{self.agent_name} agent: assistant (resptime: {time.time()-t1}): \n {llm_response}\n',self.log_file,color = 'red',print = self.print_log)

    def test_each(self, future_table_trial, model_type):
        simulation_strategy_str = simulation_strategy_str_dict_global[model_type]
        
        future_question_str = f'\n\n Now please simulate the student answering the following future math question: \n\n' + self.get_trial_result_str(future_table_trial, 'future')
        user_prompt = simulation_strategy_str + future_question_str + output_format_template_dict[self.agent_name]

        self.message_list.append({"role": "user","content": user_prompt})
        t1 = time.time()
        llm_response = response_llm_gpt(self.message_list,self.backbone_type,self.temperature)
        self.message_list.append({"role": "assistant","content": llm_response})
        if self.log_file is not None: store_log(f'{self.agent_name} agent: user: \n {user_prompt}\n',self.log_file,color = 'green',print = self.print_log)
        if self.log_file is not None: store_log(f'{self.agent_name} agent: assistant (resptime: {time.time()-t1}): \n {llm_response}\n',self.log_file,color = 'red',print = self.print_log)
        response_dict = self.extract_response_pattern(llm_response)

        if model_type == 'self-refine':
            response_dict_eval_feedback = self.action_self_feedback()
            response_dict = self.action_self_refine()

        return response_dict
    
    def action_self_feedback(self):
        user_prompt = f'Now please reflect and evaluate whether your previous simulation is good enough, based on all information above. Then give feedback about how you can make further improvements in simulations. Your output should be a json format with feedback as the key and detailed feedback text as the value.'

        self.message_list.append({"role": "user","content": user_prompt})
        t1 = time.time()
        llm_response = response_llm_gpt(self.message_list,self.backbone_type,self.temperature)
        self.message_list.append({"role": "assistant","content": llm_response})
        if self.log_file is not None: store_log(f'{self.agent_name} agent: user: \n {user_prompt}\n',self.log_file,color = 'green',print = self.print_log)
        if self.log_file is not None: store_log(f'{self.agent_name} agent: assistant (resptime: {time.time()-t1}): \n {llm_response}\n',self.log_file,color = 'red',print = self.print_log)
        response_dict = self.extract_response_pattern(llm_response)

        return response_dict
    
    def action_self_refine(self):
        user_prompt = f'Now based on your previously generated feedback and self-evaluation, please adjust and refine your previous simulations. \n\n' + output_format_template_dict[self.agent_name]

        self.message_list.append({"role": "user","content": user_prompt})
        t1 = time.time()
        llm_response = response_llm_gpt(self.message_list,self.backbone_type,self.temperature)
        self.message_list.append({"role": "assistant","content": llm_response})
        if self.log_file is not None: store_log(f'{self.agent_name} agent: user: \n {user_prompt}\n',self.log_file,color = 'green',print = self.print_log)
        if self.log_file is not None: store_log(f'{self.agent_name} agent: assistant (resptime: {time.time()-t1}): \n {llm_response}\n',self.log_file,color = 'red',print = self.print_log)
        response_dict = self.extract_response_pattern(llm_response)

        return response_dict
          
    def extract_response_pattern(self,response):
        try:
            evaluation_data = json_repair.loads(response.strip('```json').strip('```'))
            assert (isinstance(evaluation_data, list) or isinstance(evaluation_data, dict)), "Parsed response is not a list or dict"
            if isinstance(evaluation_data, dict): evaluation_data = [evaluation_data]
            return evaluation_data
           
        except Exception as e:
            print("Error parsing LLM response:", e)
            print("Raw output:\n", response)



async def run_exp_async(agent_config,user_idx_start,user_idx_end):
    demo_table_origin = pd.read_csv(agent_config['dataset_path'])
    user_id_list = list(set(demo_table_origin['username']))
    user_id_list.sort()

    print(f'All user num: {len(user_id_list)}')

    user_id_list_selected = user_id_list[user_idx_start:user_idx_end]
    
    print('user num: ',len(user_id_list_selected))

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=100)

    tasks = []
    for user_id in user_id_list_selected:
        print(f'username: {user_id}')
        demo_user = demo_table_origin[demo_table_origin['username']==user_id]
        if os.path.exists(agent_config['result_path']) == False:
            os.makedirs(agent_config['result_path'])
        
        root_folder = agent_config['result_path'] + '/' + '_'.join([str(agent_config[sim_item]) for sim_item in agent_config['sim_config_list']])
        
        if os.path.exists(root_folder) == False:
            os.makedirs(root_folder)
        if os.path.exists(root_folder + '/' + str(int(user_id))) == False:
            os.makedirs(root_folder + '/' + str(int(user_id)))
        task = loop.run_in_executor(executor, student_simulation_experiment, user_id, agent_config)
        tasks.append(task)

    await asyncio.gather(*tasks)

def exp_main():
    result_root_folder = 'result_sim/'
    dataset_type_list = ['timecare', 'bio', 'write'] # 
    dataset_dict = {d: [f'./dataset/{d}/dataset_{d}.csv'] for d in dataset_type_list}
    epoch_num_dict = {'timecare': 2, 'bio': 1, 'write': 1}
    dataset_user_total_dict = {'timecare': 79, 'bio': 21, 'write': 33}
    behavior_column_list_dict = {'timecare': ['resptime'], 'bio': ['resptime'], 'write': ['WPM','text_input']}

    dataset_name = '' # 

    simulation_config = {
        'sim_config_list': ['sim_strategy','gpt_type'],
        'dataset_path': dataset_dict[dataset_name][0],
        'dataset_type': dataset_name,
        'result_path': f'{result_root_folder}/{dataset_name}/', # neurips_example_3_past_8, codebench_example_3_past_8
        'sim_strategy': 'standard', # ['standard', 'cot', 'pot'] modulation/trend/scale: change order or agent number --> test cases: m, t, s, mt, ts, ms, mts. We already have none and mts.
        'gpt_type': 3, # [0,1,2,3,4], 1:llama3-8b, 2:llama3-70b, 3:gpt4o-mini, 4:gpt4o, 5:gpt3.5, 6:gpt4, 7:, 8:
        'behavior_column_list': behavior_column_list_dict[dataset_name],
        'print_log': False,
    }

    epoch_num = epoch_num_dict[dataset_name]
    batch_size = int(np.ceil(dataset_user_total_dict[dataset_name]/epoch_num))
    # batch_size = 1
    for dataset_name in ['timecare']: # 'timecare', 'bio', 'write'
        for gpt_type_value in [3]:
            simulation_config['gpt_type'] = gpt_type_value
            simulation_config['dataset_path'] = dataset_dict[dataset_name][0]
            simulation_config['dataset_type'] = dataset_name
            simulation_config['behavior_column_list'] = behavior_column_list_dict[dataset_name]
            simulation_config['result_path'] = f'{result_root_folder}/{dataset_name}/'
            for sim_model in ['self-refine']:
                simulation_config['sim_strategy'] = sim_model
                for i in range(epoch_num):
                    # if i in [0,1,2,3,4,5,6,7,8,9] and gpt_type_value == 1: continue
                    # if i <= 14 and gpt_type_value == 8: continue
                    t1 = time.time()
                    print(i,sim_model)
                    # asyncio.run(run_exp(simulation_config,i*2,(i+1)*2))
                    asyncio.run(run_exp_async(simulation_config,i*batch_size,(i+1)*batch_size))
                    print('one epoch time: ',time.time()-t1)

if __name__ == "__main__":
    exp_main()

