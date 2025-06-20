
def store_log(input_string,log_path,color=None, attrs=None, print=False):
    with open(log_path, 'a') as f:
        f.write(input_string + '\n')
        f.flush()
    if(print):
        cprint(input_string, color=color, attrs=attrs)

def response_llm_gpt_aiml(message_list,backbone_type,temperature,timeout=120):
    # 5:gpt3.5, 6:gpt4, 7:gpt-o1, 8:gpt-o1 mini
    response = ''
    except_waiting_time = 1
    max_waiting_time = 32
    current_sleep_time = 0.5

    start_time = time.time()
    while response == '':
        try:
            completion = client.chat.completions.create(
                model=model_name, 
                messages=message_list,
                temperature=0,
                timeout = timeout,
                max_tokens=max_tokens
                )
            
            response = completion.choices[0].message.content
        except Exception as e:
            print('error in _response_llm_gpt',e)
            time.sleep(current_sleep_time)
            if except_waiting_time < max_waiting_time:
                except_waiting_time *= 2
            current_sleep_time = np.random.randint(0, except_waiting_time-1)
    end_time = time.time()   
    # print('llm response time: ',end_time-start_time)     
    return response


class Metric_Evaluation_Agent(object):
    def __init__(self,backbone_type,log_folder,print_log,temperature):
        self.backbone_type = backbone_type
        self.agent_name = 'metric_evaluation_agent'
        self.log_file = f'{log_folder}/{self.agent_name}.txt'
        self.print_log = print_log
        self.temperature = temperature
        self.task = task_prompt_template_dict[self.agent_name]
        self.sys_prompt = sys_prompt_template_dict[self.agent_name]

        self.sys_message = {"role": "system","content": self.sys_prompt}
        store_log(f'{self.agent_name} agent: system: \n {self.sys_prompt}\n',self.log_file,color = 'white',print = self.print_log)




    def evaluate(self,creativity_task,metric,user_answer):
        user_prompt = f'{self.task}\n\nHere is the creativity task:\n{creativity_task}\n\nHere is the metric:\n{metric}\n\nHere is the user response for evaluation:\n #User Response#: {user_answer}\n\n{output_format_template_dict[self.agent_name]}'
        message_list = [self.sys_message,{"role": "user","content": user_prompt}]
        t1 = time.time()
        llm_response = response_llm_gpt(message_list,self.backbone_type,self.temperature)
        
        store_log(f'{self.agent_name} agent: user: \n {user_prompt}\n',self.log_file,color = 'green',print = self.print_log)
        store_log(f'{self.agent_name} agent: assistant (resptime: {time.time()-t1}): \n {llm_response}\n',self.log_file,color = 'red',print = self.print_log)

        self.evaluation_dict = self.extract_response_pattern(llm_response)

        return self.evaluation_dict 

    def extract_response_pattern(self,response):
        try:
            evaluation_data = json_repair.loads(response.strip('```json').strip('```'))
            assert (isinstance(evaluation_data, list) or isinstance(evaluation_data, dict)), "Parsed response is not a list or dict"
            if isinstance(evaluation_data, dict): evaluation_data = [evaluation_data]
            return evaluation_data
           
        except Exception as e:
            print("Error parsing LLM response:", e)
            print("Raw output:\n", response)

sys_prompt_template_dict = {
    'thought_evaluation_agent': 'You are an intelligent assistant who is good at evaluating and analyzing how user response is affected by textual feedback in creativity tasks.',
}

task_prompt_template_dict = {

    'causal_effect_agent': '''

'''}


output_format_template_dict = {
    
    'natural_judge_agent': '''
Your output should be in json format including a dictionary, like exactly below:
{
    "judgement": "<yes or no>",
    "refined_hypothesis": "<the refined hypothesis or raw hypothesis if you think it cannot be refined>",
    "reason": "<your 1-2 sentence reason why you have such refinement or why you think it cannot be refined>"
}
}
