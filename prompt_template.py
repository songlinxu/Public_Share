
simulation_strategy_cot_str = '''
You should follow the step-by-step simulation strategy below:

**Step-by-step Reasoning Process**: 
[step 1]: (the first step for solving this simulation task)
.....
[step n]: (the n-th step for solving the simulation task, if any)

Here is one example step-by-step reasoning, but you can further extend it based on the suitable scenario:

[step 1]: analyzing the math question difficulty.
[step 2]: connecting the future math question with the student past math question performance history.
[step 3]: make simulations.

'''

simulation_strategy_pot_str = '''
You should follow the simulation strategy below:

First, perform step-by-step reasoning like below:
**Step-by-step Reasoning Process**: 
[step 1]: (the first step for solving this task)
.....
[step n]: (the n-th step for solving the task, if any)

Here is one example step-by-step reasoning, but you can further extend it based on the suitable scenario:

[step 1]: analyzing the math question difficulty.
[step 2]: connecting the future math question with the student past math question performance history.
[step 3]: make simulations.

Second, translate your reasoning into codes, like example below:
**Code Translation Process**
Based on the steps above, I generate the codes below to solve this task.
---
My codes here.
---

Third, execute your codes to get results, like example below:
**Code Execution Process**
Based on the codes above, I generate the answer conclusion below:

'''

simulation_strategy_str_dict_global = {'standard': '', 'cot': simulation_strategy_cot_str, 'self-refine': '', 'pot': simulation_strategy_pot_str, 'mcl': ''}

sys_prompt_template_dict = {
    'student_simulator_agent': 'You are an intelligent assistant which is good at mimicking a real student to answer math questions based on its unique persona and past math question performance history.'
}

task_prompt_template_dict = {
    'student_simulator_agent': '''
A student is answering math questions. I will give you the student education background and past math question answering performance (accuracy and response time).

Each math trial contains three numbers, and the student needs to make a binary choice to decide whether the difference between the first two numbers is divisible by the third number.

Math Questions Examples:
For the trial 49 ≡ 20 (mod 5): Calculate 49 - 20 = 29, then determine if 29 is divisible by 5. Select "No" because 29 ÷ 5 = 5.8 (not a whole number).
For the trial 51 ≡ 33 (mod 6): Calculate 51 - 33 = 18, then determine if 18 is divisible by 6. Select "Yes" because 18 ÷ 6 = 3 (whole number).

For each math question, accuracy = 1 means that the student makes a correct selection; accuracy = 0 means that the student makes a wrong selection;  

The response time means how much time it take for the student to answer the corresonding math question.

Based on the student unique educational background and past performance history, you will simulate the student behavior in answering a few future math questions. 
''',
}

output_format_template_dict = { 
    'student_simulator_agent': '''
Your output should be in json format to include a dictionary, like exactly below:
    {
        "trial_id_global": "<trial id>",
        "user_accuracy": "<user accuracy: either 0 or 1>",
        "response_time": "<user response time in seconds>",
        "reasoning": "<your reasoning for the accuracy and response time>"
    }

Instructions:
Keep the JSON structure exactly as above.
Do not output any additional text outside of this JSON.

Below is one toy example to format your output:

    {
        "trial_id_global": "4",
        "user_accuracy": "0",
        "response_time": "6.78",
        "reasoning": "This trial was a bit difficult for the student, based on its history performance."
    }

Do NOT output any other information outside the json format data.

''',
    'student_simulator_agent_multi': '''
Your output should be in json format using a list to include dictionaries, like exactly below:
[
    {
        "trial_id_global": "<trial id>",
        "user_accuracy": "<user accuracy: either 0 or 1>",
        "response_time": "<user response time in seconds>",
        "reasoning": "<your reasoning for the accuracy and response time>"
    },
    {
        "trial_id_global": "<trial id>",
        "user_accuracy": "<user accuracy: either 0 or 1>",
        "response_time": "<user response time in seconds>",
        "reasoning": "<your reasoning for the accuracy and response time>"
    }
]

Instructions:
Keep the JSON structure exactly as above.
Do not output any additional text outside of this JSON.

Below is one toy example to format your output:

[
    {
        "trial_id_global": "4",
        "user_accuracy": "0",
        "response_time": "6.78",
        "reasoning": "This trial was a bit difficult for the student, based on its history performance."
    },
    {
        "trial_id_global": "5",
        "user_accuracy": "1",
        "response_time": "5.45",
        "reasoning": "This trial was a bit easy for the student, based on its history performance."
    }
]


''',
    

}
