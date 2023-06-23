import streamlit as st
import subprocess
import threading
import queue
import time
import boto3

def list_folders(bucket_name, parent_folder):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(
        Bucket=bucket_name,
        Prefix=parent_folder,
        Delimiter='/'
    )

    folders = []
    for content in response.get('CommonPrefixes', []):
        folders.append(content.get('Prefix'))

    return folders


def stream_command(command):
    q_stdout = queue.Queue()
    q_stderr = queue.Queue()

    def _stream_command(command, q_stdout, q_stderr):
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        while True:
            output = process.stdout.readline()
            if output:
                q_stdout.put(output.strip())
            error = process.stderr.readline()
            if error:
                q_stderr.put(error.strip())
            if output == b'' and error == b'' and process.poll() is not None:
                break
        process.poll()

    thread = threading.Thread(target=_stream_command, args=(command, q_stdout, q_stderr))
    thread.start()

    # st.write('Command output:')
    log = st.empty()
    while thread.is_alive() or not q_stdout.empty() or not q_stderr.empty():
        while not q_stdout.empty():
            log.write(q_stdout.get().decode())
        while not q_stderr.empty():
            log.warning(q_stderr.get().decode()) # warnings are displayed in red in Streamlit
        time.sleep(0.1)

def run_script(script_path):
    result = subprocess.Popen(['bash', script_path])

    # you can also wait for the script to finish with:
    result.wait()

    return result

st.title("LLM as a Service ☁️🚀")

with st.expander("Text Generation"):
    model_tg_original = st.selectbox(
    'Choose Model',
    ('EleutherAI/pythia-12b', 'EleutherAI/pythia-70m', 'tiiuae/falcon-40b', 'tiiuae/falcon-7b'),
    key="tg-model")

    model_tg_original_mod = model_tg_original.replace("/", "\/" )

    # model_tg = model_tg_original.replace("/", "-" )
    model_tg = model_tg_original.split('/')[1]
    model_tg = model_tg.replace(".", "-" )
    model_tg = model_tg.lower()
    model_tg = (model_tg[:45]) if len(model_tg) > 45 else model_tg

    model_version_tg = st.radio(
    "Choose Model Version",
    ('Base', 'Finetuned'),
    key='mv-tg')

    if model_version_tg == 'Base':
        download_base = st.checkbox('Download base', key='dl-base-tg')
        model_version_tg_input = 'base'

    if model_version_tg == 'Finetuned':

        folders = list_folders('paradigm-llm-models', model_tg_original + '/')
        # print(f"Folders = {folders}")
        folders = [i for i in folders if 'finetune' in i]
        folders_filtered = []
        for f in folders:
            folders_filtered.append(f.split(model_tg_original)[1].split('/')[1])
        # print(f"Folders = {folders_filtered}")
        model_version_tg_input = st.selectbox(
            'Choose Finetuned Model',
            folders_filtered,
            key="ftm-version-tg")

        # model_version_tg_input = st.text_input('Enter fintuned model version', key='ftm-version-tg')

    memory_requested = st.number_input('Request memory (Gi)', min_value=2, max_value=1000, key='momery-tg')

    if st.button("Deploy", key="deploy-tg"):
        if model_version_tg == 'Base':
            if download_base:
                with st.spinner('Unleashing the agents...'):
                    bash_commands = []
                    bash_commands.append("rm -r last-deployed-scripts")
                    bash_commands.append("mkdir last-deployed-scripts")
                    bash_commands.append("cp text-generation-utils/general-download-base.py last-deployed-scripts/general-download-base.py")
                    bash_commands.append("cp text-generation-utils/general-text-generation.py last-deployed-scripts/general-text-generation.py")
                    bash_commands.append("cp text-generation-utils/requirements.general-download-base last-deployed-scripts/requirements.general-download-base")
                    bash_commands.append("cp text-generation-utils/requirements.general-text-generation last-deployed-scripts/requirements.general-text-generation")

                    bash_commands.append("cd last-deployed-scripts")
                    bash_commands.append(f"mv general-download-base.py {model_tg}-download-base.py")
                    bash_commands.append(f"mv general-text-generation.py {model_tg}-text-generation.py")
                    bash_commands.append(f"mv requirements.general-download-base requirements.{model_tg}-download-base")
                    bash_commands.append(f"mv requirements.general-text-generation requirements.{model_tg}-text-generation")

                    bash_commands.append(f"sed -i 's/<MODELNAME>/{model_tg_original_mod}/g' {model_tg}-download-base.py")
                    bash_commands.append(f"sed -i 's/<MODELNAME>/{model_tg_original_mod}/g' {model_tg}-text-generation.py")
                    bash_commands.append(f"sed -i 's/<MODELVERSION>/{model_version_tg_input}/g' {model_tg}-text-generation.py")
                    
                    bash_commands.append(f"paradigm launch --steps {model_tg}-download-base {model_tg}-text-generation")

                    bash_commands.append(f'paradigm deploy --steps {model_tg}-download-base --dependencies "{model_tg}-text-generation:{model_tg}-download-base" --deployment {model_tg}-text-generation --deployment_port 8000 --deployment_memory {int(memory_requested)}Gi')

                    with open('latest_instructions.sh', 'w') as f:
                        for item in bash_commands:
                            f.write(f"{item}\n")
                    
                    run_script("latest_instructions.sh")
                    # deployment_name = logs.split('deployment.apps/')[1].split(' ')[0]
                    # print(f"***Deployment found - {deployment_name}")

            else:
                with st.spinner('Unleashing the agents..'):
                    bash_commands = []
                    bash_commands.append("rm -r last-deployed-scripts")
                    bash_commands.append("mkdir last-deployed-scripts")
                    bash_commands.append("cp text-generation-utils/general-text-generation.py last-deployed-scripts/general-text-generation.py")
                    bash_commands.append("cp text-generation-utils/requirements.general-text-generation last-deployed-scripts/requirements.general-text-generation")

                    bash_commands.append("cd last-deployed-scripts")
                    bash_commands.append(f"mv general-text-generation.py {model_tg}-text-generation.py")
                    bash_commands.append(f"mv requirements.general-text-generation requirements.{model_tg}-text-generation")

                    bash_commands.append(f"sed -i 's/<MODELNAME>/{model_tg_original_mod}/g' {model_tg}-text-generation.py")
                    bash_commands.append(f"sed -i 's/<MODELVERSION>/{model_version_tg_input}/g' {model_tg}-text-generation.py")
                
                    bash_commands.append(f"paradigm launch --steps {model_tg}-text-generation")
                    bash_commands.append(f"paradigm deploy --deployment {model_tg}-text-generation --deployment_port 8000 --deployment_memory {int(memory_requested)}Gi")

                    with open('latest_instructions.sh', 'w') as f:
                        for item in bash_commands:
                            f.write(f"{item}\n")
                    
                    run_script("latest_instructions.sh")
                    # deployment_name = logs.split('deployment.apps/')[1].split(' ')[0]
                    # print(f"***Deployment found - {deployment_name}")

        elif model_version_tg == 'Finetuned':
            with st.spinner('Unleashing the agents..'):
                bash_commands = []
                bash_commands.append("rm -r last-deployed-scripts")
                bash_commands.append("mkdir last-deployed-scripts")
                bash_commands.append("cp text-generation-utils/general-text-generation.py last-deployed-scripts/general-text-generation.py")
                bash_commands.append("cp text-generation-utils/requirements.general-text-generation last-deployed-scripts/requirements.general-text-generation")

                bash_commands.append("cd last-deployed-scripts")
                bash_commands.append(f"mv general-text-generation.py {model_tg}-text-generation.py")
                bash_commands.append(f"mv requirements.general-text-generation requirements.{model_tg}-text-generation")

                bash_commands.append(f"sed -i 's/<MODELNAME>/{model_tg_original_mod}/g' {model_tg}-text-generation.py")
                bash_commands.append(f"sed -i 's/<MODELVERSION>/{model_version_tg_input}/g' {model_tg}-text-generation.py")
            
                bash_commands.append(f"paradigm launch --steps {model_tg}-text-generation")
                bash_commands.append(f"paradigm deploy --deployment {model_tg}-text-generation --deployment_port 8000 --deployment_memory {int(memory_requested)}Gi")

                with open('latest_instructions.sh', 'w') as f:
                    for item in bash_commands:
                        f.write(f"{item}\n")
                
                run_script("latest_instructions.sh")
                # deployment_name = logs.split('deployment.apps/')[1].split(' ')[0]
                # print(f"***Deployment found - {deployment_name}")



with st.expander("Conversational"):
    model_con_original = st.selectbox(
    'Choose Model',
    ('OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5','tiiuae/falcon-40b-instruct', 'tiiuae/falcon-7b-instruct', 'mosaicml/mpt-7b-instruct',
            'mosaicml/mpt-7b-chat', 'databricks/dolly-v2-12b'),
    key="con-model")

    model_con_original_mod = model_con_original.replace("/", "\/" )

    # model_con = model_con_original.replace("/", "-" )
    model_con = model_con_original.split('/')[1]
    model_con = model_con.replace(".", "-" )
    model_con = model_con.lower()
    model_con = (model_con[:45]) if len(model_con) > 45 else model_con

    model_version_con = st.radio(
    "Choose Model Version",
    ('Base', 'Finetuned'),
    key='mv-con')

    if model_version_con == 'Base':
        download_base = st.checkbox('Download base', key='dl-base-con')
        model_version_con_input = 'base'

    if model_version_con == 'Finetuned':
        folders = list_folders('paradigm-llm-models', model_con_original + '/')
        # print(f"Folders = {folders}")
        folders = [i for i in folders if 'finetune' in i]
        folders_filtered = []
        for f in folders:
            folders_filtered.append(f.split(model_con_original)[1].split('/')[1])
        # print(f"Folders = {folders_filtered}")
        model_version_con_input = st.selectbox(
            'Choose Finetuned Model',
            folders_filtered,
            key="ftm-version-con")

    memory_requested = st.number_input('Request memory (Gi)', min_value=2, max_value=1000, key='momery-con')

    if st.button("Deploy", key="deploy-con"):
        if model_version_con == 'Base':
            if download_base:
                with st.spinner('Unleashing the agents...'):
                    bash_commands = []
                    bash_commands.append("rm -r last-deployed-scripts")
                    bash_commands.append("mkdir last-deployed-scripts")
                    bash_commands.append("cp conversational-utils/general-download-base.py last-deployed-scripts/general-download-base.py")
                    bash_commands.append("cp conversational-utils/general-conversational.py last-deployed-scripts/general-conversational.py")
                    bash_commands.append("cp conversational-utils/requirements.general-download-base last-deployed-scripts/requirements.general-download-base")
                    bash_commands.append("cp conversational-utils/requirements.general-conversational last-deployed-scripts/requirements.general-conversational")

                    bash_commands.append("cd last-deployed-scripts")
                    bash_commands.append(f"mv general-download-base.py {model_con}-download-base.py")
                    bash_commands.append(f"mv general-conversational.py {model_con}-conversational.py")
                    bash_commands.append(f"mv requirements.general-download-base requirements.{model_con}-download-base")
                    bash_commands.append(f"mv requirements.general-conversational requirements.{model_con}-conversational")

                    bash_commands.append(f"sed -i 's/<MODELNAME>/{model_con_original_mod}/g' {model_con}-download-base.py")
                    bash_commands.append(f"sed -i 's/<MODELNAME>/{model_con_original_mod}/g' {model_con}-conversational.py")
                    bash_commands.append(f"sed -i 's/<MODELVERSION>/{model_version_con_input}/g' {model_con}-conversational.py")
                    
                    bash_commands.append(f"paradigm launch --steps {model_con}-download-base {model_con}-conversational")

                    bash_commands.append(f'paradigm deploy --steps {model_con}-download-base --dependencies "{model_con}-conversational:{model_con}-download-base" --deployment {model_con}-conversational --deployment_port 8000 --deployment_memory {int(memory_requested)}Gi')

                    with open('latest_instructions.sh', 'w') as f:
                        for item in bash_commands:
                            f.write(f"{item}\n")
                    
                    run_script("latest_instructions.sh")
                    # deployment_name = logs.split('deployment.apps/')[1].split(' ')[0]
                    # print(f"***Deployment found - {deployment_name}")

            else:
                with st.spinner('Unleashing the agents..'):
                    bash_commands = []
                    bash_commands.append("rm -r last-deployed-scripts")
                    bash_commands.append("mkdir last-deployed-scripts")
                    bash_commands.append("cp conversational-utils/general-conversational.py last-deployed-scripts/general-conversational.py")
                    bash_commands.append("cp conversational-utils/requirements.general-conversational last-deployed-scripts/requirements.general-conversational")

                    bash_commands.append("cd last-deployed-scripts")
                    bash_commands.append(f"mv general-conversational.py {model_con}-conversational.py")
                    bash_commands.append(f"mv requirements.general-conversational requirements.{model_con}-conversational")

                    bash_commands.append(f"sed -i 's/<MODELNAME>/{model_con_original_mod}/g' {model_con}-conversational.py")
                    bash_commands.append(f"sed -i 's/<MODELVERSION>/{model_version_con_input}/g' {model_con}-conversational.py")
                
                    bash_commands.append(f"paradigm launch --steps {model_con}-conversational")
                    bash_commands.append(f"paradigm deploy --deployment {model_con}-conversational --deployment_port 8000 --deployment_memory {int(memory_requested)}Gi")

                    with open('latest_instructions.sh', 'w') as f:
                        for item in bash_commands:
                            f.write(f"{item}\n")
                    
                    run_script("latest_instructions.sh")
                    # deployment_name = logs.split('deployment.apps/')[1].split(' ')[0]
                    # print(f"***Deployment found - {deployment_name}")

        elif model_version_con == 'Finetuned':
            with st.spinner('Unleashing the agents..'):
                bash_commands = []
                bash_commands.append("rm -r last-deployed-scripts")
                bash_commands.append("mkdir last-deployed-scripts")
                bash_commands.append("cp conversational-utils/general-conversational.py last-deployed-scripts/general-conversational.py")
                bash_commands.append("cp conversational-utils/requirements.general-conversational last-deployed-scripts/requirements.general-conversational")

                bash_commands.append("cd last-deployed-scripts")
                bash_commands.append(f"mv general-conversational.py {model_con}-conversational.py")
                bash_commands.append(f"mv requirements.general-conversational requirements.{model_con}-conversational")

                bash_commands.append(f"sed -i 's/<MODELNAME>/{model_con_original_mod}/g' {model_con}-conversational.py")
                bash_commands.append(f"sed -i 's/<MODELVERSION>/{model_version_con_input}/g' {model_con}-conversational.py")
            
                bash_commands.append(f"paradigm launch --steps {model_con}-conversational")
                bash_commands.append(f"paradigm deploy --deployment {model_con}-conversational --deployment_port 8000 --deployment_memory {int(memory_requested)}Gi")

                with open('latest_instructions.sh', 'w') as f:
                    for item in bash_commands:
                        f.write(f"{item}\n")
                
                run_script("latest_instructions.sh")
                # deployment_name = logs.split('deployment.apps/')[1].split(' ')[0]
                # print(f"***Deployment found - {deployment_name}")

with st.expander("Code Completion"):
    model_code_original = st.selectbox(
    'Choose Model',
    ('bigcode/starcoder',),
    key="code-model")

    model_code_original_mod = model_code_original.replace("/", "\/" )

    # model_code = model_code_original.replace("/", "-" )
    model_code = model_code_original.split('/')[1]
    model_code = model_code.replace(".", "-" )
    model_code = model_code.lower()
    model_code = (model_code[:45]) if len(model_code) > 45 else model_code

    model_version_code = st.radio(
    "Choose Model Version",
    ('Base', 'Finetuned'),
    key='mv-code')

    if model_version_code == 'Base':
        download_base = st.checkbox('Download base', key='dl-base-code')
        model_version_code_input = 'base'

    if model_version_code == 'Finetuned':
        folders = list_folders('paradigm-llm-models', model_code_original + '/')
        # print(f"Folders = {folders}")
        folders = [i for i in folders if 'finetune' in i]
        folders_filtered = []
        for f in folders:
            folders_filtered.append(f.split(model_code_original)[1].split('/')[1])
        # print(f"Folders = {folders_filtered}")
        model_version_code_input = st.selectbox(
            'Choose Finetuned Model',
            folders_filtered,
            key="ftm-version-code")

    memory_requested = st.number_input('Request memory (Gi)', min_value=2, max_value=1000, key='momery-code')

    if st.button("Deploy", key="deploy-code"):
        if model_version_code == 'Base':
            if download_base:
                with st.spinner('Unleashing the agents...'):
                    bash_commands = []
                    bash_commands.append("rm -r last-deployed-scripts")
                    bash_commands.append("mkdir last-deployed-scripts")
                    bash_commands.append("cp code-completion-utils/general-download-base.py last-deployed-scripts/general-download-base.py")
                    bash_commands.append("cp code-completion-utils/general-code-completion.py last-deployed-scripts/general-code-completion.py")
                    bash_commands.append("cp code-completion-utils/requirements.general-download-base last-deployed-scripts/requirements.general-download-base")
                    bash_commands.append("cp code-completion-utils/requirements.general-code-completion last-deployed-scripts/requirements.general-code-completion")

                    bash_commands.append("cd last-deployed-scripts")
                    bash_commands.append(f"mv general-download-base.py {model_code}-download-base.py")
                    bash_commands.append(f"mv general-code-completion.py {model_code}-code-completion.py")
                    bash_commands.append(f"mv requirements.general-download-base requirements.{model_code}-download-base")
                    bash_commands.append(f"mv requirements.general-code-completion requirements.{model_code}-code-completion")

                    bash_commands.append(f"sed -i 's/<MODELNAME>/{model_code_original_mod}/g' {model_code}-download-base.py")
                    bash_commands.append(f"sed -i 's/<MODELNAME>/{model_code_original_mod}/g' {model_code}-code-completion.py")
                    bash_commands.append(f"sed -i 's/<MODELVERSION>/{model_version_code_input}/g' {model_code}-code-completion.py")
                    
                    bash_commands.append(f"paradigm launch --steps {model_code}-download-base {model_code}-code-completion")

                    bash_commands.append(f'paradigm deploy --steps {model_code}-download-base --dependencies "{model_code}-code-completion:{model_code}-download-base" --deployment {model_code}-code-completion --deployment_port 8000 --deployment_memory {int(memory_requested)}Gi')

                    with open('latest_instructions.sh', 'w') as f:
                        for item in bash_commands:
                            f.write(f"{item}\n")
                    
                    run_script("latest_instructions.sh")
                    # deployment_name = logs.split('deployment.apps/')[1].split(' ')[0]
                    # print(f"***Deployment found - {deployment_name}")

            else:
                with st.spinner('Unleashing the agents..'):
                    bash_commands = []
                    bash_commands.append("rm -r last-deployed-scripts")
                    bash_commands.append("mkdir last-deployed-scripts")
                    bash_commands.append("cp code-completion-utils/general-code-completion.py last-deployed-scripts/general-code-completion.py")
                    bash_commands.append("cp code-completion-utils/requirements.general-code-completion last-deployed-scripts/requirements.general-code-completion")

                    bash_commands.append("cd last-deployed-scripts")
                    bash_commands.append(f"mv general-code-completion.py {model_code}-code-completion.py")
                    bash_commands.append(f"mv requirements.general-code-completion requirements.{model_code}-code-completion")

                    bash_commands.append(f"sed -i 's/<MODELNAME>/{model_code_original_mod}/g' {model_code}-code-completion.py")
                    bash_commands.append(f"sed -i 's/<MODELVERSION>/{model_version_code_input}/g' {model_code}-code-completion.py")
                
                    bash_commands.append(f"paradigm launch --steps {model_code}-code-completion")
                    bash_commands.append(f"paradigm deploy --deployment {model_code}-code-completion --deployment_port 8000 --deployment_memory {int(memory_requested)}Gi")

                    with open('latest_instructions.sh', 'w') as f:
                        for item in bash_commands:
                            f.write(f"{item}\n")
                    
                    run_script("latest_instructions.sh")
                    # deployment_name = logs.split('deployment.apps/')[1].split(' ')[0]
                    # print(f"***Deployment found - {deployment_name}")

        elif model_version_code == 'Finetuned':
            with st.spinner('Unleashing the agents..'):
                bash_commands = []
                bash_commands.append("rm -r last-deployed-scripts")
                bash_commands.append("mkdir last-deployed-scripts")
                bash_commands.append("cp code-completion-utils/general-code-completion.py last-deployed-scripts/general-code-completion.py")
                bash_commands.append("cp code-completion-utils/requirements.general-code-completion last-deployed-scripts/requirements.general-code-completion")

                bash_commands.append("cd last-deployed-scripts")
                bash_commands.append(f"mv general-code-completion.py {model_code}-code-completion.py")
                bash_commands.append(f"mv requirements.general-code-completion requirements.{model_code}-code-completion")

                bash_commands.append(f"sed -i 's/<MODELNAME>/{model_code_original_mod}/g' {model_code}-code-completion.py")
                bash_commands.append(f"sed -i 's/<MODELVERSION>/{model_version_code_input}/g' {model_code}-code-completion.py")
            
                bash_commands.append(f"paradigm launch --steps {model_code}-code-completion")
                bash_commands.append(f"paradigm deploy --deployment {model_code}-code-completion --deployment_port 8000 --deployment_memory {int(memory_requested)}Gi")

                with open('latest_instructions.sh', 'w') as f:
                    for item in bash_commands:
                        f.write(f"{item}\n")
                
                run_script("latest_instructions.sh")