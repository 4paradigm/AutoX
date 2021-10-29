import os, sys, time, datetime


def get_run_sh(hdfs_env, hdfs_input, path_src, time_str, app_name, mem):
    if time_str is None:
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name_env = os.path.basename(hdfs_env)
    name_input = os.path.basename(hdfs_input)
    name_output = 'output_{}'.format(time_str)

    path_input = '../input/{}'.format(name_input)
    path_output = '../{}'.format(name_output)

    # python_cmd = 'python -u main.py {} {} | tee {}/log.txt'.format(path_input, path_output, path_output)
    # python_cmd = 'python -u fengdian_ensemble.py {} -1 {} | tee {}/log.txt'.format(path_output, path_input, path_output)
    python_cmd = f'python -u run.py {path_input} {path_output} | tee {path_output}/kaggle_house_price.log'

    cmd_hdfs_get = 'hdfs dfs -get {}/{}/ .'.format(hdfs_input, name_output)
    file = open('cmd_hdfs_get.sh', 'a')
    file.write(cmd_hdfs_get)
    file.write('\n')
    file.close()

    s = []
    s += ['# !/bin/bash']
    s += ['pwd']
    s += ['echo "[+] app_name = \"{}\""'.format(app_name)]
    s += ['echo "[+] mem = \"{}\""'.format(mem)]
    s += ['echo "[+] time_str = \"{}\""'.format(time_str)]
    s += ['echo "[+] hdfs_env = \"{}\""'.format(hdfs_env)]
    s += ['echo "[+] hdfs_input = \"{}\""'.format(hdfs_input)]
    s += ['echo "[+] path_src = \"{}\""'.format(path_src)]
    s += ['echo "[+] name_env = \"{}\""'.format(name_env)]
    s += ['echo "[+] name_input = \"{}\""'.format(name_input)]
    s += ['echo "[+] name_output = \"{}\""'.format(name_output)]
    s += ['echo "[+] path_input = \"{}\""'.format(path_input)]
    s += ['echo "[+] path_output = \"{}\""'.format(path_output)]

    s += ['']
    s += ['# evn']
    s += ['echo "[+] evn......"']
    s += ['mkdir ./env']
    s += ['hdfs dfs -get {}'.format(hdfs_env)]
    s += ['tar -xzf {} -C ./env'.format(name_env)]  # 不要-v, 会打印很多子目录
    s += ['export PATH=./env/bin:$PATH']
    s += ['which python']
    s += ['export PYTHONPATH=.:$PYTHONPATH']
    s += ['export PATH=../env/bin:$PATH']  # export需要绝对路径, 如果是相对路径, cd之后, 就错了

    s += ['']
    s += ['# input']
    s += ['echo "[+] input......"']
    s += ['mkdir ./input']
    s += ['hdfs dfs -get {} ./input/'.format(hdfs_input)]

    s += ['']
    s += ['# src']
    s += ['echo "[+] src......"']
    s += ['tar -xzf {}'.format(path_src)]

    s += ['']
    s += ['# output ']
    s += ['echo "[+] output......"']
    s += ['mkdir {}'.format(name_output)]
    s += ['hdfs dfs -put -f -p {}/ {}/'.format(name_output, hdfs_input)]
    s += ['hdfs dfs -put -f autox.tar.gz {}/{}/'.format(hdfs_input, name_output)]
    s += ['hdfs dfs -put -f run.sh {}/{}/'.format(hdfs_input, name_output)]
    s += ['hdfs dfs -put -f yarn.sh {}/{}/'.format(hdfs_input, name_output)]
    s += ['hdfs dfs -put -f submit.py {}/{}/'.format(hdfs_input, name_output)]

    s += ['']
    s += ['# run']
    s += ['echo "[+] run......"']
    s += ['pwd']
    s += ['cd autox/']
    s += ['which python']
    s += ['pwd']
    s += ['echo "{}"'.format(python_cmd)]
    s += [python_cmd]
    s += ['pwd']
    s += ['cd ../']
    s += ['pwd']

    s += ['']
    s += ['# save']
    s += ['echo "[+] save......"']
    s += ['hdfs dfs -put -f {}/* {}/{}/'.format(name_output, hdfs_input, name_output)]
    s += ['echo "finised..."']
    return '\n'.join(s)


def get_yarn_sh(appname, mem=65536, queue='pico'):
    # mem(MB), 65536MB = 64GB
    s = []
    s += ['# !/bin/bash']
    s += ['yarn jar ./hadoop-patch.jar com.tfp.hadoop.yarn.launcher.Client \\']
    s += ['  --appname {} --jar ./hadoop-patch.jar \\'.format(appname)]
    s += ['  --shell_command "./run.sh " \\']
    s += ['  --queue {} \\'.format(queue)]
    s += ['  --container_memory={} \\'.format(mem)]
    s += ['  --num_containers=1 \\']
    s += ['  --shell_env HADOOP_USER_NAME=`whoami`\\']
    s += ['  --shell_env WEBHDFS_USER=`whoami` \\']
    s += ['  --file autox.tar.gz \\']
    s += ['  --file submit.py \\']
    s += ['  --file run.sh \\']
    s += ['  --file yarn.sh']
    return "\n".join(s)


def run_task(hdfs_env, hdfs_input, path_src, app_name, mem, time_str):
    print('[+] time_str = "{}"'.format(time_str))
    print('[+] hdfs_env = "{}"'.format(hdfs_env))
    print('[+] hdfs_input = "{}"'.format(hdfs_input))
    print('[+] path_src = "{}"'.format(path_src))
    print('[+] app_name = "{}"'.format(app_name))
    print('[+] mem = "{}"'.format(mem))

    s_run = get_run_sh(hdfs_env, hdfs_input, path_src, time_str, app_name, mem)
    print('\n\n# s_run = {}'.format('-' * 90))
    print(s_run)
    with open('./run.sh', 'w') as f:
        f.write(s_run)

    s_yarn = get_yarn_sh(app_name, mem)
    print('\n\n# s_yarn = {}'.format('-' * 90))
    print(s_yarn)
    with open('./yarn.sh', 'w') as f:
        f.write(s_yarn)

    # cmd = 'nohup bash yarn_run.sh > log/log_{}_{}.txt 2>&1 &'.format(app_name, time_str)
    cmd = 'bash yarn.sh'
    print(cmd)
    os.system(cmd)
    time.sleep(5)


import re


def get_tar_cmd(path_src):
    if '/' not in path_src:
        p2 = path_src
        cmd = 'tar -czf {}.tar.gz {}/'.format(p2, p2)
    else:
        p1 = re.sub(r'(.*)/(.+)', r'\1/', path_src)
        p2 = re.sub(r'(.*)/(.+)', r'\2', path_src)
        cmd = 'cd {} && tar -czf {}.tar.gz {}/ && cd - && mv {}{}.tar.gz ./'. \
            format(p1, p2, p2, p1, p2)
    return cmd, p2


def tar_src(path_src):
    print('[+] tar {}'.format(path_src))
    tar_cmd, path_src = get_tar_cmd(path_src)
    print('[+] tar_cmd = {}'.format(tar_cmd))
    os.system(tar_cmd)
    path_src += '.tar.gz'
    return path_src


path_src = 'autox'
path_src = tar_src(path_src)
if path_src is None:
    print('[+] src not exist!')
else:
    hdfs_env = '/user/caihengxing/python371028.tar.gz'
    hdfs_input = '/user/caihengxing/autox_input'
    app_name = 'kaggle_house_price'
    mem = 100000  # 单位:M
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    app_name = '{}_{}_{}'.format(app_name, os.path.basename(hdfs_input), time_str)
    run_task(hdfs_env, hdfs_input, path_src, app_name, mem, time_str)

