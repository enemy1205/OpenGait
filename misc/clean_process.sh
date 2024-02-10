kill $(ps aux | grep main_re.py | grep -v grep | awk '{print $2}')
